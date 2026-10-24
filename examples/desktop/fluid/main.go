// Command fluid fills the bottom half of a window with water: smoothed
// particle hydrodynamics on the GPU, 16,384 particles by default. Every
// substep is seven compute dispatches: predict where the particles go, count
// them into a grid with atomics, sum the counts into where each cell starts
// (two dispatches), sort the particles into cell order, sum each one's
// density over the cells around it, then apply pressure and viscosity and
// move them. The sort keeps neighbors next to each other in memory, which
// the water would otherwise shuffle as it moves. A render pipeline draws
// the same buffer as soft discs.
//
// Hold the left mouse button to pull the water to the pointer and the right
// one to blast it away.
//
//	cd examples/desktop
//	go run ./fluid [-n 16384] [-backend vulkan|dx12|metal|gl] [-frames 300]
//
// -frames exits after that many frames, a smoke test that walks the whole
// setup and teardown. -backend gl needs a context with compute shaders,
// desktop GL 4.3 or GLES 3.1.
package main

import (
	"encoding/binary"
	"flag"
	"fmt"
	"log"
	"math"
	"math/rand/v2"

	"github.com/go-gl/glfw/v3.4/glfw"
	wgpu "github.com/roffe/gowgpu"
	"github.com/roffe/gowgpu/examples/desktop/window"
	"github.com/roffe/gowgpu/gputypes"
)

const (
	particleBytes = 16 // pos: vec2<f32>, vel: vec2<f32>
	workgroup     = 256
	maxCells      = workgroup * workgroup // what the two scan dispatches sum
	maxParticles  = 1 << 17               // keeps the grid within maxCells
)

// params mirrors Params in the shaders. The box reaches bounds either side
// of the origin, so the window is 2 units high, and the water starts in the
// bottom half of it.
type params struct {
	Mouse         [2]float32 // the pointer, in box units
	Bounds        [2]float32 // the box's half extents
	Frame         [2]float32 // the window in pixels
	Dt            float32    // of one substep
	H             float32    // the smoothing radius, and the grid's cell size
	Rest          float32    // the density of water at rest
	Stiffness     float32    // pressure per unit of density over Rest
	NearStiffness float32    // of the short-range pressure that keeps particles apart
	Viscosity     float32
	Gravity       float32
	Force         float32 // of the pointer: pulls when positive, pushes when negative
	Radius        float32 // a particle's drawn radius, and how far it stays from the walls
	Count         uint32
	GridW, GridH  uint32
	_             [2]float32
}

// typesWGSL opens every shader.
const typesWGSL = `
const PI: f32 = 3.14159265;
const MOUSE_RADIUS: f32 = 0.4;

struct Particle { pos: vec2<f32>, vel: vec2<f32> }

struct Params {
    mouse: vec2<f32>,
    bounds: vec2<f32>,
    frame: vec2<f32>,
    dt: f32,
    h: f32,
    rest: f32,
    stiffness: f32,
    near_stiffness: f32,
    viscosity: f32,
    gravity: f32,
    force: f32,
    radius: f32,
    count: u32,
    grid_w: u32,
    grid_h: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
`

// funcsWGSL follows typesWGSL in the compute shaders: the grid and the kernels.
const funcsWGSL = `
// The grid has cells of h over the box, so a particle's neighbors within h
// are in its cell and the eight around it.
fn cell_of(p: vec2<f32>) -> vec2<i32> {
    let c = vec2<i32>(floor((p + params.bounds) / params.h));
    return clamp(c, vec2<i32>(0), vec2<i32>(i32(params.grid_w) - 1, i32(params.grid_h) - 1));
}

// cell_id is the place in counts and starts of the cell p is in.
fn cell_id(p: vec2<f32>) -> u32 {
    let c = cell_of(p);
    return u32(c.y) * params.grid_w + u32(c.x);
}

// Kernels over a distance r under h, each integrating to 1 over the plane,
// and the derivatives of the two that give pressure.
fn w_density(r: f32) -> f32 {
    let v = params.h - r;
    return v * v * 6.0 / (PI * pow(params.h, 4.0));
}

fn w_near(r: f32) -> f32 {
    let v = params.h - r;
    return v * v * v * 10.0 / (PI * pow(params.h, 5.0));
}

fn dw_density(r: f32) -> f32 {
    return -(params.h - r) * 12.0 / (PI * pow(params.h, 4.0));
}

fn dw_near(r: f32) -> f32 {
    let v = params.h - r;
    return -v * v * 30.0 / (PI * pow(params.h, 5.0));
}

fn w_viscosity(r: f32) -> f32 {
    let v = params.h * params.h - r * r;
    return v * v * v * 4.0 / (PI * pow(params.h, 8.0));
}
`

// predictWGSL applies gravity and the pointer to each particle's velocity
// and writes where it will be after the substep, which the next passes
// measure from. It also clears the grid.
const predictWGSL = `
@group(0) @binding(1) var<storage, read> particles: array<Particle>;
@group(0) @binding(2) var<storage, read_write> predicted: array<Particle>;
@group(0) @binding(3) var<storage, read_write> counts: array<atomic<u32>>;

@compute @workgroup_size(256)
fn predict(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = id.x;
    if (i < params.grid_w * params.grid_h) {
        atomicStore(&counts[i], 0u);
    }
    if (i >= params.count) {
        return;
    }
    let p = particles[i];
    var a = vec2<f32>(0.0, -params.gravity);
    let d = params.mouse - p.pos;
    let r = length(d);
    if (params.force != 0.0 && r > 0.0 && r < MOUSE_RADIUS) {
        let t = 1.0 - r / MOUSE_RADIUS;
        a += d / r * params.force * t;
        if (params.force > 0.0) {
            a -= p.vel * 3.0 * t; // pulling holds the water at the pointer
        }
    }
    let vel = p.vel + a * params.dt;
    predicted[i] = Particle(p.pos + vel * params.dt, vel);
}
`

// binWGSL counts the particles in each cell and gives each its rank there.
const binWGSL = `
@group(0) @binding(1) var<storage, read> predicted: array<Particle>;
@group(0) @binding(2) var<storage, read_write> counts: array<atomic<u32>>;
@group(0) @binding(3) var<storage, read_write> ranks: array<u32>;

@compute @workgroup_size(256)
fn bin(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = id.x;
    if (i >= params.count) {
        return;
    }
    ranks[i] = atomicAdd(&counts[cell_id(predicted[i].pos)], 1u);
}
`

// scanWGSL opens the two scan shaders: a prefix sum over a workgroup.
const scanWGSL = `
var<workgroup> partial: array<u32, 256>;

// scan returns the sum of v over this invocation and the ones before it.
fn scan(li: u32, v: u32) -> u32 {
    partial[li] = v;
    workgroupBarrier();
    for (var off = 1u; off < 256u; off *= 2u) {
        var add = 0u;
        if (li >= off) {
            add = partial[li - off];
        }
        workgroupBarrier();
        partial[li] += add;
        workgroupBarrier();
    }
    return partial[li];
}
`

// scanCellsWGSL sums the counts of each block of 256 cells into where each
// cell starts within the block, and each block's total into sums.
const scanCellsWGSL = `
@group(0) @binding(1) var<storage, read> counts: array<u32>;
@group(0) @binding(2) var<storage, read_write> starts: array<u32>;
@group(0) @binding(3) var<storage, read_write> sums: array<u32>;

@compute @workgroup_size(256)
fn scan_cells(@builtin(global_invocation_id) id: vec3<u32>, @builtin(local_invocation_index) li: u32, @builtin(workgroup_id) wg: vec3<u32>) {
    let cells = params.grid_w * params.grid_h;
    var v = 0u;
    if (id.x < cells) {
        v = counts[id.x];
    }
    let total = scan(li, v);
    if (id.x < cells) {
        starts[id.x] = total - v;
    }
    if (li == 255u) {
        sums[wg.x] = total;
    }
}
`

// scanBlocksWGSL turns the block totals into where each block starts, so a
// cell starts at starts[c] + sums[c / 256].
const scanBlocksWGSL = `
@group(0) @binding(1) var<storage, read_write> sums: array<u32>;

@compute @workgroup_size(256)
fn scan_blocks(@builtin(local_invocation_index) li: u32) {
    let blocks = (params.grid_w * params.grid_h + 255u) / 256u;
    var v = 0u;
    if (li < blocks) {
        v = sums[li];
    }
    let total = scan(li, v);
    if (li < blocks) {
        sums[li] = total - v;
    }
}
`

// scatterWGSL sorts the predicted particles, and where they were, into cell order.
const scatterWGSL = `
@group(0) @binding(1) var<storage, read> particles: array<Particle>;
@group(0) @binding(2) var<storage, read> predicted: array<Particle>;
@group(0) @binding(3) var<storage, read> ranks: array<u32>;
@group(0) @binding(4) var<storage, read> starts: array<u32>;
@group(0) @binding(5) var<storage, read> sums: array<u32>;
@group(0) @binding(6) var<storage, read_write> sorted: array<Particle>;
@group(0) @binding(7) var<storage, read_write> previous: array<vec2<f32>>;

@compute @workgroup_size(256)
fn scatter(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = id.x;
    if (i >= params.count) {
        return;
    }
    let c = cell_id(predicted[i].pos);
    let dest = starts[c] + sums[c / 256u] + ranks[i];
    sorted[dest] = predicted[i];
    previous[dest] = particles[i].pos;
}
`

// gridWGSL follows funcsWGSL in the shaders that read the sorted particles,
// which declare counts, starts and sums.
const gridWGSL = `
fn cell_start(c: u32) -> u32 {
    return starts[c] + sums[c / 256u];
}

// row is the range of sorted particles in the cell dy rows from home and the
// cells either side of it, which follow each other in cell order.
fn row(home: vec2<i32>, dy: i32) -> vec2<u32> {
    let y = home.y + dy;
    if (y < 0 || y >= i32(params.grid_h)) {
        return vec2<u32>(0u);
    }
    let first = u32(y) * params.grid_w + u32(max(home.x - 1, 0));
    let last = u32(y) * params.grid_w + u32(min(home.x + 1, i32(params.grid_w) - 1));
    return vec2<u32>(cell_start(first), cell_start(last) + counts[last]);
}
`

// densityWGSL sums the kernels over each particle's neighbors, itself included.
const densityWGSL = `
@group(0) @binding(1) var<storage, read> sorted: array<Particle>;
@group(0) @binding(2) var<storage, read> counts: array<u32>;
@group(0) @binding(3) var<storage, read> starts: array<u32>;
@group(0) @binding(4) var<storage, read> sums: array<u32>;
@group(0) @binding(5) var<storage, read_write> densities: array<vec2<f32>>;

@compute @workgroup_size(256)
fn density(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = id.x;
    if (i >= params.count) {
        return;
    }
    let p = sorted[i].pos;
    let home = cell_of(p);
    var rho = vec2<f32>(0.0);
    for (var dy = -1; dy <= 1; dy++) {
        let span = row(home, dy);
        for (var j = span.x; j < span.y; j++) {
            let r = distance(p, sorted[j].pos);
            if (r < params.h) {
                rho += vec2<f32>(w_density(r), w_near(r));
            }
        }
    }
    densities[i] = rho;
}
`

// forceWGSL accelerates each particle by the pressure and viscosity of its
// neighbors, moves it and keeps it in the box. The particles come out in
// cell order.
const forceWGSL = `
@group(0) @binding(1) var<storage, read> sorted: array<Particle>;
@group(0) @binding(2) var<storage, read> counts: array<u32>;
@group(0) @binding(3) var<storage, read> starts: array<u32>;
@group(0) @binding(4) var<storage, read> sums: array<u32>;
@group(0) @binding(5) var<storage, read> densities: array<vec2<f32>>;
@group(0) @binding(6) var<storage, read> previous: array<vec2<f32>>;
@group(0) @binding(7) var<storage, read_write> particles: array<Particle>;

fn pressure(rho: vec2<f32>) -> vec2<f32> {
    return vec2<f32>(params.stiffness * (rho.x - params.rest), params.near_stiffness * rho.y);
}

@compute @workgroup_size(256)
fn force(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = id.x;
    if (i >= params.count) {
        return;
    }
    let p = sorted[i];
    let rho = densities[i];
    let pr = pressure(rho);
    let home = cell_of(p.pos);
    var a = vec2<f32>(0.0);
    var visc = vec2<f32>(0.0);
    for (var dy = -1; dy <= 1; dy++) {
        let span = row(home, dy);
        for (var j = span.x; j < span.y; j++) {
            if (j == i) {
                continue;
            }
            let q = sorted[j];
            let d = q.pos - p.pos;
            let r = length(d);
            if (r >= params.h) {
                continue;
            }
            var dir = vec2<f32>(0.0, 1.0);
            if (r > 0.0) {
                dir = d / r;
            }
            let rho_j = densities[j];
            let pavg = (pr + pressure(rho_j)) * 0.5;
            a += dir * (dw_density(r) * pavg.x / rho_j.x + dw_near(r) * pavg.y / rho_j.y);
            visc += (q.vel - p.vel) * w_viscosity(r);
        }
    }
    var vel = p.vel + (a / rho.x + visc * params.viscosity) * params.dt;
    var pos = previous[i] + vel * params.dt;
    let lim = params.bounds - params.radius;
    if (abs(pos.x) > lim.x) {
        pos.x = clamp(pos.x, -lim.x, lim.x);
        vel.x *= -0.3;
    }
    if (abs(pos.y) > lim.y) {
        pos.y = clamp(pos.y, -lim.y, lim.y);
        vel.y *= -0.3;
    }
    particles[i] = Particle(pos, vel);
}
`

// renderWGSL draws each particle as a soft disc, bluer the slower it moves.
const renderWGSL = `
struct Vertex {
    @builtin(position) pos: vec4<f32>,
    @location(0) uv: vec2<f32>,
    @location(1) color: vec3<f32>,
}

// Each particle is an instance and the six vertices make its quad, a little
// wider than its radius for the soft edge.
@vertex
fn vs_main(@builtin(vertex_index) vi: u32, @location(0) pos: vec2<f32>, @location(1) vel: vec2<f32>) -> Vertex {
    var corners = array<vec2<f32>, 6>(
        vec2<f32>(-1.0, -1.0), vec2<f32>(1.0, -1.0), vec2<f32>(-1.0, 1.0),
        vec2<f32>(-1.0, 1.0), vec2<f32>(1.0, -1.0), vec2<f32>(1.0, 1.0),
    );
    let corner = corners[vi];
    let heat = clamp(length(vel) / 3.0, 0.0, 1.0);
    var out: Vertex;
    out.pos = vec4<f32>((pos + corner * params.radius * 1.3) / params.bounds, 0.0, 1.0);
    out.uv = corner;
    out.color = mix(vec3<f32>(0.1, 0.4, 0.95), vec3<f32>(0.8, 0.95, 1.0), heat);
    return out;
}

@fragment
fn fs_main(in: Vertex) -> @location(0) vec4<f32> {
    let alpha = 1.0 - smoothstep(0.6, 1.0, length(in.uv));
    return vec4<f32>(in.color * alpha, alpha);
}
`

func main() {
	n := flag.Int("n", 16384, "how many particles")
	backend := flag.String("backend", "", "vulkan, dx12, metal or gl; by default whichever of vulkan, dx12 and metal is there")
	frames := flag.Int("frames", 0, "exit after this many frames, 0 runs until the window is closed")
	flag.Parse()
	if *n <= 0 || *n > maxParticles {
		log.Fatalf("-n must be 1 to %d", maxParticles)
	}
	if err := run(uint32(*n), *frames, *backend); err != nil { //nolint:gosec // range checked above
		log.Fatal(err)
	}
}

func run(n uint32, limit int, backend string) error {
	w, err := window.Open(fmt.Sprintf("gowgpu fluid: %d", n), backend)
	if err != nil {
		return err
	}
	defer w.Close()
	aspect := float32(1.6)
	if width, height := w.Win.GetFramebufferSize(); width > 0 && height > 0 {
		aspect = float32(width) / float32(height)
	}
	s, err := newScene(w.Device, n, aspect, w.Format)
	if err != nil {
		return err
	}
	defer s.release()
	return w.Run(limit, func(view *wgpu.TextureView, width, height int, dt float32) error {
		return s.frame(w, view, width, height, dt)
	})
}

// step is one compute pass of a substep.
type step struct {
	pipeline *wgpu.ComputePipeline
	group    *wgpu.BindGroup
	groups   uint32 // workgroups to dispatch
}

// scene is the GPU side: the buffers, the seven compute steps and the draw.
type scene struct {
	p     params
	dtMax float32 // the longest substep the simulation stays stable at
	buf   []byte

	particles, predicted, sorted, previous, densities *wgpu.Buffer
	counts, ranks, starts, sums, uniforms             *wgpu.Buffer
	steps                                             [7]step
	render                                            *wgpu.RenderPipeline
	renderGroup                                       *wgpu.BindGroup

	resources []interface{ Release() } // everything made, in order
	err       error                    // the first failure while making the scene
}

func newScene(device *wgpu.Device, n uint32, aspect float32, format gputypes.TextureFormat) (*scene, error) {
	// The water fills the bottom half of the box, 2*aspect by 1, at a
	// spacing that puts about 20 neighbors within h of each particle.
	area := 2 * aspect
	spacing := float32(math.Sqrt(float64(area / float32(n))))
	h := 2.5 * spacing
	// The speed of sound is the square root of the stiffness, and the
	// deepest water, 1 unit down, compresses by gravity/stiffness.
	const gravity, compression = 4, 0.1
	stiffness := float32(gravity / compression)
	s := &scene{p: params{
		H:             h,
		Rest:          float32(n) / area,
		Stiffness:     stiffness,
		NearStiffness: 0.5 * compression * stiffness,
		Viscosity:     3 * area / float32(n), // velocities relax to the neighbors' in about a third of a second
		Gravity:       gravity,
		Radius:        0.75 * spacing,
		Count:         n,
	}}
	// A particle moves under half a cell a substep at the speed of sound
	// plus the speed of a fall through the box.
	s.dtMax = 0.4 * h / (float32(math.Sqrt(float64(stiffness))) + 3)

	s.particles = s.buffer(device, "particles", uint64(n)*particleBytes, wgpu.BufferUsageStorage|wgpu.BufferUsageVertex|wgpu.BufferUsageCopyDst)
	s.predicted = s.buffer(device, "predicted", uint64(n)*particleBytes, wgpu.BufferUsageStorage)
	s.sorted = s.buffer(device, "sorted", uint64(n)*particleBytes, wgpu.BufferUsageStorage)
	s.previous = s.buffer(device, "previous", uint64(n)*8, wgpu.BufferUsageStorage)
	s.densities = s.buffer(device, "densities", uint64(n)*8, wgpu.BufferUsageStorage)
	s.counts = s.buffer(device, "counts", maxCells*4, wgpu.BufferUsageStorage)
	s.ranks = s.buffer(device, "ranks", uint64(n)*4, wgpu.BufferUsageStorage)
	s.starts = s.buffer(device, "starts", maxCells*4, wgpu.BufferUsageStorage)
	s.sums = s.buffer(device, "sums", workgroup*4, wgpu.BufferUsageStorage)
	s.uniforms = s.buffer(device, "params", uint64(binary.Size(params{})), wgpu.BufferUsageUniform|wgpu.BufferUsageCopyDst) //nolint:gosec // a struct is not negative
	if s.err == nil {
		s.err = device.Queue().WriteBuffer(s.particles, 0, seed(n, aspect, spacing))
	}

	// No pipeline gets a layout: gowgpu derives one from its shader, and
	// GetBindGroupLayout hands it out to make the bind group with. Params is
	// binding 0 everywhere, then each step's buffers in order.
	prelude := typesWGSL + funcsWGSL
	perParticle := (n + workgroup - 1) / workgroup
	for i, c := range []struct {
		entry, wgsl string
		buffers     []*wgpu.Buffer
		groups      uint32
	}{
		{"predict", prelude + predictWGSL, []*wgpu.Buffer{s.particles, s.predicted, s.counts}, (max(n, maxCells) + workgroup - 1) / workgroup}, // it clears the grid too
		{"bin", prelude + binWGSL, []*wgpu.Buffer{s.predicted, s.counts, s.ranks}, perParticle},
		{"scan_cells", prelude + scanWGSL + scanCellsWGSL, []*wgpu.Buffer{s.counts, s.starts, s.sums}, maxCells / workgroup},
		{"scan_blocks", prelude + scanWGSL + scanBlocksWGSL, []*wgpu.Buffer{s.sums}, 1},
		{"scatter", prelude + scatterWGSL, []*wgpu.Buffer{s.particles, s.predicted, s.ranks, s.starts, s.sums, s.sorted, s.previous}, perParticle},
		{"density", prelude + gridWGSL + densityWGSL, []*wgpu.Buffer{s.sorted, s.counts, s.starts, s.sums, s.densities}, perParticle},
		{"force", prelude + gridWGSL + forceWGSL, []*wgpu.Buffer{s.sorted, s.counts, s.starts, s.sums, s.densities, s.previous, s.particles}, perParticle},
	} {
		s.steps[i] = s.step(device, c.entry, c.wgsl, c.buffers)
		s.steps[i].groups = c.groups
	}
	s.draw(device, format)
	if s.err != nil {
		s.release()
		return nil, s.err
	}
	return s, nil
}

func (s *scene) buffer(device *wgpu.Device, label string, size uint64, usage gputypes.BufferUsage) *wgpu.Buffer {
	if s.err != nil {
		return nil
	}
	b, err := device.CreateBuffer(&wgpu.BufferDescriptor{Label: label, Size: size, Usage: usage})
	if err != nil {
		s.err = fmt.Errorf("%s: %w", label, err)
		return nil
	}
	s.resources = append(s.resources, b)
	return b
}

func (s *scene) step(device *wgpu.Device, entry, wgsl string, buffers []*wgpu.Buffer) step {
	if s.err != nil {
		return step{}
	}
	module, err := device.CreateShaderModule(&wgpu.ShaderModuleDescriptor{WGSL: wgsl})
	if err != nil {
		s.err = fmt.Errorf("%s: %w", entry, err)
		return step{}
	}
	defer module.Release()
	pipeline, err := device.CreateComputePipeline(&wgpu.ComputePipelineDescriptor{Label: entry, Module: module, EntryPoint: entry})
	if err != nil {
		s.err = fmt.Errorf("%s: %w", entry, err)
		return step{}
	}
	s.resources = append(s.resources, pipeline)
	entries := []wgpu.BindGroupEntry{{Binding: 0, Buffer: s.uniforms}}
	for i, b := range buffers {
		entries = append(entries, wgpu.BindGroupEntry{Binding: uint32(i + 1), Buffer: b})
	}
	group, err := device.CreateBindGroup(&wgpu.BindGroupDescriptor{Layout: pipeline.GetBindGroupLayout(0), Entries: entries})
	if err != nil {
		s.err = fmt.Errorf("%s: %w", entry, err)
		return step{}
	}
	s.resources = append(s.resources, group)
	return step{pipeline: pipeline, group: group}
}

// draw makes the render pipeline, drawing the particles buffer as instances
// with premultiplied alpha.
func (s *scene) draw(device *wgpu.Device, format gputypes.TextureFormat) {
	if s.err != nil {
		return
	}
	module, err := device.CreateShaderModule(&wgpu.ShaderModuleDescriptor{WGSL: typesWGSL + renderWGSL})
	if err != nil {
		s.err = fmt.Errorf("draw: %w", err)
		return
	}
	defer module.Release()
	over := gputypes.BlendComponent{SrcFactor: gputypes.BlendFactorOne, DstFactor: gputypes.BlendFactorOneMinusSrcAlpha, Operation: gputypes.BlendOperationAdd}
	s.render, err = device.CreateRenderPipeline(&wgpu.RenderPipelineDescriptor{
		Label: "draw",
		Vertex: wgpu.VertexState{
			Module:     module,
			EntryPoint: "vs_main",
			Buffers: []gputypes.VertexBufferLayout{{
				ArrayStride: particleBytes,
				StepMode:    gputypes.VertexStepModeInstance,
				Attributes: []gputypes.VertexAttribute{
					{Format: gputypes.VertexFormatFloat32x2, Offset: 0, ShaderLocation: 0},
					{Format: gputypes.VertexFormatFloat32x2, Offset: 8, ShaderLocation: 1},
				},
			}},
		},
		Fragment: &wgpu.FragmentState{
			Module:     module,
			EntryPoint: "fs_main",
			Targets: []gputypes.ColorTargetState{{
				Format:    format,
				Blend:     &gputypes.BlendState{Color: over, Alpha: over},
				WriteMask: gputypes.ColorWriteMaskAll,
			}},
		},
	})
	if err != nil {
		s.err = fmt.Errorf("draw: %w", err)
		return
	}
	s.resources = append(s.resources, s.render)
	s.renderGroup, err = device.CreateBindGroup(&wgpu.BindGroupDescriptor{
		Layout:  s.render.GetBindGroupLayout(0),
		Entries: []wgpu.BindGroupEntry{{Binding: 0, Buffer: s.uniforms}},
	})
	if err != nil {
		s.err = fmt.Errorf("draw: %w", err)
		return
	}
	s.resources = append(s.resources, s.renderGroup)
}

// seed fills the bottom half of the box with particles a spacing apart,
// nudged a little so no column falls as one.
func seed(n uint32, aspect, spacing float32) []byte {
	buf := make([]byte, 0, int(n)*particleBytes)
	cols := max(1, int(2*aspect/spacing))
	for i := range int(n) {
		x := -aspect + spacing*(0.5+float32(i%cols)+0.1*(rand.Float32()-0.5)) //nolint:gosec // not a secret
		y := -1 + spacing*(0.5+float32(i/cols))
		for _, v := range [4]float32{x, y, 0, 0} {
			buf = binary.LittleEndian.AppendUint32(buf, math.Float32bits(v))
		}
	}
	return buf
}

// frame reads the pointer, runs as many substeps as dt needs and draws the
// result, all in one submit.
func (s *scene) frame(w *window.Window, view *wgpu.TextureView, width, height int, dt float32) error {
	aspect := float32(width) / float32(height)
	s.p.Bounds = [2]float32{aspect, 1}
	s.p.Frame = [2]float32{float32(width), float32(height)}
	s.p.GridW = uint32(math.Ceil(float64(2 * aspect / s.p.H)))
	s.p.GridH = uint32(math.Ceil(float64(2 / s.p.H)))
	if s.p.GridW*s.p.GridH > maxCells { // ponytail: a very wide window overfills the edge cells rather than growing the grid
		s.p.GridW = maxCells / s.p.GridH
	}
	x, y := w.Win.GetCursorPos()
	ww, wh := w.Win.GetSize()
	s.p.Mouse = [2]float32{float32(2*x/float64(ww)-1) * aspect, float32(1 - 2*y/float64(wh))}
	s.p.Force = 0
	if w.Win.GetMouseButton(glfw.MouseButtonLeft) == glfw.Press {
		s.p.Force = 40
	}
	if w.Win.GetMouseButton(glfw.MouseButtonRight) == glfw.Press {
		s.p.Force = -80
	}
	substeps := max(1, int(math.Ceil(float64(dt/s.dtMax))))
	s.p.Dt = dt / float32(substeps)
	s.buf, _ = binary.Append(s.buf[:0], binary.LittleEndian, &s.p)
	if err := w.Device.Queue().WriteBuffer(s.uniforms, 0, s.buf); err != nil {
		return err
	}

	enc, err := w.Device.CreateCommandEncoder(nil)
	if err != nil {
		return err
	}
	cp, err := enc.BeginComputePass(nil)
	if err != nil {
		return err
	}
	for range substeps {
		for _, st := range s.steps {
			cp.SetPipeline(st.pipeline)
			cp.SetBindGroup(0, st.group, nil)
			cp.Dispatch(st.groups, 1, 1)
		}
	}
	if err := cp.End(); err != nil {
		return err
	}

	rp, err := enc.BeginRenderPass(&wgpu.RenderPassDescriptor{
		ColorAttachments: []wgpu.RenderPassColorAttachment{{
			View:       view,
			LoadOp:     gputypes.LoadOpClear,
			StoreOp:    gputypes.StoreOpStore,
			ClearValue: gputypes.Color{R: 0.02, G: 0.02, B: 0.04, A: 1},
		}},
	})
	if err != nil {
		return err
	}
	rp.SetPipeline(s.render)
	rp.SetBindGroup(0, s.renderGroup, nil)
	rp.SetVertexBuffer(0, s.particles, 0)
	rp.Draw(gputypes.DrawArgs{VertexCount: 6, InstanceCount: s.p.Count})
	if err := rp.End(); err != nil {
		return err
	}

	cmd, err := enc.Finish()
	if err != nil {
		return err
	}
	_, err = w.Device.Queue().Submit(cmd)
	return err
}

func (s *scene) release() {
	for i := len(s.resources) - 1; i >= 0; i-- {
		s.resources[i].Release()
	}
}
