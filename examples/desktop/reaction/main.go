// Command reaction runs Gray-Scott reaction-diffusion in a window: two
// chemicals spread and react on a grid, and depending on how fast the first is
// fed and the second removed they grow into coral, worms, spots that split, or
// a maze. A compute shader steps the grid between two storage textures, back
// and forth, many times a frame, and a render pass shades the result.
//
// Paint with the left mouse button, erase with the right. Keys 1 to 5 pick
// the pattern, R starts over, Space pauses.
//
//	cd examples/desktop
//	go run ./reaction [-steps 16] [-scale 2] [-backend vulkan|dx12|metal|gl] [-frames 300]
//
// -frames exits after that many frames, a smoke test that walks the whole
// setup and teardown. -backend gl needs a context with compute shaders,
// desktop GL 4.3 or GLES 3.1.
package main

import (
	"encoding/binary"
	"flag"
	"log"

	"github.com/go-gl/glfw/v3.4/glfw"
	wgpu "github.com/roffe/gowgpu"
	"github.com/roffe/gowgpu/examples/desktop/window"
	"github.com/roffe/gowgpu/gputypes"
)

const workgroup = 8 // by 8 cells

// params mirrors Params in the shaders.
type params struct {
	Size  [2]uint32 // of the grid
	Reset uint32    // 1 seeds the grid afresh
	Paint float32   // 1 adds the second chemical under the brush, -1 clears it
	Mouse [2]float32
	Brush float32 // radius, in cells
	Feed  float32
	Kill  float32
	_     float32
	View  [2]float32 // the window in pixels
}

// patterns are feed and kill rates that grow something to look at.
var patterns = [][2]float32{
	{0.0545, 0.062},  // coral
	{0.0367, 0.0649}, // mitosis, spots that split
	{0.078, 0.061},   // worms
	{0.029, 0.057},   // maze
	{0.039, 0.058},   // holes
}

const paramsWGSL = `
struct Params {
    size: vec2<u32>,
    reset: u32,
    paint: f32,
    mouse: vec2<f32>,
    brush: f32,
    feed: f32,
    kill: f32,
    view: vec2<f32>,
}

@group(0) @binding(0) var<uniform> params: Params;
`

const stepWGSL = paramsWGSL + `
@group(0) @binding(1) var src: texture_2d<f32>;
@group(0) @binding(2) var dst: texture_storage_2d<rgba32float, write>;

fn hash(p: vec2<u32>) -> f32 {
    var h = p.x * 1597334677u ^ p.y * 3812015801u;
    h = (h ^ (h >> 16u)) * 2246822519u;
    h = (h ^ (h >> 13u)) * 3266489917u;
    return f32((h ^ (h >> 16u)) >> 8u) / 16777216.0;
}

// at is the chemicals at p, the grid wrapping around at its edges.
fn at(p: vec2<i32>) -> vec2<f32> {
    let s = vec2<i32>(params.size);
    return textureLoad(src, (p + s) % s, 0).xy;
}

@compute @workgroup_size(8, 8)
fn step(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x >= params.size.x || id.y >= params.size.y) {
        return;
    }
    let p = vec2<i32>(id.xy);
    var v: vec2<f32>;
    if (params.reset != 0u) {
        // all of the first, and blobs of the second in some cells of a grid
        v = vec2<f32>(1.0, 0.0);
        let cell = id.xy / 48u;
        if (hash(cell) < 0.35 && distance(vec2<f32>(id.xy), vec2<f32>(cell * 48u + 24u)) < 8.0) {
            v.y = 1.0;
        }
    } else {
        let c = at(p);
        let lap = 0.2 * (at(p + vec2<i32>(1, 0)) + at(p - vec2<i32>(1, 0)) + at(p + vec2<i32>(0, 1)) + at(p - vec2<i32>(0, 1)))
            + 0.05 * (at(p + vec2<i32>(1, 1)) + at(p - vec2<i32>(1, 1)) + at(p + vec2<i32>(1, -1)) + at(p - vec2<i32>(1, -1)))
            - c;
        let abb = c.x * c.y * c.y;
        v = clamp(c + vec2<f32>(lap.x - abb + params.feed * (1.0 - c.x), 0.5 * lap.y + abb - (params.kill + params.feed) * c.y),
            vec2<f32>(0.0), vec2<f32>(1.0));
    }
    if (params.paint != 0.0 && distance(vec2<f32>(id.xy) + 0.5, params.mouse) < params.brush) {
        v.y = select(0.0, 0.9, params.paint > 0.0);
    }
    textureStore(dst, p, vec4<f32>(v, 0.0, 1.0));
}
`

const showWGSL = paramsWGSL + `
@group(0) @binding(1) var grid: texture_2d<f32>;

@vertex
fn vs_main(@builtin(vertex_index) i: u32) -> @builtin(position) vec4<f32> {
    let xy = vec2<f32>(f32((i << 1u) & 2u), f32(i & 2u));
    return vec4<f32>(xy * 2.0 - 1.0, 0.0, 1.0);
}

// b is the second chemical at q, in cells, filtered by hand: rgba32float is
// not filterable without a feature.
fn b(q: vec2<f32>) -> f32 {
    let s = vec2<i32>(params.size);
    let f = q - 0.5;
    let i = vec2<i32>(floor(f));
    let t = fract(f);
    let v00 = textureLoad(grid, (i + s) % s, 0).y;
    let v10 = textureLoad(grid, (i + vec2<i32>(1, 0) + s) % s, 0).y;
    let v01 = textureLoad(grid, (i + vec2<i32>(0, 1) + s) % s, 0).y;
    let v11 = textureLoad(grid, (i + vec2<i32>(1, 1) + s) % s, 0).y;
    return mix(mix(v00, v10, t.x), mix(v01, v11, t.x), t.y);
}

// The second chemical is a height field, lit from the top left.
@fragment
fn fs_main(@builtin(position) pos: vec4<f32>) -> @location(0) vec4<f32> {
    let q = pos.xy * vec2<f32>(params.size) / params.view;
    let h = b(q);
    let n = normalize(vec3<f32>(b(q - vec2<f32>(1.0, 0.0)) - b(q + vec2<f32>(1.0, 0.0)),
        b(q - vec2<f32>(0.0, 1.0)) - b(q + vec2<f32>(0.0, 1.0)), 0.08));
    let light = max(dot(n, normalize(vec3<f32>(-0.5, -0.6, 0.6))), 0.0);
    let t = clamp(h * 3.0, 0.0, 1.0);
    let base = mix(vec3<f32>(0.02, 0.03, 0.08), mix(vec3<f32>(0.05, 0.45, 0.6), vec3<f32>(1.0, 0.85, 0.5), t * t), t);
    let spec = pow(max(dot(reflect(vec3<f32>(0.5, 0.6, -0.6), n), vec3<f32>(0.0, 0.0, 1.0)), 0.0), 24.0) * t;
    return vec4<f32>(base * (0.35 + 0.8 * light) + spec * 0.5, 1.0);
}
`

func main() {
	steps := flag.Int("steps", 16, "simulation steps every 60th of a second, an even number")
	scale := flag.Int("scale", 2, "window pixels a grid cell covers")
	backend := flag.String("backend", "", "vulkan, dx12, metal or gl; by default whichever of vulkan, dx12 and metal is there")
	frames := flag.Int("frames", 0, "exit after this many frames, 0 runs until the window is closed")
	flag.Parse()
	if *steps < 2 || *steps%2 != 0 || *scale < 1 {
		log.Fatal("-steps must be even and at least 2, -scale at least 1")
	}
	if err := run(*steps, *scale, *frames, *backend); err != nil {
		log.Fatal(err)
	}
}

func run(steps, scale, limit int, backend string) error {
	w, err := window.Open("gowgpu reaction-diffusion", backend)
	if err != nil {
		return err
	}
	defer w.Close()
	s, err := newSim(w.Device, w.Format)
	if err != nil {
		return err
	}
	defer s.release()

	var (
		p      = params{Brush: 10}
		paused bool
		buf    []byte
		due    float32 // steps owed, so the speed does not follow the frame rate
	)
	pick := func(i int) { p.Feed, p.Kill = patterns[i][0], patterns[i][1] }
	pick(0)
	w.Win.SetKeyCallback(func(_ *glfw.Window, key glfw.Key, _ int, action glfw.Action, _ glfw.ModifierKey) {
		if action != glfw.Press {
			return
		}
		switch {
		case key >= glfw.Key1 && key < glfw.Key1+glfw.Key(len(patterns)):
			pick(int(key - glfw.Key1))
		case key == glfw.KeyR:
			p.Reset = 1
		case key == glfw.KeySpace:
			paused = !paused
		}
	})
	return w.Run(limit, func(view *wgpu.TextureView, width, height int, dt float32) error {
		gw, gh := max(width/scale, 1), max(height/scale, 1)
		if uint32(gw) != p.Size[0] || uint32(gh) != p.Size[1] { //nolint:gosec // positive
			if err := s.resize(w.Device, gw, gh); err != nil {
				return err
			}
			p.Size = [2]uint32{uint32(gw), uint32(gh)} //nolint:gosec // positive
			p.Reset = 1
		}
		p.View = [2]float32{float32(width), float32(height)}

		x, y := w.Win.GetCursorPos()
		ww, wh := w.Win.GetSize()
		p.Mouse = [2]float32{float32(x / float64(ww) * float64(gw)), float32(y / float64(wh) * float64(gh))}
		switch {
		case w.Win.GetMouseButton(glfw.MouseButtonLeft) == glfw.Press:
			p.Paint = 1
		case w.Win.GetMouseButton(glfw.MouseButtonRight) == glfw.Press:
			p.Paint = -1
		default:
			p.Paint = 0
		}

		buf, _ = binary.Append(buf[:0], binary.LittleEndian, &p)
		if err := w.Device.Queue().WriteBuffer(s.uniforms, 0, buf); err != nil {
			return err
		}
		due += dt * 60 * float32(steps)
		n := int(due) &^ 1 // even, so the steps end on grid 0
		due -= float32(n)
		if paused {
			n = 0
		}
		if p.Reset != 0 || paused && p.Paint != 0 {
			n = max(n, 2) // one step writes grid 1, the next grid 0, the one shown
		}
		p.Reset = 0
		return s.frame(w.Device, view, n)
	})
}

// sim is the GPU side: two grids the steps go back and forth between, the
// parameters, and the pipelines that step and show the grid.
type sim struct {
	uniforms   *wgpu.Buffer
	grids      [2]*wgpu.Texture
	views      [2]*wgpu.TextureView
	step       *wgpu.ComputePipeline
	show       *wgpu.RenderPipeline
	stepGroups [2]*wgpu.BindGroup // [i] reads grid i and writes the other
	showGroup  *wgpu.BindGroup    // reads grid 0, where an even number of steps ends
	size       [2]uint32
}

func newSim(device *wgpu.Device, format gputypes.TextureFormat) (*sim, error) {
	s := &sim{}
	var err error
	if s.uniforms, err = device.CreateBuffer(&wgpu.BufferDescriptor{
		Label: "params",
		Size:  uint64(binary.Size(params{})), //nolint:gosec // a struct is not negative
		Usage: wgpu.BufferUsageUniform | wgpu.BufferUsageCopyDst,
	}); err != nil {
		return nil, err
	}
	cs, err := device.CreateShaderModule(&wgpu.ShaderModuleDescriptor{WGSL: stepWGSL})
	if err != nil {
		s.release()
		return nil, err
	}
	defer cs.Release()
	if s.step, err = device.CreateComputePipeline(&wgpu.ComputePipelineDescriptor{
		Label: "step", Module: cs, EntryPoint: "step",
	}); err != nil {
		s.release()
		return nil, err
	}
	rs, err := device.CreateShaderModule(&wgpu.ShaderModuleDescriptor{WGSL: showWGSL})
	if err != nil {
		s.release()
		return nil, err
	}
	defer rs.Release()
	if s.show, err = device.CreateRenderPipeline(&wgpu.RenderPipelineDescriptor{
		Label:  "show",
		Vertex: wgpu.VertexState{Module: rs, EntryPoint: "vs_main"},
		Fragment: &wgpu.FragmentState{
			Module:     rs,
			EntryPoint: "fs_main",
			Targets:    []gputypes.ColorTargetState{{Format: format, WriteMask: gputypes.ColorWriteMaskAll}},
		},
	}); err != nil {
		s.release()
		return nil, err
	}
	return s, nil
}

// resize makes the grids width by height cells and the bind groups on them.
func (s *sim) resize(device *wgpu.Device, width, height int) error {
	s.releaseSized()
	var err error
	for i := range s.grids {
		if s.grids[i], err = device.CreateTexture(&wgpu.TextureDescriptor{
			Label:         "grid",
			Size:          wgpu.Extent3D{Width: uint32(width), Height: uint32(height), DepthOrArrayLayers: 1}, //nolint:gosec // positive
			MipLevelCount: 1,
			SampleCount:   1,
			Dimension:     gputypes.TextureDimension2D,
			Format:        gputypes.TextureFormatRGBA32Float,
			Usage:         wgpu.TextureUsageStorageBinding | wgpu.TextureUsageTextureBinding,
		}); err != nil {
			return err
		}
		if s.views[i], err = device.CreateTextureView(s.grids[i], nil); err != nil {
			return err
		}
	}
	for i := range s.stepGroups {
		if s.stepGroups[i], err = device.CreateBindGroup(&wgpu.BindGroupDescriptor{
			Layout: s.step.GetBindGroupLayout(0),
			Entries: []wgpu.BindGroupEntry{
				{Binding: 0, Buffer: s.uniforms},
				{Binding: 1, TextureView: s.views[i]},
				{Binding: 2, TextureView: s.views[1-i]},
			},
		}); err != nil {
			return err
		}
	}
	s.showGroup, err = device.CreateBindGroup(&wgpu.BindGroupDescriptor{
		Layout: s.show.GetBindGroupLayout(0),
		Entries: []wgpu.BindGroupEntry{
			{Binding: 0, Buffer: s.uniforms},
			{Binding: 1, TextureView: s.views[0]},
		},
	})
	s.size = [2]uint32{uint32(width), uint32(height)} //nolint:gosec // positive
	return err
}

// frame runs steps steps, each a pass of its own so the grid it wrote is
// what the next reads, and draws the grid into view, in one submit.
func (s *sim) frame(device *wgpu.Device, view *wgpu.TextureView, steps int) error {
	enc, err := device.CreateCommandEncoder(nil)
	if err != nil {
		return err
	}
	for i := range steps {
		cp, err := enc.BeginComputePass(nil)
		if err != nil {
			return err
		}
		cp.SetPipeline(s.step)
		cp.SetBindGroup(0, s.stepGroups[i%2], nil)
		cp.Dispatch((s.size[0]+workgroup-1)/workgroup, (s.size[1]+workgroup-1)/workgroup, 1)
		if err := cp.End(); err != nil {
			return err
		}
	}

	rp, err := enc.BeginRenderPass(&wgpu.RenderPassDescriptor{
		ColorAttachments: []wgpu.RenderPassColorAttachment{{
			View:    view,
			LoadOp:  gputypes.LoadOpClear,
			StoreOp: gputypes.StoreOpStore,
		}},
	})
	if err != nil {
		return err
	}
	rp.SetPipeline(s.show)
	rp.SetBindGroup(0, s.showGroup, nil)
	rp.Draw(gputypes.DrawArgs{VertexCount: 3, InstanceCount: 1})
	if err := rp.End(); err != nil {
		return err
	}

	cmd, err := enc.Finish()
	if err != nil {
		return err
	}
	_, err = device.Queue().Submit(cmd)
	return err
}

func (s *sim) releaseSized() {
	if s.showGroup != nil {
		s.showGroup.Release()
	}
	for i := range 2 {
		if s.stepGroups[i] != nil {
			s.stepGroups[i].Release()
		}
		if s.views[i] != nil {
			s.views[i].Release()
		}
		if s.grids[i] != nil {
			s.grids[i].Release()
		}
	}
	s.showGroup, s.stepGroups, s.views, s.grids = nil, [2]*wgpu.BindGroup{}, [2]*wgpu.TextureView{}, [2]*wgpu.Texture{}
}

func (s *sim) release() {
	s.releaseSized()
	if s.show != nil {
		s.show.Release()
	}
	if s.step != nil {
		s.step.Release()
	}
	if s.uniforms != nil {
		s.uniforms.Release()
	}
}
