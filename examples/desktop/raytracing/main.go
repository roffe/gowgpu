// Command raytracing path traces the scene of Ray Tracing in One Weekend in a
// window: a few hundred diffuse, metal and glass spheres on a checkered floor
// under a sky, with depth of field. A compute shader adds samples to every
// pixel each frame and a render pass shows their average, so the picture
// clears up while the camera stands still.
//
// Drag with the left mouse button to orbit, scroll to move closer or away.
// Moving the camera or resizing the window starts the samples over, and
// while the camera moves each frame traces -move-spp samples rather than
// -spp, so the picture is less grainy than a single sample.
//
//	cd examples/desktop
//	go run ./raytracing [-spp 1] [-move-spp 4] [-bounces 8] [-aperture 0.1] [-backend vulkan|dx12|metal|gl] [-frames 300]
//
// -frames exits after that many frames, a smoke test that walks the whole
// setup and teardown. -backend gl needs a context with compute shaders,
// desktop GL 4.3 or GLES 3.1.
package main

import (
	"encoding/binary"
	"flag"
	"log"
	"math"
	"math/rand/v2"

	"github.com/go-gl/glfw/v3.4/glfw"
	wgpu "github.com/roffe/gowgpu"
	"github.com/roffe/gowgpu/examples/desktop/window"
	"github.com/roffe/gowgpu/gputypes"
)

const (
	workgroup  = 8 // by 8 pixels
	pixelBytes = 16
	maxFrames  = 1 << 14 // f32 sums stop gaining from more samples about here
)

// params mirrors Params in the shaders.
type params struct {
	Origin  [4]float32 // the lens centre, w its radius
	Corner  [4]float32 // the bottom left of the focus plane
	Horiz   [4]float32
	Vert    [4]float32
	U, V    [4]float32 // the lens' axes
	Size    [2]uint32
	Frame   uint32 // samples so far, 0 starts over
	Count   uint32 // spheres
	SPP     uint32
	Bounces uint32
	_       [2]uint32
}

// sphere mirrors Sphere in the shaders.
type sphere struct {
	Pos    [4]float32 // centre and radius
	Albedo [4]float32 // colour and kind: 0 diffuse, 1 metal, 2 glass
	Mat    [4]float32 // fuzz of metal, index of refraction of glass
}

const (
	diffuse = iota
	metal
	glass
)

const paramsWGSL = `
struct Params {
    origin: vec4<f32>,
    corner: vec4<f32>,
    horiz: vec4<f32>,
    vert: vec4<f32>,
    u: vec4<f32>,
    v: vec4<f32>,
    size: vec2<u32>,
    frame: u32,
    count: u32,
    spp: u32,
    bounces: u32,
}
`

const traceWGSL = paramsWGSL + `
struct Sphere {
    pos: vec4<f32>,
    albedo: vec4<f32>,
    mat: vec4<f32>,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> spheres: array<Sphere>;
@group(0) @binding(2) var<storage, read_write> accum: array<vec4<f32>>;

const NONE = 0xffffffffu;
const FLOOR = 0xfffffffeu;
const TAU = 6.2831853;

var<private> seed: u32;

// rand is PCG, a number in [0, 1).
fn rand() -> f32 {
    seed = seed * 747796405u + 2891336453u;
    var w = ((seed >> ((seed >> 28u) + 4u)) ^ seed) * 277803737u;
    w = (w >> 22u) ^ w;
    return f32(w >> 8u) / 16777216.0;
}

fn on_sphere() -> vec3<f32> {
    let z = rand() * 2.0 - 1.0;
    let a = rand() * TAU;
    let r = sqrt(max(1.0 - z * z, 0.0));
    return vec3<f32>(r * cos(a), r * sin(a), z);
}

fn in_disk() -> vec2<f32> {
    let r = sqrt(rand());
    let a = rand() * TAU;
    return r * vec2<f32>(cos(a), sin(a));
}

struct Hit {
    t: f32,
    n: vec3<f32>, // facing the ray
    i: u32,       // the sphere, FLOOR or NONE
    front: bool,  // the ray comes from outside
}

// hit finds the nearest of the spheres and the floor, the plane y = 0, which
// is a plane rather than RTIOW's huge sphere: f32 is too coarse for that one.
fn hit(ro: vec3<f32>, rd: vec3<f32>) -> Hit {
    var h: Hit;
    h.t = 1e30;
    h.i = NONE;
    if (rd.y < 0.0) {
        let t = -ro.y / rd.y;
        if (t > 0.001) {
            h.t = t;
            h.i = FLOOR;
        }
    }
    for (var i = 0u; i < params.count; i++) {
        let s = spheres[i].pos;
        let oc = ro - s.xyz;
        let b = dot(oc, rd);
        let c = dot(oc, oc) - s.w * s.w;
        let disc = b * b - c;
        if (disc < 0.0) {
            continue;
        }
        let sq = sqrt(disc);
        var t = -b - sq;
        if (t < 0.001) {
            t = -b + sq;
        }
        if (t < 0.001 || t >= h.t) {
            continue;
        }
        h.t = t;
        h.i = i;
    }
    if (h.i == FLOOR) {
        h.n = vec3<f32>(0.0, 1.0, 0.0);
        h.front = true;
    } else if (h.i != NONE) {
        let s = spheres[h.i].pos;
        let out = (ro + rd * h.t - s.xyz) / s.w;
        h.front = dot(rd, out) < 0.0;
        h.n = select(-out, out, h.front);
    }
    return h;
}

fn sky(rd: vec3<f32>) -> vec3<f32> {
    return mix(vec3<f32>(1.0), vec3<f32>(0.5, 0.7, 1.0), 0.5 * (rd.y + 1.0));
}

// schlick is how much glass reflects at an angle.
fn schlick(cosine: f32, ratio: f32) -> f32 {
    var r0 = (1.0 - ratio) / (1.0 + ratio);
    r0 = r0 * r0;
    return r0 + (1.0 - r0) * pow(1.0 - cosine, 5.0);
}

fn trace(origin: vec3<f32>, dir: vec3<f32>) -> vec3<f32> {
    var ro = origin;
    var rd = dir;
    var light = vec3<f32>(1.0);
    for (var bounce = 0u; bounce < params.bounces; bounce++) {
        let h = hit(ro, rd);
        if (h.i == NONE) {
            return light * sky(rd);
        }
        ro = ro + rd * h.t;
        if (h.i == FLOOR) {
            let checker = (i32(floor(ro.x)) + i32(floor(ro.z))) & 1;
            light *= select(vec3<f32>(0.9), vec3<f32>(0.2, 0.3, 0.1), checker == 1);
            rd = h.n + on_sphere();
        } else {
            let s = spheres[h.i];
            let kind = u32(s.albedo.w);
            if (kind == 0u) {
                light *= s.albedo.rgb;
                rd = h.n + on_sphere();
            } else if (kind == 1u) {
                light *= s.albedo.rgb;
                rd = reflect(rd, h.n) + s.mat.x * on_sphere();
                if (dot(rd, h.n) <= 0.0) {
                    return vec3<f32>(0.0);
                }
            } else {
                let ratio = select(s.mat.y, 1.0 / s.mat.y, h.front);
                let cosine = min(dot(-rd, h.n), 1.0);
                let sine = sqrt(1.0 - cosine * cosine);
                if (ratio * sine > 1.0 || schlick(cosine, ratio) > rand()) {
                    rd = reflect(rd, h.n);
                } else {
                    rd = refract(rd, h.n, ratio);
                }
            }
        }
        let len = length(rd);
        rd = select(h.n, rd / len, len > 1e-6); // a diffuse bounce can cancel out
    }
    return vec3<f32>(0.0);
}

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x >= params.size.x || id.y >= params.size.y) {
        return;
    }
    let idx = id.y * params.size.x + id.x;
    seed = idx;
    rand();
    seed ^= params.frame * 2654435769u;
    rand();

    var c = vec3<f32>(0.0);
    for (var s = 0u; s < params.spp; s++) {
        let uv = (vec2<f32>(id.xy) + vec2<f32>(rand(), rand())) / vec2<f32>(params.size);
        let lens = params.origin.w * in_disk();
        let ro = params.origin.xyz + params.u.xyz * lens.x + params.v.xyz * lens.y;
        let at = params.corner.xyz + uv.x * params.horiz.xyz + (1.0 - uv.y) * params.vert.xyz;
        c += trace(ro, normalize(at - ro));
    }
    if (any(c != c)) {
        return; // a NaN would spoil the pixel for good
    }
    let sample = vec4<f32>(c, f32(params.spp));
    if (params.frame == 0u) {
        accum[idx] = sample;
    } else {
        accum[idx] += sample;
    }
}
`

const showWGSL = paramsWGSL + `
@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> accum: array<vec4<f32>>;

@vertex
fn vs_main(@builtin(vertex_index) i: u32) -> @builtin(position) vec4<f32> {
    let xy = vec2<f32>(f32((i << 1u) & 2u), f32(i & 2u));
    return vec4<f32>(xy * 2.0 - 1.0, 0.0, 1.0);
}

@fragment
fn fs_main(@builtin(position) pos: vec4<f32>) -> @location(0) vec4<f32> {
    let p = vec2<u32>(pos.xy);
    let a = accum[p.y * params.size.x + p.x];
    let c = a.rgb / max(a.w, 1.0);
    return vec4<f32>(sqrt(clamp(c, vec3<f32>(0.0), vec3<f32>(1.0))), 1.0); // gamma 2
}
`

func main() {
	spp := flag.Uint("spp", 1, "samples per pixel each frame")
	moveSPP := flag.Uint("move-spp", 4, "samples per pixel each frame while the camera moves")
	bounces := flag.Uint("bounces", 8, "how many times a ray bounces at most")
	aperture := flag.Float64("aperture", 0.1, "lens diameter, 0 keeps everything sharp")
	backend := flag.String("backend", "", "vulkan, dx12, metal or gl; by default whichever of vulkan, dx12 and metal is there")
	frames := flag.Int("frames", 0, "exit after this many frames, 0 runs until the window is closed")
	flag.Parse()
	if *spp == 0 || *spp > 64 || *moveSPP == 0 || *moveSPP > 64 || *bounces == 0 || *bounces > 64 {
		log.Fatal("-spp, -move-spp and -bounces go from 1 to 64")
	}
	n := [3]uint32{uint32(*spp), uint32(*moveSPP), uint32(*bounces)} //nolint:gosec // range checked above
	if err := run(params{Bounces: n[2]}, n[0], n[1], *aperture, *frames, *backend); err != nil {
		log.Fatal(err)
	}
}

func run(p params, spp, moveSPP uint32, aperture float64, limit int, backend string) error {
	w, err := window.Open("gowgpu raytracing", backend)
	if err != nil {
		return err
	}
	defer w.Close()
	spheres := scene()
	p.Count = uint32(len(spheres)) //nolint:gosec // a few hundred
	s, err := newTracer(w.Device, spheres, w.Format)
	if err != nil {
		return err
	}
	defer s.release()

	// The camera orbits the middle of the scene, from where RTIOW looks.
	var (
		cam       = orbit{yaw: math.Atan2(3, 13), pitch: math.Asin(2 / math.Sqrt(182)), dist: math.Sqrt(182)}
		dragging  bool
		lastX     float64
		lastY     float64
		buf       []byte
		lastW     int
		lastH     int
		lastOrbit orbit
	)
	w.Win.SetScrollCallback(func(_ *glfw.Window, _, dy float64) {
		cam.dist = min(max(cam.dist*math.Pow(0.9, dy), 2), 40)
	})
	return w.Run(limit, func(view *wgpu.TextureView, width, height int, _ float32) error {
		x, y := w.Win.GetCursorPos()
		if w.Win.GetMouseButton(glfw.MouseButtonLeft) == glfw.Press {
			if dragging {
				cam.yaw += (x - lastX) * 0.005
				cam.pitch = min(max(cam.pitch+(y-lastY)*0.005, 0.02), 1.5)
			}
			dragging = true
		} else {
			dragging = false
		}
		lastX, lastY = x, y

		if width != lastW || height != lastH {
			if err := s.resize(w.Device, width, height); err != nil {
				return err
			}
			lastW, lastH = width, height
			p.Frame = 0
		}
		// ponytail: more samples while moving, reprojecting the old ones would
		// hide the grain better
		p.SPP = spp
		if cam != lastOrbit {
			lastOrbit = cam
			p.Frame, p.SPP = 0, moveSPP
		}
		p.Size = [2]uint32{uint32(width), uint32(height)} //nolint:gosec // GLFW sizes are positive
		p.Origin, p.Corner, p.Horiz, p.Vert, p.U, p.V = cam.lens(float64(width)/float64(height), aperture)
		buf, _ = binary.Append(buf[:0], binary.LittleEndian, &p)
		if err := w.Device.Queue().WriteBuffer(s.uniforms, 0, buf); err != nil {
			return err
		}
		trace := p.Frame < maxFrames
		if err := s.frame(w.Device, view, width, height, trace); err != nil {
			return err
		}
		if trace {
			p.Frame++
		}
		return nil
	})
}

// scene is RTIOW's final scene: a grid of small random spheres around three
// big ones, glass in the middle, diffuse brown behind and metal in front. The
// same seed gives the same scene every run.
func scene() []sphere {
	r := rand.New(rand.NewPCG(1, 2)) //nolint:gosec // not a secret
	f := func() float32 { return r.Float32() }
	spheres := []sphere{
		{Pos: [4]float32{0, 1, 0, 1}, Albedo: [4]float32{1, 1, 1, glass}, Mat: [4]float32{0, 1.5}},
		{Pos: [4]float32{-4, 1, 0, 1}, Albedo: [4]float32{0.4, 0.2, 0.1, diffuse}},
		{Pos: [4]float32{4, 1, 0, 1}, Albedo: [4]float32{0.7, 0.6, 0.5, metal}},
	}
	for a := -11; a < 11; a++ {
		for b := -11; b < 11; b++ {
			x, z := float32(a)+0.9*f(), float32(b)+0.9*f()
			if dx, dz := x-4, z; dx*dx+dz*dz < 0.81 {
				continue
			}
			s := sphere{Pos: [4]float32{x, 0.2, z, 0.2}}
			switch pick := f(); {
			case pick < 0.8:
				s.Albedo = [4]float32{f() * f(), f() * f(), f() * f(), diffuse}
			case pick < 0.95:
				s.Albedo = [4]float32{0.5 + f()/2, 0.5 + f()/2, 0.5 + f()/2, metal}
				s.Mat[0] = f() / 2
			default:
				s.Albedo = [4]float32{1, 1, 1, glass}
				s.Mat[1] = 1.5
			}
			spheres = append(spheres, s)
		}
	}
	return spheres
}

// orbit is a camera looking at the middle of the scene from dist away.
type orbit struct{ yaw, pitch, dist float64 }

type vec3 [3]float64

func (a vec3) add(b vec3) vec3      { return vec3{a[0] + b[0], a[1] + b[1], a[2] + b[2]} }
func (a vec3) scale(s float64) vec3 { return vec3{a[0] * s, a[1] * s, a[2] * s} }
func (a vec3) cross(b vec3) vec3 {
	return vec3{a[1]*b[2] - a[2]*b[1], a[2]*b[0] - a[0]*b[2], a[0]*b[1] - a[1]*b[0]}
}
func (a vec3) unit() vec3 { return a.scale(1 / math.Sqrt(a[0]*a[0]+a[1]*a[1]+a[2]*a[2])) }
func (a vec3) f32(w float64) [4]float32 {
	return [4]float32{float32(a[0]), float32(a[1]), float32(a[2]), float32(w)}
}

// lens returns RTIOW's camera, vertical field of view 20°, focused on the
// middle of the scene: the lens centre with its radius, the bottom left corner
// of the focus plane, the plane across and up, and the lens' axes.
func (o orbit) lens(aspect, aperture float64) (origin, corner, horiz, vert, u, v [4]float32) {
	const vfov = 20 * math.Pi / 180
	from := vec3{math.Cos(o.pitch) * math.Cos(o.yaw), math.Sin(o.pitch), math.Cos(o.pitch) * math.Sin(o.yaw)}.scale(o.dist)
	back := from.unit()
	right := vec3{0, 1, 0}.cross(back).unit()
	up := back.cross(right)
	h := 2 * math.Tan(vfov/2) * o.dist
	hv, vv := right.scale(h*aspect), up.scale(h)
	c := from.add(hv.scale(-0.5)).add(vv.scale(-0.5)).add(back.scale(-o.dist))
	return from.f32(aperture / 2), c.f32(0), hv.f32(0), vv.f32(0), right.f32(0), up.f32(0)
}

// tracer is the GPU side: the spheres, the parameters, the per pixel sums
// and the two pipelines that write and show them.
type tracer struct {
	spheres, uniforms, accum *wgpu.Buffer
	trace                    *wgpu.ComputePipeline
	show                     *wgpu.RenderPipeline
	traceGroup, showGroup    *wgpu.BindGroup
}

func newTracer(device *wgpu.Device, spheres []sphere, format gputypes.TextureFormat) (*tracer, error) {
	t := &tracer{}
	data, _ := binary.Append(nil, binary.LittleEndian, spheres)
	var err error
	if t.spheres, err = device.CreateBuffer(&wgpu.BufferDescriptor{
		Label: "spheres",
		Size:  uint64(len(data)),
		Usage: wgpu.BufferUsageStorage | wgpu.BufferUsageCopyDst,
	}); err != nil {
		return nil, err
	}
	if err = device.Queue().WriteBuffer(t.spheres, 0, data); err != nil {
		t.release()
		return nil, err
	}
	if t.uniforms, err = device.CreateBuffer(&wgpu.BufferDescriptor{
		Label: "params",
		Size:  uint64(binary.Size(params{})), //nolint:gosec // a struct is not negative
		Usage: wgpu.BufferUsageUniform | wgpu.BufferUsageCopyDst,
	}); err != nil {
		t.release()
		return nil, err
	}

	cs, err := device.CreateShaderModule(&wgpu.ShaderModuleDescriptor{WGSL: traceWGSL})
	if err != nil {
		t.release()
		return nil, err
	}
	defer cs.Release()
	if t.trace, err = device.CreateComputePipeline(&wgpu.ComputePipelineDescriptor{
		Label: "trace", Module: cs, EntryPoint: "main",
	}); err != nil {
		t.release()
		return nil, err
	}

	rs, err := device.CreateShaderModule(&wgpu.ShaderModuleDescriptor{WGSL: showWGSL})
	if err != nil {
		t.release()
		return nil, err
	}
	defer rs.Release()
	if t.show, err = device.CreateRenderPipeline(&wgpu.RenderPipelineDescriptor{
		Label:  "show",
		Vertex: wgpu.VertexState{Module: rs, EntryPoint: "vs_main"},
		Fragment: &wgpu.FragmentState{
			Module:     rs,
			EntryPoint: "fs_main",
			Targets:    []gputypes.ColorTargetState{{Format: format, WriteMask: gputypes.ColorWriteMaskAll}},
		},
	}); err != nil {
		t.release()
		return nil, err
	}
	return t, nil
}

// resize makes the sums for a frame of width by height pixels, and the bind
// groups that hand them to the pipelines.
func (t *tracer) resize(device *wgpu.Device, width, height int) error {
	t.releaseSized()
	var err error
	if t.accum, err = device.CreateBuffer(&wgpu.BufferDescriptor{
		Label: "accum",
		Size:  uint64(width) * uint64(height) * pixelBytes, //nolint:gosec // GLFW sizes are positive
		Usage: wgpu.BufferUsageStorage,
	}); err != nil {
		return err
	}
	if t.traceGroup, err = device.CreateBindGroup(&wgpu.BindGroupDescriptor{
		Layout: t.trace.GetBindGroupLayout(0),
		Entries: []wgpu.BindGroupEntry{
			{Binding: 0, Buffer: t.uniforms},
			{Binding: 1, Buffer: t.spheres},
			{Binding: 2, Buffer: t.accum},
		},
	}); err != nil {
		return err
	}
	t.showGroup, err = device.CreateBindGroup(&wgpu.BindGroupDescriptor{
		Layout: t.show.GetBindGroupLayout(0),
		Entries: []wgpu.BindGroupEntry{
			{Binding: 0, Buffer: t.uniforms},
			{Binding: 1, Buffer: t.accum},
		},
	})
	return err
}

// frame adds samples to the sums when trace is set and draws their average
// into view, in one submit.
func (t *tracer) frame(device *wgpu.Device, view *wgpu.TextureView, width, height int, trace bool) error {
	enc, err := device.CreateCommandEncoder(nil)
	if err != nil {
		return err
	}
	if trace {
		cp, err := enc.BeginComputePass(nil)
		if err != nil {
			return err
		}
		cp.SetPipeline(t.trace)
		cp.SetBindGroup(0, t.traceGroup, nil)
		cp.Dispatch(uint32((width+workgroup-1)/workgroup), uint32((height+workgroup-1)/workgroup), 1) //nolint:gosec // GLFW sizes are positive
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
	rp.SetPipeline(t.show)
	rp.SetBindGroup(0, t.showGroup, nil)
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

func (t *tracer) releaseSized() {
	for _, g := range []*wgpu.BindGroup{t.showGroup, t.traceGroup} {
		if g != nil {
			g.Release()
		}
	}
	if t.accum != nil {
		t.accum.Release()
	}
	t.showGroup, t.traceGroup, t.accum = nil, nil, nil
}

func (t *tracer) release() {
	t.releaseSized()
	if t.show != nil {
		t.show.Release()
	}
	if t.trace != nil {
		t.trace.Release()
	}
	if t.uniforms != nil {
		t.uniforms.Release()
	}
	if t.spheres != nil {
		t.spheres.Release()
	}
}
