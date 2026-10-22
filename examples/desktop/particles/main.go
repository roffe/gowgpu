// Command particles runs a GPU particle simulation in a window. A compute
// shader moves the particles every frame and a render pipeline draws them
// from the same buffer, instanced and additively blended, through gowgpu on
// whatever backend the machine has.
//
// The particles fall toward the mouse pointer and swirl around it. Hold a
// mouse button to push them away. While the pointer is outside the window
// they chase a point moving on its own.
//
//	cd examples/desktop
//	go run ./particles [-n 200000] [-backend vulkan|dx12|metal] [-frames 300]
//
// -frames exits after that many frames, a smoke test that walks the whole
// setup and teardown. The GL backend cannot run this one: it compiles compute
// shaders as GLSL 3.30, where naga declares a storage buffer as a uniform
// block, so the shader fails to compile.
package main

import (
	"encoding/binary"
	"flag"
	"fmt"
	"log"
	"math"
	"math/rand/v2"
	"time"

	"github.com/go-gl/glfw/v3.4/glfw"
	wgpu "github.com/roffe/gowgpu"
	"github.com/roffe/gowgpu/examples/desktop/window"
	"github.com/roffe/gowgpu/gputypes"
)

const (
	particleBytes = 16 // pos: vec2<f32>, vel: vec2<f32>
	workgroup     = 256
)

// params mirrors Params in the shaders.
type params struct {
	Target [2]float32 // where the particles fall to, in clip space
	Frame  [2]float32 // the window in pixels
	Dt     float32
	Time   float32
	Aspect float32 // width over height, so the pull looks the same both ways
	Count  uint32
	Repel  float32 // 1 while a mouse button is held
	_      [3]float32
}

const computeWGSL = `
struct Particle { pos: vec2<f32>, vel: vec2<f32> }

struct Params {
    target: vec2<f32>,
    frame: vec2<f32>,
    dt: f32,
    time: f32,
    aspect: f32,
    count: u32,
    repel: f32,
}

@group(0) @binding(0) var<storage, read_write> particles: array<Particle>;
@group(0) @binding(1) var<uniform> params: Params;

@compute @workgroup_size(256)
fn step(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = id.x;
    if (i >= params.count) {
        return;
    }
    var p = particles[i];

    // Square space: clip space stretched back to the window's aspect.
    var d = params.target - p.pos;
    d.x *= params.aspect;
    let r = max(length(d), 0.03);
    let dir = d / r;
    let pull = min(2.0 / (r * r), 60.0) * (1.0 - 2.0 * params.repel);
    let swirl = min(1.2 / r, 20.0);
    p.vel += (dir * pull + vec2<f32>(-dir.y, dir.x) * swirl) * params.dt;

    p.vel *= exp(-0.6 * params.dt);
    let speed = length(p.vel);
    if (speed > 3.0) {
        p.vel *= 3.0 / speed;
    }
    p.pos += vec2<f32>(p.vel.x / params.aspect, p.vel.y) * params.dt;

    if (abs(p.pos.x) > 1.0) {
        p.pos.x = clamp(p.pos.x, -1.0, 1.0);
        p.vel.x = -p.vel.x * 0.5;
    }
    if (abs(p.pos.y) > 1.0) {
        p.pos.y = clamp(p.pos.y, -1.0, 1.0);
        p.vel.y = -p.vel.y * 0.5;
    }
    particles[i] = p;
}
`

const renderWGSL = `
struct Params {
    target: vec2<f32>,
    frame: vec2<f32>,
    dt: f32,
    time: f32,
    aspect: f32,
    count: u32,
    repel: f32,
}

@group(0) @binding(0) var<uniform> params: Params;

struct Vertex {
    @builtin(position) pos: vec4<f32>,
    @location(0) color: vec4<f32>,
}

// Each particle is an instance and the six vertices make its quad, 3 px wide.
@vertex
fn vs_main(@builtin(vertex_index) vi: u32, @location(0) pos: vec2<f32>, @location(1) vel: vec2<f32>) -> Vertex {
    var corners = array<vec2<f32>, 6>(
        vec2<f32>(-1.0, -1.0), vec2<f32>(1.0, -1.0), vec2<f32>(-1.0, 1.0),
        vec2<f32>(-1.0, 1.0), vec2<f32>(1.0, -1.0), vec2<f32>(1.0, 1.0),
    );
    let heat = clamp(length(vel) / 2.5, 0.0, 1.0);
    let color = mix(vec3<f32>(0.1, 0.3, 1.0), vec3<f32>(1.0, 0.8, 0.4), heat);
    var out: Vertex;
    out.pos = vec4<f32>(pos + corners[vi] * 1.5 / params.frame, 0.0, 1.0);
    out.color = vec4<f32>(color * 0.3, 1.0);
    return out;
}

@fragment
fn fs_main(in: Vertex) -> @location(0) vec4<f32> {
    return in.color;
}
`

func main() {
	n := flag.Int("n", 200_000, "how many particles")
	backend := flag.String("backend", "", "vulkan, dx12 or metal; by default whichever is there")
	frames := flag.Int("frames", 0, "exit after this many frames, 0 runs until the window is closed")
	flag.Parse()
	if *n <= 0 || *n > math.MaxUint32 {
		log.Fatal("-n must be a positive 32-bit count")
	}
	if err := run(uint32(*n), *frames, *backend); err != nil { //nolint:gosec // range checked above
		log.Fatal(err)
	}
}

func run(n uint32, limit int, backend string) error {
	w, err := window.Open(fmt.Sprintf("gowgpu particles: %d", n), backend)
	if err != nil {
		return err
	}
	defer w.Close()
	s, err := newScene(w.Device, n, w.Format)
	if err != nil {
		return err
	}
	defer s.release()

	var (
		p      = params{Count: n}
		buf    []byte
		inside bool
		start  = time.Now()
	)
	w.Win.SetCursorEnterCallback(func(_ *glfw.Window, entered bool) { inside = entered })
	return w.Run(limit, func(view *wgpu.TextureView, width, height int, dt float32) error {
		p.Dt = dt
		p.Time = float32(time.Since(start).Seconds())
		p.Frame = [2]float32{float32(width), float32(height)}
		p.Aspect = float32(width) / float32(height)
		p.Target, p.Repel = target(w.Win, inside, p.Time)
		buf, _ = binary.Append(buf[:0], binary.LittleEndian, &p)
		if err := w.Device.Queue().WriteBuffer(s.uniforms, 0, buf); err != nil {
			return err
		}
		return s.frame(w.Device, view)
	})
}

// target is where the particles fall to: the pointer while it is in the
// window, else a point drifting on its own. Repel is 1 while a button is held.
func target(win *glfw.Window, inside bool, t float32) (pos [2]float32, repel float32) {
	if !inside {
		return [2]float32{0.6 * float32(math.Cos(float64(t)*0.7)), 0.6 * float32(math.Sin(float64(t)*1.1))}, 0
	}
	x, y := win.GetCursorPos()
	w, h := win.GetSize()
	if win.GetMouseButton(glfw.MouseButtonLeft) == glfw.Press || win.GetMouseButton(glfw.MouseButtonRight) == glfw.Press {
		repel = 1
	}
	return [2]float32{float32(2*x/float64(w) - 1), float32(1 - 2*y/float64(h))}, repel
}

// scene is the GPU side: the particles, the parameters and the two pipelines
// that read them.
type scene struct {
	n                         uint32
	particles, uniforms       *wgpu.Buffer
	compute                   *wgpu.ComputePipeline
	render                    *wgpu.RenderPipeline
	computeGroup, renderGroup *wgpu.BindGroup
}

func newScene(device *wgpu.Device, n uint32, format gputypes.TextureFormat) (*scene, error) {
	s := &scene{n: n}
	var err error
	if s.particles, err = device.CreateBuffer(&wgpu.BufferDescriptor{
		Label: "particles",
		Size:  uint64(n) * particleBytes,
		Usage: wgpu.BufferUsageStorage | wgpu.BufferUsageVertex | wgpu.BufferUsageCopyDst,
	}); err != nil {
		return nil, err
	}
	if s.uniforms, err = device.CreateBuffer(&wgpu.BufferDescriptor{
		Label: "params",
		Size:  uint64(binary.Size(params{})), //nolint:gosec // a struct is not negative
		Usage: wgpu.BufferUsageUniform | wgpu.BufferUsageCopyDst,
	}); err != nil {
		s.release()
		return nil, err
	}
	if err = device.Queue().WriteBuffer(s.particles, 0, seed(n)); err != nil {
		s.release()
		return nil, err
	}

	// Neither pipeline gets a layout: gowgpu derives one from the shader, and
	// GetBindGroupLayout hands it out to make the bind group with.
	cs, err := device.CreateShaderModule(&wgpu.ShaderModuleDescriptor{WGSL: computeWGSL})
	if err != nil {
		s.release()
		return nil, err
	}
	defer cs.Release()
	if s.compute, err = device.CreateComputePipeline(&wgpu.ComputePipelineDescriptor{
		Label: "step", Module: cs, EntryPoint: "step",
	}); err != nil {
		s.release()
		return nil, err
	}
	if s.computeGroup, err = device.CreateBindGroup(&wgpu.BindGroupDescriptor{
		Layout: s.compute.GetBindGroupLayout(0),
		Entries: []wgpu.BindGroupEntry{
			{Binding: 0, Buffer: s.particles},
			{Binding: 1, Buffer: s.uniforms},
		},
	}); err != nil {
		s.release()
		return nil, err
	}

	rs, err := device.CreateShaderModule(&wgpu.ShaderModuleDescriptor{WGSL: renderWGSL})
	if err != nil {
		s.release()
		return nil, err
	}
	defer rs.Release()
	additive := gputypes.BlendComponent{SrcFactor: gputypes.BlendFactorOne, DstFactor: gputypes.BlendFactorOne, Operation: gputypes.BlendOperationAdd}
	if s.render, err = device.CreateRenderPipeline(&wgpu.RenderPipelineDescriptor{
		Label: "draw",
		Vertex: wgpu.VertexState{
			Module:     rs,
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
			Module:     rs,
			EntryPoint: "fs_main",
			Targets: []gputypes.ColorTargetState{{
				Format:    format,
				Blend:     &gputypes.BlendState{Color: additive, Alpha: additive},
				WriteMask: gputypes.ColorWriteMaskAll,
			}},
		},
	}); err != nil {
		s.release()
		return nil, err
	}
	if s.renderGroup, err = device.CreateBindGroup(&wgpu.BindGroupDescriptor{
		Layout:  s.render.GetBindGroupLayout(0),
		Entries: []wgpu.BindGroupEntry{{Binding: 0, Buffer: s.uniforms}},
	}); err != nil {
		s.release()
		return nil, err
	}
	return s, nil
}

// seed spreads the particles over the window, each with a little speed.
func seed(n uint32) []byte {
	buf := make([]byte, 0, int(n)*particleBytes)
	for range n {
		for _, v := range [4]float32{rand.Float32()*2 - 1, rand.Float32()*2 - 1, (rand.Float32() - 0.5) * 0.4, (rand.Float32() - 0.5) * 0.4} { //nolint:gosec // not a secret
			buf = binary.LittleEndian.AppendUint32(buf, math.Float32bits(v))
		}
	}
	return buf
}

// frame steps the particles and draws them over a cleared frame, in one submit.
func (s *scene) frame(device *wgpu.Device, view *wgpu.TextureView) error {
	enc, err := device.CreateCommandEncoder(nil)
	if err != nil {
		return err
	}
	cp, err := enc.BeginComputePass(nil)
	if err != nil {
		return err
	}
	cp.SetPipeline(s.compute)
	cp.SetBindGroup(0, s.computeGroup, nil)
	cp.Dispatch((s.n+workgroup-1)/workgroup, 1, 1)
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
	rp.Draw(gputypes.DrawArgs{VertexCount: 6, InstanceCount: s.n})
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

func (s *scene) release() {
	if s.renderGroup != nil {
		s.renderGroup.Release()
	}
	if s.render != nil {
		s.render.Release()
	}
	if s.computeGroup != nil {
		s.computeGroup.Release()
	}
	if s.compute != nil {
		s.compute.Release()
	}
	if s.uniforms != nil {
		s.uniforms.Release()
	}
	if s.particles != nil {
		s.particles.Release()
	}
}
