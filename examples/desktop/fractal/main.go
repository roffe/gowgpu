// Command fractal shows two fractals in a window, both drawn by a single
// fragment shader over the whole frame: the Mandelbulb, ray marched with a
// distance estimate, its power slowly breathing, and an endless ride into
// the Mandelbrot set.
//
// The ride zooms in on a target and every few halvings of the scale samples
// the view and picks a new target at random where the detail is, so each
// ride goes somewhere else, with a palette of its own. The target's orbit is
// iterated on the CPU in as many bits as the zoom needs, and the shader
// iterates only how far each pixel's orbit is from it, which fits in f32
// (perturbation, with Zhuoran's rebasing where that breaks down). Near 1e-34
// a pixel f32 itself runs out and a new ride starts.
//
// On the Mandelbulb drag with the left mouse button to orbit and scroll to
// move closer. On the ride scroll to go faster or slower, and [ and ] halve
// or double the iterations. M switches between the two, P pauses, R starts
// over.
//
//	cd examples/desktop
//	go run ./fractal [-mandelbrot] [-speed 1] [-seed n] [-backend vulkan|dx12|metal|gl] [-frames 300]
//
// -frames exits after that many frames, a smoke test that walks the whole
// setup and teardown.
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

// params mirrors Params in the shader.
type params struct {
	Origin  [4]float32 // the camera, w the Mandelbulb's power
	Forward [4]float32 // w is 1 for the Mandelbrot set
	Right   [4]float32 // scaled by the field of view and aspect
	Up      [4]float32
	View    [2]float32 // the window in pixels
	MaxIter uint32     // for the Mandelbrot set
	RefLen  uint32     // points of the reference orbit
	Scale   float32    // Mandelbrot units per pixel
	Hue     float32    // picks the Mandelbrot palette, a new one each ride
	Offset  [2]float32 // from the reference point to the middle of the view
}

const shaderWGSL = `
struct Params {
    origin: vec4<f32>,
    forward: vec4<f32>,
    right: vec4<f32>,
    up: vec4<f32>,
    view: vec2<f32>,
    max_iter: u32,
    ref_len: u32,
    scale: f32,
    hue: f32,
    offset: vec2<f32>,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> orbit: array<vec2<f32>>; // the reference point's

@vertex
fn vs_main(@builtin(vertex_index) i: u32) -> @builtin(position) vec4<f32> {
    let xy = vec2<f32>(f32((i << 1u) & 2u), f32(i & 2u));
    return vec4<f32>(xy * 2.0 - 1.0, 0.0, 1.0);
}

// palette is Inigo Quilez's cosine palette.
fn palette(t: f32, d: vec3<f32>) -> vec3<f32> {
    return 0.5 + 0.5 * cos(6.2831853 * (t + d));
}

// bulb returns the distance estimate to the Mandelbulb from p and an orbit
// trap, how close the orbit came to the origin, to colour it by.
fn bulb(p: vec3<f32>) -> vec2<f32> {
    let power = params.origin.w;
    var z = p;
    var dr = 1.0;
    var r = length(z);
    var trap = 1e9;
    for (var i = 0; i < 10; i++) {
        if (r > 2.0) {
            break;
        }
        let theta = acos(clamp(z.z / max(r, 1e-6), -1.0, 1.0)) * power;
        let phi = atan2(z.y, z.x) * power;
        dr = pow(r, power - 1.0) * power * dr + 1.0;
        z = pow(r, power) * vec3<f32>(sin(theta) * cos(phi), sin(phi) * sin(theta), cos(theta)) + p;
        trap = min(trap, dot(z, z));
        r = length(z);
    }
    return vec2<f32>(0.5 * log(max(r, 1e-6)) * r / dr, trap);
}

fn normal(p: vec3<f32>, eps: f32) -> vec3<f32> {
    let k = vec2<f32>(1.0, -1.0);
    return normalize(k.xyy * bulb(p + k.xyy * eps).x + k.yyx * bulb(p + k.yyx * eps).x
        + k.yxy * bulb(p + k.yxy * eps).x + k.xxx * bulb(p + k.xxx * eps).x);
}

fn background(rd: vec3<f32>) -> vec3<f32> {
    return mix(vec3<f32>(0.02, 0.02, 0.05), vec3<f32>(0.12, 0.1, 0.2), 0.5 + 0.5 * rd.y);
}

fn mandelbulb(pos: vec2<f32>) -> vec3<f32> {
    let uv = (2.0 * pos - params.view) / params.view.y * vec2<f32>(1.0, -1.0);
    let ro = params.origin.xyz;
    let rd = normalize(params.forward.xyz + uv.x * params.right.xyz + uv.y * params.up.xyz);

    // march only inside the sphere of radius 1.25 that holds the bulb
    let b = dot(ro, rd);
    let disc = b * b - dot(ro, ro) + 1.25 * 1.25;
    if (disc < 0.0) {
        return background(rd);
    }
    var t = max(-b - sqrt(disc), 0.0);
    let far = -b + sqrt(disc);
    let pixel = 1.5 / params.view.y; // how wide a pixel is a unit away
    var steps = 0;
    var hit = false;
    var trap = 0.0;
    for (; steps < 160; steps++) {
        let d = bulb(ro + rd * t);
        if (d.x < pixel * t) {
            hit = true;
            trap = d.y;
            break;
        }
        t += d.x;
        if (t > far) {
            break;
        }
    }
    if (!hit) {
        return background(rd);
    }

    let p = ro + rd * t;
    let n = normal(p, pixel * t * 0.5);
    let sun = normalize(vec3<f32>(0.6, 0.8, -0.4));
    let ao = 1.0 - f32(steps) / 160.0;
    let base = palette(0.6 + 0.4 * sqrt(trap), vec3<f32>(0.0, 0.15, 0.3));
    let diffuse = max(dot(n, sun), 0.0);
    let rim = pow(1.0 - max(dot(n, -rd), 0.0), 3.0);
    let spec = pow(max(dot(reflect(rd, n), sun), 0.0), 32.0);
    var c = base * (0.15 + 0.85 * diffuse) * ao * ao + vec3<f32>(0.6, 0.5, 1.0) * rim * 0.4 * ao + spec * 0.4;
    return mix(c, background(rd), smoothstep(2.0, 6.0, t));
}

fn cmul(a: vec2<f32>, b: vec2<f32>) -> vec2<f32> {
    return vec2<f32>(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x);
}

// mandelbrot iterates dz, how far the pixel's orbit is from the reference
// orbit, Z: z = Z + dz, so dz' = (2Z + dz) dz + dc. Where z comes closer to 0
// than dz is, or Z runs out, dz takes over as z and Z starts over from 0.
fn mandelbrot(pos: vec2<f32>) -> vec3<f32> {
    let dc = (pos - 0.5 * params.view) * params.scale * vec2<f32>(1.0, -1.0) + params.offset;
    var dz = vec2<f32>(0.0);
    var z = vec2<f32>(0.0);
    var n = 0u;
    var i = 0u;
    for (; i < params.max_iter; i++) {
        dz = cmul(2.0 * orbit[n] + dz, dz) + dc;
        n++;
        z = orbit[n] + dz;
        if (dot(z, z) > 256.0) {
            break;
        }
        if (dot(z, z) < dot(dz, dz) || n + 1u >= params.ref_len) {
            dz = z;
            n = 0u;
        }
    }
    if (i == params.max_iter) {
        return vec3<f32>(0.0);
    }
    let smooth_i = f32(i) - log2(log2(dot(z, z))) + 4.0;
    // the square root stretches the colour cycles as the counts grow, so deep
    // views do not turn to noise; each ride shifts the channels its own way
    return palette(sqrt(max(smooth_i, 0.0)) * 0.3, fract(vec3<f32>(0.0, 0.1, 0.2) + params.hue * vec3<f32>(1.0, 2.0, 3.0)));
}

@fragment
fn fs_main(@builtin(position) pos: vec4<f32>) -> @location(0) vec4<f32> {
    var c: vec3<f32>;
    if (params.forward.w > 0.5) {
        c = mandelbrot(pos.xy);
    } else {
        c = mandelbulb(pos.xy);
    }
    return vec4<f32>(pow(c, vec3<f32>(1.0 / 2.2)), 1.0);
}
`

func main() {
	brot := flag.Bool("mandelbrot", false, "start on the ride into the Mandelbrot set rather than the Mandelbulb")
	speed := flag.Float64("speed", 1, "how many times a second the ride halves the scale")
	seed := flag.Uint64("seed", 0, "the ride's random seed, 0 picks one; the same seed takes the same rides")
	backend := flag.String("backend", "", "vulkan, dx12, metal or gl; by default whichever of vulkan, dx12 and metal is there")
	frames := flag.Int("frames", 0, "exit after this many frames, 0 runs until the window is closed")
	flag.Parse()
	if *seed == 0 {
		*seed = rand.Uint64() //nolint:gosec // not a secret
	}
	log.Printf("-seed %d", *seed)
	if err := run(*brot, newRide(*speed, *seed), *frames, *backend); err != nil {
		log.Fatal(err)
	}
}

// view is where the Mandelbulb is looked at from.
type view struct{ yaw, pitch, dist float64 }

var bulbStart = view{yaw: 0.6, pitch: 0.4, dist: 3.2}

func run(brot bool, r *ride, limit int, backend string) error {
	w, err := window.Open("gowgpu fractal", backend)
	if err != nil {
		return err
	}
	defer w.Close()
	f, err := newFractal(w.Device, w.Format)
	if err != nil {
		return err
	}
	defer f.release()

	var orbitBuf []byte
	upload := func(orbit []complex128) error {
		orbitBuf = orbitBuf[:0]
		for _, z := range orbit {
			orbitBuf = binary.LittleEndian.AppendUint32(orbitBuf, math.Float32bits(float32(real(z))))
			orbitBuf = binary.LittleEndian.AppendUint32(orbitBuf, math.Float32bits(float32(imag(z))))
		}
		return w.Device.Queue().WriteBuffer(f.orbit, 0, orbitBuf)
	}
	if err := upload(r.orbit); err != nil {
		return err
	}
	gen := r.gen

	var (
		v        = bulbStart
		paused   bool
		t        float64 // the Mandelbulb's breathing
		lastX    float64
		lastY    float64
		dragging bool
		buf      []byte
		p        params
	)
	w.Win.SetKeyCallback(func(_ *glfw.Window, key glfw.Key, _ int, action glfw.Action, _ glfw.ModifierKey) {
		if action != glfw.Press {
			return
		}
		switch key {
		case glfw.KeyM:
			brot = !brot
		case glfw.KeyP:
			paused = !paused
		case glfw.KeyR:
			v, t = bulbStart, 0
			r.restart()
		case glfw.KeyLeftBracket:
			r.iters = max(r.iters/2, 1.0/16)
			log.Printf("%d iterations", r.iterations())
		case glfw.KeyRightBracket:
			r.iters = min(r.iters*2, 64)
			log.Printf("%d iterations", r.iterations())
		}
	})
	w.Win.SetScrollCallback(func(_ *glfw.Window, _, dy float64) {
		if brot {
			r.speed = min(max(r.speed*math.Pow(1.25, dy), 0.1), 8)
			return
		}
		v.dist = min(max(v.dist*math.Pow(0.92, dy), 1.3), 8)
	})
	return w.Run(limit, func(target *wgpu.TextureView, width, height int, dt float32) error {
		x, y := w.Win.GetCursorPos()
		if !brot && w.Win.GetMouseButton(glfw.MouseButtonLeft) == glfw.Press {
			if dragging {
				v.yaw -= (x - lastX) * 0.005
				v.pitch = min(max(v.pitch+(y-lastY)*0.005, -1.5), 1.5)
			}
			dragging = true
		} else {
			dragging = false
		}
		lastX, lastY = x, y
		if !paused {
			if brot {
				r.step(float64(dt))
			} else {
				t += float64(dt)
			}
		}

		p.Origin, p.Forward, p.Right, p.Up = v.camera(8 + 2*math.Sin(t*0.3))
		if brot {
			p.Forward[3] = 1
		}
		p.View = [2]float32{float32(width), float32(height)}
		if brot {
			orbit := r.poll(width, height)
			if r.gen != gen { // a new ride, from the whole set again
				orbit, gen = r.orbit, r.gen
			}
			if orbit != nil {
				if err := upload(orbit); err != nil {
					return err
				}
			}
		}
		p.MaxIter, p.RefLen = uint32(r.iterations()), uint32(len(r.orbit)) //nolint:gosec // capped at maxOrbit
		p.Scale, p.Hue, p.Offset = float32(r.scale), r.hue, r.offset()
		buf, _ = binary.Append(buf[:0], binary.LittleEndian, &p)
		if err := w.Device.Queue().WriteBuffer(f.uniforms, 0, buf); err != nil {
			return err
		}
		return f.frame(w.Device, target)
	})
}

// camera returns the Mandelbulb camera orbiting the origin: where it is, with
// the power in w, where it looks, and its right and up axes scaled to a field
// of view of about 50° across the height.
func (v view) camera(power float64) (origin, forward, right, up [4]float32) {
	cp, sp := math.Cos(v.pitch), math.Sin(v.pitch)
	o := [3]float64{v.dist * cp * math.Sin(v.yaw), v.dist * sp, -v.dist * cp * math.Cos(v.yaw)}
	f := [3]float64{-o[0] / v.dist, -o[1] / v.dist, -o[2] / v.dist}
	r := [3]float64{f[2], 0, -f[0]} // flat on the ground
	rl := math.Hypot(r[0], r[2])
	r = [3]float64{r[0] / rl, 0, r[2] / rl}
	u := [3]float64{f[1]*r[2] - f[2]*r[1], f[2]*r[0] - f[0]*r[2], f[0]*r[1] - f[1]*r[0]}
	const half = 0.45 // tan of half the field of view, the shader scales x by the aspect
	vec := func(a [3]float64, s, w float64) [4]float32 {
		return [4]float32{float32(a[0] * s), float32(a[1] * s), float32(a[2] * s), float32(w)}
	}
	return vec(o, 1, power), vec(f, 1, 0), vec(r, half, 0), vec(u, half, 0)
}

// fractal is the GPU side: the parameters, the reference orbit and the
// pipeline that draws them.
type fractal struct {
	uniforms *wgpu.Buffer
	orbit    *wgpu.Buffer
	pipeline *wgpu.RenderPipeline
	group    *wgpu.BindGroup
}

func newFractal(device *wgpu.Device, format gputypes.TextureFormat) (*fractal, error) {
	f := &fractal{}
	var err error
	if f.uniforms, err = device.CreateBuffer(&wgpu.BufferDescriptor{
		Label: "params",
		Size:  uint64(binary.Size(params{})), //nolint:gosec // a struct is not negative
		Usage: wgpu.BufferUsageUniform | wgpu.BufferUsageCopyDst,
	}); err != nil {
		return nil, err
	}
	if f.orbit, err = device.CreateBuffer(&wgpu.BufferDescriptor{
		Label: "orbit",
		Size:  maxOrbit * 8,
		Usage: wgpu.BufferUsageStorage | wgpu.BufferUsageCopyDst,
	}); err != nil {
		f.release()
		return nil, err
	}
	sm, err := device.CreateShaderModule(&wgpu.ShaderModuleDescriptor{WGSL: shaderWGSL})
	if err != nil {
		f.release()
		return nil, err
	}
	defer sm.Release()
	if f.pipeline, err = device.CreateRenderPipeline(&wgpu.RenderPipelineDescriptor{
		Label:  "fractal",
		Vertex: wgpu.VertexState{Module: sm, EntryPoint: "vs_main"},
		Fragment: &wgpu.FragmentState{
			Module:     sm,
			EntryPoint: "fs_main",
			Targets:    []gputypes.ColorTargetState{{Format: format, WriteMask: gputypes.ColorWriteMaskAll}},
		},
	}); err != nil {
		f.release()
		return nil, err
	}
	if f.group, err = device.CreateBindGroup(&wgpu.BindGroupDescriptor{
		Layout:  f.pipeline.GetBindGroupLayout(0),
		Entries: []wgpu.BindGroupEntry{{Binding: 0, Buffer: f.uniforms}, {Binding: 1, Buffer: f.orbit}},
	}); err != nil {
		f.release()
		return nil, err
	}
	return f, nil
}

func (f *fractal) frame(device *wgpu.Device, target *wgpu.TextureView) error {
	enc, err := device.CreateCommandEncoder(nil)
	if err != nil {
		return err
	}
	rp, err := enc.BeginRenderPass(&wgpu.RenderPassDescriptor{
		ColorAttachments: []wgpu.RenderPassColorAttachment{{
			View:    target,
			LoadOp:  gputypes.LoadOpClear,
			StoreOp: gputypes.StoreOpStore,
		}},
	})
	if err != nil {
		return err
	}
	rp.SetPipeline(f.pipeline)
	rp.SetBindGroup(0, f.group, nil)
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

func (f *fractal) release() {
	if f.group != nil {
		f.group.Release()
	}
	if f.pipeline != nil {
		f.pipeline.Release()
	}
	if f.orbit != nil {
		f.orbit.Release()
	}
	if f.uniforms != nil {
		f.uniforms.Release()
	}
}
