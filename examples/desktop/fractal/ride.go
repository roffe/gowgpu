package main

import (
	"math"
	"math/big"
	"math/rand/v2"
	"slices"
)

const (
	startScale = 3.0 / 800 // the whole set across 800 pixels
	// ponytail: f32 deltas underflow below about 1e-37, so a ride starts over
	// before; an extended exponent type in the shader would go on
	minScale = 1e-34
	maxOrbit = 50000
	// a new target every this many halvings of the scale
	retargetEvery = 3
	sampleCols    = 48
	sampleRows    = 30
)

// ride is the flight into the Mandelbrot set. It zooms in on a target and,
// every few halvings of the scale, samples the view and picks a new target at
// random among the spots with the most detail, so every ride goes somewhere
// else. The target is also the reference point the shader's perturbation
// iterates against.
type ride struct {
	speed float64 // halvings of the scale a second
	iters float64 // times the iterations the view seems to need

	rng    *rand.Rand
	gen    int // which ride this is, so a late sample of the one before is dropped
	scale  float64
	center bigc         // the middle of the view
	target bigc         // where it heads
	orbit  []complex128 // the target's
	need   int          // iterations the last sample needed
	hue    float32
	since  float64 // seconds since the last target
	busy   bool    // a sample is running
	next   chan retarget
}

// retarget is what a sample of the view found.
type retarget struct {
	gen    int
	target bigc
	orbit  []complex128
	need   int
	ok     bool
}

func newRide(speed float64, seed uint64) *ride {
	r := &ride{speed: speed, iters: 1, rng: rand.New(rand.NewPCG(seed, seed^0x9e3779b97f4a7c15)), next: make(chan retarget, 1)} //nolint:gosec // not a secret
	r.restart()
	return r
}

// restart begins a new ride from the whole set, with a palette of its own.
func (r *ride) restart() {
	r.gen++
	r.scale = startScale
	r.center = cplx(-0.6, 0, 64)
	r.target = r.center
	r.orbit = referenceOrbit(r.target, maxOrbit)
	r.need, r.since, r.busy = 500, retargetEvery, false
	r.hue = r.rng.Float32()
}

// step moves the ride on by dt seconds: the scale shrinks, and the centre
// closes in on the target twice as fast, so the target ends up in the middle.
func (r *ride) step(dt float64) {
	r.scale *= math.Exp2(-r.speed * dt)
	if r.scale < minScale {
		r.restart()
		return
	}
	r.since += r.speed * dt
	a := 1 - math.Exp2(-2*r.speed*dt)
	d := r.target.sub(r.center)
	r.center = r.center.add(d.scaled(a, precFor(r.scale)))
}

// offset is from the target to the middle of the view.
func (r *ride) offset() [2]float32 {
	re, im := r.center.sub(r.target).f64()
	return [2]float32{float32(re), float32(im)}
}

func (r *ride) iterations() int {
	return min(max(int(r.iters*float64(r.need)*1.5), 300), maxOrbit-1)
}

// poll starts a sample of the view of w by h pixels when one is due, and
// returns the target and orbit it found once it is done, nil until then.
func (r *ride) poll(w, h int) []complex128 {
	select {
	case n := <-r.next:
		r.busy = false
		if n.gen == r.gen && n.ok {
			r.target, r.orbit, r.need, r.since = n.target, n.orbit, n.need, 0
			return r.orbit
		}
	default:
	}
	if !r.busy && r.since >= retargetEvery {
		r.busy = true
		go func(gen int, center, target bigc, orbit []complex128, scale float64, maxIter int, seed uint64) {
			n := sample(center, target, orbit, scale, w, h, maxIter, rand.New(rand.NewPCG(seed, seed))) //nolint:gosec // not a secret
			n.gen = gen
			r.next <- n
		}(r.gen, r.center, r.target, r.orbit, r.scale, min(2*r.iterations(), maxOrbit-1), r.rng.Uint64())
	}
	return nil
}

// sample iterates a grid over the view the way the shader does, and picks a
// new target at random among the escaping points of the middle of the view
// that took the most iterations, which lie on the edge of the set where the
// detail is. Its orbit is iterated in as many bits as the scale needs.
func sample(center, ref bigc, orbit []complex128, scale float64, w, h, maxIter int, rng *rand.Rand) retarget {
	ore, oim := center.sub(ref).f64()
	off := complex(ore, oim)
	type cell struct {
		x, y   float64 // pixels from the middle
		escape int
	}
	var cells []cell
	need := 0
	for j := range sampleRows {
		for i := range sampleCols {
			x := ((float64(i)+0.5)/sampleCols - 0.5) * float64(w)
			y := ((float64(j)+0.5)/sampleRows - 0.5) * float64(h)
			e := escape(orbit, complex(x*scale, -y*scale)+off, maxIter)
			if e < maxIter {
				need = max(need, e)
				if math.Abs(x) < 0.3*float64(w) && math.Abs(y) < 0.3*float64(h) {
					cells = append(cells, cell{x, y, e})
				}
			}
		}
	}
	if len(cells) == 0 {
		return retarget{} // nothing escapes in the middle, stay the course
	}
	slices.SortFunc(cells, func(a, b cell) int { return b.escape - a.escape })
	pick := cells[rng.IntN(max(len(cells)/8, 1))]
	prec := precFor(scale)
	target := center.add(cplx(pick.x*scale, -pick.y*scale, prec))
	return retarget{target: target, orbit: referenceOrbit(target, maxOrbit), need: need, ok: true}
}

// escape is how many iterations c = ref + dc takes to escape, iterating only
// dz against the reference orbit, as the shader does.
func escape(orbit []complex128, dc complex128, maxIter int) int {
	var dz complex128
	n := 0
	for i := range maxIter {
		dz = (2*orbit[n]+dz)*dz + dc
		n++
		z := orbit[n] + dz
		zz := real(z)*real(z) + imag(z)*imag(z)
		if zz > 256 {
			return i
		}
		if zz < real(dz)*real(dz)+imag(dz)*imag(dz) || n+1 >= len(orbit) {
			dz, n = z, 0
		}
	}
	return maxIter
}

// referenceOrbit iterates c in its own precision until it escapes or n points.
func referenceOrbit(c bigc, n int) []complex128 {
	prec := max(c.re.Prec(), 64)
	z := cplx(0, 0, prec)
	orbit := make([]complex128, 0, n)
	for range n {
		re, im := z.f64()
		orbit = append(orbit, complex(re, im))
		if re*re+im*im > 4 {
			break
		}
		z = z.mul(z).add(c)
	}
	return orbit
}

// precFor is the bits a point needs to tell pixels apart at scale.
func precFor(scale float64) uint {
	return uint(64 + max(math.Log2(startScale/scale), 0) + 32)
}

// bigc is a complex number in big floats.
type bigc struct{ re, im *big.Float }

func cplx(re, im float64, prec uint) bigc {
	return bigc{new(big.Float).SetPrec(prec).SetFloat64(re), new(big.Float).SetPrec(prec).SetFloat64(im)}
}

func (a bigc) prec() uint { return max(a.re.Prec(), a.im.Prec()) }

func (a bigc) bf(b bigc) *big.Float { return new(big.Float).SetPrec(max(a.prec(), b.prec())) }

func (a bigc) add(b bigc) bigc { return bigc{a.bf(b).Add(a.re, b.re), a.bf(b).Add(a.im, b.im)} }

func (a bigc) sub(b bigc) bigc { return bigc{a.bf(b).Sub(a.re, b.re), a.bf(b).Sub(a.im, b.im)} }

func (a bigc) mul(b bigc) bigc {
	t := a.bf(b)
	return bigc{
		a.bf(b).Sub(t.Mul(a.re, b.re), a.bf(b).Mul(a.im, b.im)),
		a.bf(b).Add(a.bf(b).Mul(a.re, b.im), a.bf(b).Mul(a.im, b.re)),
	}
}

// scaled is a times s, in at least prec bits.
func (a bigc) scaled(s float64, prec uint) bigc {
	p := max(a.prec(), prec)
	k := new(big.Float).SetPrec(p).SetFloat64(s)
	return bigc{new(big.Float).SetPrec(p).Mul(a.re, k), new(big.Float).SetPrec(p).Mul(a.im, k)}
}

func (a bigc) f64() (float64, float64) {
	re, _ := a.re.Float64()
	im, _ := a.im.Float64()
	return re, im
}
