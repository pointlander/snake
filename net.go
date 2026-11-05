// Copyright 2026 The Snake Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"math"
	"math/rand/v2"
	"runtime"
	"sync"
)

// network is a two-layer feedforward classifier.
// The map is window → hidden ReLU → class logits.
type network struct {
	window int
	hidden int
	nClass int
	mean   float32
	invStd float32
	w1     []float32 // hidden × window
	b1     []float32
	w2     []float32 // nClass × hidden
	b2     []float32
	x      []float32
	h      []float32
	z      []float32
}

func (n *network) predict(at func(feature int) float32) uint8 {
	for i := 0; i < n.window; i++ {
		n.x[i] = (at(i) - n.mean) * n.invStd
	}
	forward(n.w1, n.b1, n.w2, n.b2, n.x, n.h, nil, n.z, n.window, n.hidden, n.nClass)
	best := 0
	for c := 1; c < n.nClass; c++ {
		if n.z[c] > n.z[best] {
			best = c
		}
	}
	return uint8(best)
}

// forward writes ReLU activations to h and logits to z.
// pre receives the hidden pre-activations when non-nil.
func forward(w1, b1, w2, b2, x, h, pre, z []float32, window, hidden, nClass int) {
	for j := 0; j < hidden; j++ {
		s := b1[j]
		row := j * window
		for i := 0; i < window; i++ {
			s += w1[row+i] * x[i]
		}
		if pre != nil {
			pre[j] = s
		}
		if s < 0 {
			s = 0
		}
		h[j] = s
	}
	for c := 0; c < nClass; c++ {
		s := b2[c]
		row := c * hidden
		for j := 0; j < hidden; j++ {
			s += w2[row+j] * h[j]
		}
		z[c] = s
	}
}

func softmax(z []float32) {
	max := z[0]
	for _, v := range z[1:] {
		if v > max {
			max = v
		}
	}
	var sum float32
	for i, v := range z {
		e := float32(math.Exp(float64(v - max)))
		z[i] = e
		sum += e
	}
	inv := 1 / sum
	for i := range z {
		z[i] *= inv
	}
}

type grader struct {
	gw1   []float32
	gb1   []float32
	gw2   []float32
	gb2   []float32
	x     []float32
	h     []float32
	pre   []float32
	z     []float32
	dh    []float32
	loss  float64
	count int
}

func (g *grader) zero() {
	clear(g.gw1)
	clear(g.gb1)
	clear(g.gw2)
	clear(g.gb2)
	g.loss = 0
	g.count = 0
}

func (g *grader) accum(n *network, y uint8) {
	forward(n.w1, n.b1, n.w2, n.b2, g.x, g.h, g.pre, g.z, n.window, n.hidden, n.nClass)
	softmax(g.z)
	p := g.z[y]
	if p < 1e-12 {
		p = 1e-12
	}
	g.loss += -math.Log(float64(p))
	g.count++
	g.z[y] -= 1
	hidden := n.hidden
	window := n.window
	for c := 0; c < n.nClass; c++ {
		dz := g.z[c]
		g.gb2[c] += dz
		row := c * hidden
		for j := 0; j < hidden; j++ {
			g.gw2[row+j] += dz * g.h[j]
		}
	}
	for j := 0; j < hidden; j++ {
		var s float32
		for c := 0; c < n.nClass; c++ {
			s += n.w2[c*hidden+j] * g.z[c]
		}
		if g.pre[j] <= 0 {
			s = 0
		}
		g.dh[j] = s
		g.gb1[j] += s
	}
	for j := 0; j < hidden; j++ {
		dh := g.dh[j]
		if dh == 0 {
			continue
		}
		row := j * window
		for i := 0; i < window; i++ {
			g.gw1[row+i] += dh * g.x[i]
		}
	}
}

func rawFeature(pr []float32, stateAt []int32, ranks []float32, window, example, feature int) float32 {
	idx := example - window + feature
	if stateAt != nil {
		id := stateAt[idx]
		if id < 0 {
			return 0
		}
		return ranks[id]
	}
	return pr[idx]
}

func trainNetwork(pr []float32, stateAt []int32, ranks []float32, labels []uint8, examples []int, nClass int, cfg config, logf func(string, ...any)) *network {
	window := cfg.window
	hidden := cfg.hidden
	if hidden < 1 {
		hidden = 32
	}
	epochs := cfg.epochs
	if epochs < 1 {
		epochs = 1
	}
	batch := cfg.batch
	if batch < 1 {
		batch = 64
	}
	lr := float32(cfg.lr)
	if lr <= 0 {
		lr = 0.05
	}
	n := &network{
		window: window,
		hidden: hidden,
		nClass: nClass,
		invStd: 1,
		w1:     make([]float32, hidden*window),
		b1:     make([]float32, hidden),
		w2:     make([]float32, nClass*hidden),
		b2:     make([]float32, nClass),
		x:      make([]float32, window),
		h:      make([]float32, hidden),
		z:      make([]float32, nClass),
	}
	rng := rand.New(rand.NewPCG(cfg.seed+0x9e3779b97f4a7c15, cfg.seed^0x6a09e667f3bcc909))
	he(rng, n.w1, window)
	xavier(rng, n.w2, hidden)

	var sum, sumsq float64
	counted := 0
	for _, ex := range examples {
		for f := 0; f < window; f++ {
			v := float64(rawFeature(pr, stateAt, ranks, window, ex, f))
			sum += v
			sumsq += v * v
			counted++
		}
	}
	if counted > 0 {
		mean := sum / float64(counted)
		variance := sumsq/float64(counted) - mean*mean
		if variance < 1e-12 {
			variance = 1
		}
		n.mean = float32(mean)
		n.invStd = float32(1 / math.Sqrt(variance))
	}

	workers := runtime.GOMAXPROCS(0)
	if workers < 1 {
		workers = 1
	}
	if workers > len(examples) {
		workers = len(examples)
	}
	parts := make([]grader, workers)
	for i := range parts {
		parts[i] = grader{
			gw1: make([]float32, len(n.w1)),
			gb1: make([]float32, hidden),
			gw2: make([]float32, len(n.w2)),
			gb2: make([]float32, nClass),
			x:   make([]float32, window),
			h:   make([]float32, hidden),
			pre: make([]float32, hidden),
			z:   make([]float32, nClass),
			dh:  make([]float32, hidden),
		}
	}
	order := make([]int, len(examples))
	copy(order, examples)
	if logf != nil {
		logf("training network on %d examples  window %d  hidden %d  epochs %d", len(examples), window, hidden, epochs)
	}
	for epoch := 1; epoch <= epochs; epoch++ {
		for i := len(order) - 1; i > 0; i-- {
			j := rng.IntN(i + 1)
			order[i], order[j] = order[j], order[i]
		}
		var lossSum float64
		seen := 0
		step := batch * workers
		if step < 1 {
			step = 1
		}
		for start := 0; start < len(order); start += step {
			end := start + step
			if end > len(order) {
				end = len(order)
			}
			span := end - start
			var wg sync.WaitGroup
			for w := 0; w < workers; w++ {
				a := start + span*w/workers
				b := start + span*(w+1)/workers
				wg.Go(func() {
					g := &parts[w]
					g.zero()
					for i := a; i < b; i++ {
						ex := order[i]
						for f := 0; f < window; f++ {
							v := rawFeature(pr, stateAt, ranks, window, ex, f)
							g.x[f] = (v - n.mean) * n.invStd
						}
						g.accum(n, labels[ex])
					}
				})
			}
			wg.Wait()
			total := 0
			lossSum += parts[0].loss
			total += parts[0].count
			for w := 1; w < workers; w++ {
				add(parts[0].gw1, parts[w].gw1)
				add(parts[0].gb1, parts[w].gb1)
				add(parts[0].gw2, parts[w].gw2)
				add(parts[0].gb2, parts[w].gb2)
				lossSum += parts[w].loss
				total += parts[w].count
			}
			if total == 0 {
				continue
			}
			seen += total
			scale := lr / float32(total)
			descend(n.w1, parts[0].gw1, scale)
			descend(n.b1, parts[0].gb1, scale)
			descend(n.w2, parts[0].gw2, scale)
			descend(n.b2, parts[0].gb2, scale)
		}
		if logf != nil && seen > 0 {
			logf("epoch %d/%d  loss %.4f", epoch, epochs, lossSum/float64(seen))
		}
	}
	return n
}

func add(dst, src []float32) {
	for i := range dst {
		dst[i] += src[i]
	}
}

func descend(w, g []float32, scale float32) {
	for i := range w {
		w[i] -= scale * g[i]
	}
}

func he(rng *rand.Rand, w []float32, fanIn int) {
	if fanIn < 1 {
		fanIn = 1
	}
	std := math.Sqrt(2 / float64(fanIn))
	for i := range w {
		w[i] = float32(rng.NormFloat64() * std)
	}
}

func xavier(rng *rand.Rand, w []float32, fanIn int) {
	if fanIn < 1 {
		fanIn = 1
	}
	std := math.Sqrt(1 / float64(fanIn))
	for i := range w {
		w[i] = float32(rng.NormFloat64() * std)
	}
}
