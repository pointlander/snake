// Copyright 2026 The Snake Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"math"
	"math/rand/v2"
	"strings"
	"testing"
)

func TestPageRankMatchesPowerIteration(t *testing.T) {
	m := newMarkov()
	ab, _ := m.observe('a', 'b')
	bc, _ := m.observe('b', 'c')
	ca, _ := m.observe('c', 'a')
	cd, _ := m.observe('c', 'd')
	// ab -> bc (weight 1), ab -> cd (weight 3)
	// bc -> ca (weight 2)
	// ca -> ab (weight 4)
	// cd -> bc (weight 1), and a self-loop so nothing is dangling
	m.link(ab, bc)
	for i := 0; i < 3; i++ {
		m.link(ab, cd)
	}
	for i := 0; i < 2; i++ {
		m.link(bc, ca)
	}
	for i := 0; i < 4; i++ {
		m.link(ca, ab)
	}
	m.link(cd, bc)
	m.link(cd, cd)

	const damping = 0.85
	want := powerPageRank(m, damping)
	rng := rand.New(rand.NewPCG(7, 11))
	got := m.pageRank(rng, damping, 400)

	var sum float64
	maxDiff := 0.0
	for i := range want {
		sum += float64(got[i])
		d := math.Abs(float64(got[i]) - want[i])
		if d > maxDiff {
			maxDiff = d
		}
	}
	if math.Abs(sum-1) > 1e-4 {
		t.Fatalf("ranks sum to %f", sum)
	}
	if maxDiff > 0.03 {
		t.Fatalf("max |mc-power| = %f\nmc %v\npower %v", maxDiff, got, want)
	}
}

func TestPageRankCycleIsUniform(t *testing.T) {
	m := newMarkov()
	a, _ := m.observe('a', 'b')
	b, _ := m.observe('b', 'c')
	c, _ := m.observe('c', 'a')
	m.link(a, b)
	m.link(b, c)
	m.link(c, a)
	rng := rand.New(rand.NewPCG(1, 2))
	got := m.pageRank(rng, 0.85, 300)
	for i, r := range got {
		if math.Abs(float64(r)-1.0/3.0) > 0.03 {
			t.Fatalf("node %d rank %f, want ~1/3 (%v)", i, r, got)
		}
	}
}

func powerPageRank(m *markov, damping float64) []float64 {
	n := len(m.out)
	p := make([]float64, n)
	next := make([]float64, n)
	u := 1 / float64(n)
	for i := range p {
		p[i] = u
	}
	for iter := 0; iter < 300; iter++ {
		for i := range next {
			next[i] = 0
		}
		dangling := 0.0
		for i, edges := range m.out {
			if m.sum[i] == 0 {
				dangling += p[i]
				continue
			}
			scale := damping * p[i] / float64(m.sum[i])
			for _, e := range edges {
				next[e.to] += scale * float64(e.w)
			}
		}
		share := (1-damping)/float64(n) + damping*dangling/float64(n)
		for i := range next {
			next[i] += share
		}
		p, next = next, p
	}
	return p
}

func TestForestSeparatesThreshold(t *testing.T) {
	// Feature 0 is the label. The other features are noise.
	const (
		n      = 400
		window = 4
	)
	pr := make([]float32, n)
	labels := make([]uint8, n)
	examples := make([]int, 0, n-window-1)
	rng := rand.New(rand.NewPCG(3, 5))
	for t := 1; t < n; t++ {
		pr[t] = float32(rng.Float64())
	}
	for t := window + 1; t < n; t++ {
		if pr[t-window] >= 0.5 {
			labels[t] = 1
		}
		examples = append(examples, t)
	}
	cfg := config{
		window:      window,
		trees:       8,
		maxDepth:    6,
		minLeaf:     1,
		mtry:        window,
		splitSample: 200,
		seed:        9,
	}
	f := trainForest(pr, nil, nil, labels, examples, 2, cfg, nil)
	correct := 0
	for _, t := range examples {
		origin := t - window
		pred := f.predict(func(feature int) float32 { return pr[origin+feature] })
		if pred == labels[t] {
			correct++
		}
	}
	acc := float64(correct) / float64(len(examples))
	if acc < 0.95 {
		t.Fatalf("accuracy %f", acc)
	}
}

func TestRepeatingTextBeatsMajority(t *testing.T) {
	// "aaab" makes an asymmetric 4-cycle of bigrams, so PageRank values
	// identify the state and a short window identifies the next rune.
	text := []rune(strings.Repeat("aaab", 500))
	cut := int(float64(len(text)) * 0.9)
	cfg := config{
		window:       4,
		damping:      0.85,
		walksPerNode: 20,
		trees:        8,
		maxDepth:     8,
		minLeaf:      1,
		mtry:         4,
		splitSample:  256,
		seed:         1,
	}
	m := experiment(text, cut, cfg, nil)
	if m.Nodes < 3 {
		t.Fatalf("nodes %d", m.Nodes)
	}
	if m.Recomputes != m.Nodes+1 {
		t.Fatalf("recomputes %d nodes %d", m.Recomputes, m.Nodes)
	}
	if m.ValN != len(text)-cut {
		t.Fatalf("val %d", m.ValN)
	}
	acc := float64(m.ValCorrect) / float64(m.ValN)
	base := float64(m.Baseline) / float64(m.ValN)
	if acc < 0.95 {
		t.Fatalf("validation accuracy %f baseline %f metrics %+v", acc, base, m)
	}
	if acc <= base {
		t.Fatalf("accuracy %f did not beat baseline %f", acc, base)
	}
}
