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

func TestNetworkSeparatesThreshold(t *testing.T) {
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
		window: window,
		hidden: 8,
		epochs: 40,
		batch:  32,
		lr:     0.1,
		seed:   9,
	}
	f := trainNetwork(pr, nil, nil, labels, examples, 2, cfg, nil)
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

func TestGenerateContinuesPrompt(t *testing.T) {
	text := []rune(strings.Repeat("aaab", 500))
	cut := int(float64(len(text)) * 0.9)
	cfg := config{
		window:       4,
		damping:      0.85,
		walksPerNode: 20,
		hidden:       16,
		epochs:       40,
		batch:        32,
		lr:           0.1,
		seed:         1,
	}
	_, trained := experiment(text, cut, cfg, nil)
	if trained == nil {
		t.Fatal("missing model")
	}
	prompt := []rune("aaabaaab")
	const n = 12
	rng := rand.New(rand.NewPCG(1, 2))
	out := trained.generate(prompt, n, rng)
	if len(out) != len(prompt)+n {
		t.Fatalf("length %d", len(out))
	}
	if string(out[:len(prompt)]) != string(prompt) {
		t.Fatalf("prompt dropped: %q", string(out))
	}
	for _, r := range out[len(prompt):] {
		if r != 'a' && r != 'b' {
			t.Fatalf("generated %q", string(out))
		}
	}
	// No spaces exist in this alphabet, so every rollout ties and the
	// search follows the network along the cycle.
	if string(out[len(prompt):]) != "aaabaaabaaab" {
		t.Fatalf("continuation %q", string(out))
	}
	// A prompt that never occurs still yields training runes.
	other := trained.generate([]rune("zz"), 4, rng)
	if string(other[:2]) != "zz" || len(other) != 6 {
		t.Fatalf("unseen prompt: %q", string(other))
	}
	for _, r := range other[2:] {
		if r != 'a' && r != 'b' {
			t.Fatalf("unseen continuation %q", string(other))
		}
	}
}

func TestMCTSAvoidsSpaceRun(t *testing.T) {
	// Context is ignored. Space has most of the mass, so greedy decoding
	// is a run of spaces. Search should sample the other class and refuse
	// to continue the run.
	const window = 4
	net := &network{
		window: window,
		hidden: 1,
		nClass: 2,
		invStd: 1,
		w1:     make([]float32, window),
		b1:     make([]float32, 1),
		w2:     make([]float32, 2),
		b2:     []float32{2, 0},
		x:      make([]float32, window),
		h:      make([]float32, 1),
		z:      make([]float32, 2),
	}
	trained := &model{
		graph:   newMarkov(),
		net:     net,
		classes: []rune{' ', 'a'},
	}
	rng := rand.New(rand.NewPCG(1, 2))
	const n = 24
	greedy := make([]rune, 0, n)
	for i := 0; i < n; i++ {
		c := net.predict(func(int) float32 { return 0 })
		greedy = append(greedy, trained.classes[c])
	}
	if !strings.Contains(string(greedy), "    ") {
		t.Fatalf("greedy did not collapse: %q", string(greedy))
	}
	out := trained.generate([]rune("a"), n, rng)
	if string(out[:1]) != "a" || len(out) != n+1 {
		t.Fatalf("prompt: %q", string(out))
	}
	body := string(out[1:])
	if strings.Contains(body, "  ") || strings.Count(body, "a") < n-2 {
		t.Fatalf("continuation %q", body)
	}
	// A prompt that already ends on a space must not extend that run.
	spaced := trained.generate([]rune("a "), 8, rng)
	if len(spaced) != 10 || spaced[2] == ' ' || strings.Contains(string(spaced[2:]), "  ") {
		t.Fatalf("extended a space run: %q", string(spaced))
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
		hidden:       16,
		epochs:       40,
		batch:        32,
		lr:           0.1,
		seed:         1,
	}
	m, _ := experiment(text, cut, cfg, nil)
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
