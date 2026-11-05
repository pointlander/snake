// Copyright 2026 The Snake Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"math"
	"math/rand/v2"
)

const (
	// Distinct next runes drawn from the predictor at the root.
	mctsCandidates = 8
	// Predictor rollouts used to score every candidate before extra search.
	mctsRollouts = 4
	// Additional rollouts given to whichever child PUCT prefers.
	mctsExtra = 16
	// Sampled continuation length. Long enough to see a space run.
	mctsHorizon = 8
	// Prior weight for the extra rollouts.
	mctsCPuct = 1
)

// generate continues prompt by n runes. Monte Carlo tree search samples
// the predictor: each candidate next rune is scored by continuations drawn
// from the predictor, and a run of spaces scores worse than a single space.
func (m *model) generate(prompt []rune, n int, rng *rand.Rand) []rune {
	if n < 0 {
		n = 0
	}
	out := make([]rune, len(prompt), len(prompt)+n)
	copy(out, prompt)
	if m == nil || m.pred == nil || m.graph == nil || len(m.classes) == 0 || m.window < 1 || n == 0 {
		return out
	}
	st := m.newState(prompt)
	for i := 0; i < n; i++ {
		next := m.search(st, rng)
		out = append(out, next)
		st.push(m, next)
	}
	return out
}

// search picks the next rune. Candidates are sampled from the predictor
// without replacement. A rollout is another sample from the predictor, not
// a move the tree chooses: a tree-chosen continuation would spend the
// later plies canceling a space run, and the run would no longer count
// against the rune that started it.
func (m *model) search(st searchState, rng *rand.Rand) rune {
	root := &mctsNode{probs: m.policy(st)}
	for len(root.children) < mctsCandidates {
		c, ok := sampleUnused(root.probs, root.children, rng)
		if !ok {
			break
		}
		root.children = append(root.children, &mctsNode{class: c, prior: root.probs[c]})
	}
	if len(root.children) == 0 {
		return m.classes[samplePrior(root.probs, rng)]
	}
	scratch := make([]float32, len(m.classes))
	roll := func(ch *mctsNode) {
		cur := st.clone()
		runes := make([]rune, 1, mctsHorizon)
		runes[0] = m.classes[ch.class]
		cur.push(m, runes[0])
		for depth := 1; depth < mctsHorizon; depth++ {
			m.fillProbs(cur, scratch)
			r := m.classes[samplePrior(scratch, rng)]
			runes = append(runes, r)
			cur.push(m, r)
		}
		root.visits++
		ch.visits++
		ch.value += collapseScore(st.run, runes)
	}
	for i := 0; i < mctsRollouts; i++ {
		for _, ch := range root.children {
			roll(ch)
		}
	}
	for i := 0; i < mctsExtra; i++ {
		roll(root.puctChild())
	}
	return m.classes[root.best().class]
}

type searchState struct {
	ring []float32
	head int
	prev rune
	ok   bool
	run  int // consecutive spaces ending at prev
}

func (m *model) newState(prompt []rune) searchState {
	window := m.window
	st := searchState{ring: make([]float32, window)}
	for feature := 0; feature < window; feature++ {
		st.ring[feature] = m.rankAt(prompt, len(prompt)-window+feature)
	}
	if len(prompt) > 0 {
		st.prev = prompt[len(prompt)-1]
		st.ok = true
		for i := len(prompt) - 1; i >= 0 && prompt[i] == ' '; i-- {
			st.run++
		}
	}
	return st
}

func (s searchState) clone() searchState {
	s.ring = append([]float32(nil), s.ring...)
	return s
}

func (s *searchState) push(m *model, r rune) {
	var rank float32
	if s.ok {
		rank = m.pairRank(s.prev, r)
	}
	s.ring[s.head] = rank
	s.head++
	if s.head == len(s.ring) {
		s.head = 0
	}
	s.prev = r
	s.ok = true
	if r == ' ' {
		s.run++
	} else {
		s.run = 0
	}
}

func (m *model) policy(st searchState) []float32 {
	p := make([]float32, len(m.classes))
	m.fillProbs(st, p)
	return p
}

func (m *model) fillProbs(st searchState, dst []float32) {
	window := len(st.ring)
	m.pred.probs(func(feature int) float32 {
		return st.ring[(st.head+feature)%window]
	}, dst)
}

// collapseScore rewards letters and a single space, and penalizes a run.
// run is the number of spaces already emitted before rs.
func collapseScore(run int, rs []rune) float64 {
	if len(rs) == 0 {
		return 0
	}
	var score float64
	for _, r := range rs {
		if r == ' ' {
			run++
			if run >= 2 {
				score -= 1
			} else {
				score += 1
			}
		} else {
			run = 0
			score += 1
		}
	}
	return score / float64(len(rs))
}

type mctsNode struct {
	class    int
	prior    float32
	visits   int
	value    float64
	probs    []float32
	children []*mctsNode
}

func (nd *mctsNode) puctChild() *mctsNode {
	best := nd.children[0]
	bestU := puct(nd, best)
	for _, ch := range nd.children[1:] {
		if u := puct(nd, ch); u > bestU {
			best = ch
			bestU = u
		}
	}
	return best
}

func puct(parent, ch *mctsNode) float64 {
	q := 0.0
	if ch.visits > 0 {
		q = ch.value / float64(ch.visits)
	}
	return q + mctsCPuct*float64(ch.prior)*math.Sqrt(float64(parent.visits))/(1+float64(ch.visits))
}

// best is the child with the highest average rollout score.
// Equal scores follow the network prior.
func (nd *mctsNode) best() *mctsNode {
	best := nd.children[0]
	bestQ := best.mean()
	for _, ch := range nd.children[1:] {
		q := ch.mean()
		switch {
		case q > bestQ:
			best, bestQ = ch, q
		case q == bestQ && ch.prior > best.prior:
			best, bestQ = ch, q
		}
	}
	return best
}

func (nd *mctsNode) mean() float64 {
	if nd.visits == 0 {
		return math.Inf(-1)
	}
	return nd.value / float64(nd.visits)
}

// samplePrior draws a class from a softmax distribution.
func samplePrior(p []float32, rng *rand.Rand) int {
	r := rng.Float64()
	var acc float64
	last := len(p) - 1
	for i, v := range p {
		acc += float64(v)
		if r < acc {
			return i
		}
	}
	return last
}

// sampleUnused draws from p, skipping classes that are already children.
func sampleUnused(p []float32, children []*mctsNode, rng *rand.Rand) (int, bool) {
	used := make([]bool, len(p))
	for _, ch := range children {
		if ch.class >= 0 && ch.class < len(used) {
			used[ch.class] = true
		}
	}
	var sum float64
	for i, v := range p {
		if !used[i] {
			sum += float64(v)
		}
	}
	if sum <= 0 {
		return 0, false
	}
	r := rng.Float64() * sum
	var acc float64
	last := 0
	found := false
	for i, v := range p {
		if used[i] {
			continue
		}
		last = i
		found = true
		acc += float64(v)
		if r <= acc {
			return i, true
		}
	}
	return last, found
}

// rankAt is the frozen PageRank of the bigram ending at pos.
// A missing position, or a bigram the training graph never saw, is zero.
func (m *model) rankAt(text []rune, pos int) float32 {
	if m == nil || m.graph == nil || pos < 1 || pos >= len(text) {
		return 0
	}
	return m.pairRank(text[pos-1], text[pos])
}

func (m *model) pairRank(a, b rune) float32 {
	if m == nil || m.graph == nil {
		return 0
	}
	id, ok := m.graph.idOf[pairKey(a, b)]
	if !ok {
		return 0
	}
	return m.ranks[id]
}
