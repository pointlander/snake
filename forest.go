// Copyright 2026 The Snake Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"math"
	"math/rand/v2"
	"sort"
	"sync"
)

// node is one decision-tree node. feature < 0 marks a leaf.
type node struct {
	feature int16
	class   uint8
	thr     float32
	left    int32
	right   int32
}

type tree struct {
	nodes []node
}

func (t *tree) predict(at func(feature int) float32) uint8 {
	i := 0
	for t.nodes[i].feature >= 0 {
		if at(int(t.nodes[i].feature)) <= t.nodes[i].thr {
			i = int(t.nodes[i].left)
		} else {
			i = int(t.nodes[i].right)
		}
	}
	return t.nodes[i].class
}

// forest is a majority vote of classification trees.
type forest struct {
	trees  []tree
	nClass int
	votes  []int
}

func (f *forest) tally(at func(feature int) float32) {
	for i := range f.votes {
		f.votes[i] = 0
	}
	for i := range f.trees {
		f.votes[f.trees[i].predict(at)]++
	}
}

func (f *forest) predict(at func(feature int) float32) uint8 {
	f.tally(at)
	best, bestN := uint8(0), -1
	for c := 0; c < f.nClass; c++ {
		if f.votes[c] > bestN {
			best = uint8(c)
			bestN = f.votes[c]
		}
	}
	return best
}

// probs writes each class's share of the tree votes into dst.
func (f *forest) probs(at func(feature int) float32, dst []float32) {
	f.tally(at)
	var inv float32
	if len(f.trees) > 0 {
		inv = 1 / float32(len(f.trees))
	}
	for c := range dst {
		if c < len(f.votes) {
			dst[c] = float32(f.votes[c]) * inv
		} else {
			dst[c] = 0
		}
	}
}

type splitItem struct {
	v float32
	y uint8
}

type builder struct {
	pr          []float32
	stateAt     []int32
	ranks       []float32
	labels      []uint8
	window      int
	nClass      int
	maxDepth    int
	minLeaf     int
	mtry        int
	focus       int
	splitSample int
	rng         *rand.Rand
	nodes       []node
	items       []splitItem
	left        []int
	right       []int
	perm        []int
	chosen      []int
}

func (b *builder) value(example, feature int) float32 {
	idx := example - b.window + feature
	if b.stateAt != nil {
		id := b.stateAt[idx]
		if id < 0 {
			return 0
		}
		return b.ranks[id]
	}
	return b.pr[idx]
}

func (b *builder) majority(idx []int, lo, hi int) uint8 {
	for i := range b.left {
		b.left[i] = 0
	}
	for _, id := range idx[lo:hi] {
		b.left[b.labels[id]]++
	}
	best, bestN := 0, -1
	for c, n := range b.left {
		if n > bestN {
			best, bestN = c, n
		}
	}
	return uint8(best)
}

// bestSplit picks a Gini split. Candidate thresholds come from a subsample
// of the node, so every training row still reaches a leaf while split
// search stays bounded.
func (b *builder) bestSplit(idx []int, lo, hi int) (feat int, thr float32, ok bool) {
	n := hi - lo
	if n < 2*b.minLeaf {
		return 0, 0, false
	}
	sampleN := b.splitSample
	if sampleN > n {
		sampleN = n
	}
	if cap(b.items) < sampleN {
		b.items = make([]splitItem, sampleN)
	}
	items := b.items[:sampleN]

	bestScore := math.Inf(1)
	chosen := b.chooseFeatures()
	// Reject cuts that would peel off fewer rows than a leaf. On a subsample,
	// scale that minimum so a one-row fluke is not the selected threshold.
	minSide := b.minLeaf
	if sampleN < n {
		minSide = int(float64(sampleN) * float64(b.minLeaf) / float64(n))
		if minSide < 1 {
			minSide = 1
		}
	}
	for _, feature := range chosen {
		for i := 0; i < sampleN; i++ {
			id := idx[lo+b.rng.IntN(n)]
			items[i].v = b.value(id, feature)
			items[i].y = b.labels[id]
		}
		sort.Slice(items, func(a, c int) bool { return items[a].v < items[c].v })
		if items[0].v == items[len(items)-1].v {
			continue
		}
		for i := range b.right {
			b.right[i] = 0
		}
		for i := range items {
			b.right[items[i].y]++
		}
		for i := range b.left {
			b.left[i] = 0
		}
		nL, nR := 0, len(items)
		for i := 0; i < len(items)-1; i++ {
			y := items[i].y
			b.left[y]++
			b.right[y]--
			nL++
			nR--
			if nL < minSide || nR < minSide || items[i].v == items[i+1].v {
				continue
			}
			score := weightedGini(b.left, nL) + weightedGini(b.right, nR)
			if score < bestScore {
				bestScore = score
				feat = feature
				thr = items[i].v
				ok = true
			}
		}
	}
	return feat, thr, ok
}

// chooseFeatures returns the feature indexes considered at this split.
// The most recent scores are always included: that is where the Markov
// state sits. The rest of mtry is a random sample of the older scores.
func (b *builder) chooseFeatures() []int {
	mtry := b.mtry
	if mtry > b.window {
		mtry = b.window
	}
	if mtry < 1 {
		mtry = 1
	}
	focus := b.focus
	if focus > mtry {
		focus = mtry
	}
	if focus > b.window {
		focus = b.window
	}
	if focus < 0 {
		focus = 0
	}
	chosen := b.chosen[:0]
	for i := 0; i < focus; i++ {
		chosen = append(chosen, b.window-focus+i)
	}
	rest := mtry - focus
	pool := b.window - focus
	if rest > pool {
		rest = pool
	}
	for i := range pool {
		b.perm[i] = i
	}
	for i := 0; i < rest; i++ {
		j := i + b.rng.IntN(pool-i)
		b.perm[i], b.perm[j] = b.perm[j], b.perm[i]
		chosen = append(chosen, b.perm[i])
	}
	b.chosen = chosen
	return chosen
}

func weightedGini(counts []int, n int) float64 {
	if n == 0 {
		return 0
	}
	var sumSq float64
	for _, c := range counts {
		if c == 0 {
			continue
		}
		cf := float64(c)
		sumSq += cf * cf
	}
	nf := float64(n)
	// n*gini = n - sum(c^2)/n, and the constant n is the same for every split.
	return nf - sumSq/nf
}

func (b *builder) build(idx []int, lo, hi, depth int) int32 {
	id := int32(len(b.nodes))
	b.nodes = append(b.nodes, node{feature: -1})
	if hi-lo < 2*b.minLeaf || depth >= b.maxDepth || b.pure(idx, lo, hi) {
		b.nodes[id].class = b.majority(idx, lo, hi)
		return id
	}
	feat, thr, ok := b.bestSplit(idx, lo, hi)
	if !ok {
		b.nodes[id].class = b.majority(idx, lo, hi)
		return id
	}
	mid := b.partition(idx, lo, hi, feat, thr)
	if mid-lo < b.minLeaf || hi-mid < b.minLeaf {
		b.nodes[id].class = b.majority(idx, lo, hi)
		return id
	}
	left := b.build(idx, lo, mid, depth+1)
	right := b.build(idx, mid, hi, depth+1)
	b.nodes[id].feature = int16(feat)
	b.nodes[id].thr = thr
	b.nodes[id].left = left
	b.nodes[id].right = right
	return id
}

func (b *builder) pure(idx []int, lo, hi int) bool {
	y0 := b.labels[idx[lo]]
	for _, id := range idx[lo+1 : hi] {
		if b.labels[id] != y0 {
			return false
		}
	}
	return true
}

func (b *builder) partition(idx []int, lo, hi, feature int, thr float32) int {
	i, j := lo, hi
	for i < j {
		if b.value(idx[i], feature) <= thr {
			i++
			continue
		}
		j--
		idx[i], idx[j] = idx[j], idx[i]
	}
	return i
}

func trainTree(pr []float32, stateAt []int32, ranks []float32, labels []uint8, examples []int, nClass int, cfg config, rng *rand.Rand) tree {
	n := len(examples)
	idx := make([]int, n)
	for i := range idx {
		idx[i] = examples[rng.IntN(n)]
	}
	mtry := cfg.mtry
	if mtry <= 0 {
		mtry = int(math.Sqrt(float64(cfg.window)))
		if mtry < 1 {
			mtry = 1
		}
	}
	minLeaf := cfg.minLeaf
	if minLeaf < 1 {
		minLeaf = 1
	}
	b := &builder{
		pr:          pr,
		stateAt:     stateAt,
		ranks:       ranks,
		labels:      labels,
		window:      cfg.window,
		nClass:      nClass,
		maxDepth:    cfg.maxDepth,
		minLeaf:     minLeaf,
		mtry:        mtry,
		focus:       cfg.focus,
		splitSample: cfg.splitSample,
		rng:         rng,
		left:        make([]int, nClass),
		right:       make([]int, nClass),
		perm:        make([]int, cfg.window),
		chosen:      make([]int, 0, mtry),
	}
	for i := range b.perm {
		b.perm[i] = i
	}
	if b.splitSample < 2 {
		b.splitSample = 2
	}
	b.build(idx, 0, n, 0)
	return tree{nodes: b.nodes}
}

func trainForest(pr []float32, stateAt []int32, ranks []float32, labels []uint8, examples []int, nClass int, cfg config, logf func(string, ...any)) *forest {
	trees := make([]tree, cfg.trees)
	var wg sync.WaitGroup
	for i := 0; i < cfg.trees; i++ {
		wg.Add(1)
		go func(i int) {
			defer wg.Done()
			s := cfg.seed + uint64(i)*0x9e3779b97f4a7c15
			rng := rand.New(rand.NewPCG(s, s^0x6a09e667f3bcc909))
			trees[i] = trainTree(pr, stateAt, ranks, labels, examples, nClass, cfg, rng)
			if logf != nil {
				logf("tree %d/%d  nodes %d", i+1, cfg.trees, len(trees[i].nodes))
			}
		}(i)
	}
	wg.Wait()
	return &forest{trees: trees, nClass: nClass, votes: make([]int, nClass)}
}
