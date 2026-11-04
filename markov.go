// Copyright 2026 The Snake Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import "math/rand/v2"

// edge is a weighted directed link in the second-order Markov graph.
type edge struct {
	to int
	w  int
}

// markov is a second-order Markov chain over runes.
// A node is an ordered pair of runes. An edge (ab -> bc) is the
// transition observed when the text contains the trigram abc.
type markov struct {
	out  [][]edge
	sum  []int
	pair [][2]rune
	idOf map[uint64]int
}

func newMarkov() *markov {
	return &markov{idOf: make(map[uint64]int)}
}

func pairKey(a, b rune) uint64 {
	return uint64(uint32(a))<<32 | uint64(uint32(b))
}

// observe returns the node for the bigram (a, b), adding it the first time.
func (m *markov) observe(a, b rune) (id int, added bool) {
	k := pairKey(a, b)
	if id, ok := m.idOf[k]; ok {
		return id, false
	}
	id = len(m.out)
	m.idOf[k] = id
	m.out = append(m.out, nil)
	m.sum = append(m.sum, 0)
	m.pair = append(m.pair, [2]rune{a, b})
	return id, true
}

// link records one observation of the transition from -> to.
func (m *markov) link(from, to int) {
	edges := m.out[from]
	for i := range edges {
		if edges[i].to == to {
			m.out[from][i].w++
			m.sum[from]++
			return
		}
	}
	m.out[from] = append(edges, edge{to: to, w: 1})
	m.sum[from]++
}

func (m *markov) edges() int {
	n := 0
	for _, e := range m.out {
		n += len(e)
	}
	return n
}

// pageRank estimates weighted PageRank by the Monte Carlo complete-path
// method. Each node starts the same number of walks. A walk follows a
// weight-proportional edge with probability damping and stops with
// probability 1-damping. A node with no out-edges jumps to a uniform
// random node on a continuation step, matching the usual dangling-node
// correction. Visit counts are normalized to sum to one.
func (m *markov) pageRank(rng *rand.Rand, damping float64, walksPerNode int) []float32 {
	n := len(m.out)
	ranks := make([]float32, n)
	if n == 0 {
		return ranks
	}
	if walksPerNode < 1 {
		walksPerNode = 1
	}
	visits := make([]uint64, n)
	var total uint64
	walks := n * walksPerNode
	for w := 0; w < walks; w++ {
		node := w % n
		visits[node]++
		total++
		for rng.Float64() < damping {
			s := m.sum[node]
			if s == 0 {
				node = rng.IntN(n)
			} else {
				r := rng.IntN(s)
				acc := 0
				next := m.out[node][len(m.out[node])-1].to
				for _, e := range m.out[node] {
					acc += e.w
					if r < acc {
						next = e.to
						break
					}
				}
				node = next
			}
			visits[node]++
			total++
		}
	}
	inv := 1 / float64(total)
	for i, v := range visits {
		ranks[i] = float32(float64(v) * inv)
	}
	return ranks
}
