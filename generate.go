// Copyright 2026 The Snake Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

// generate continues prompt by n runes. Each rune is the network's
// prediction from the PageRank of the previous window of states.
func (m *model) generate(prompt []rune, n int) []rune {
	if n < 0 {
		n = 0
	}
	out := make([]rune, len(prompt), len(prompt)+n)
	copy(out, prompt)
	if m == nil || m.net == nil || m.graph == nil || len(m.classes) == 0 || n == 0 {
		return out
	}
	window := m.net.window
	ring := make([]float32, window)
	for feature := 0; feature < window; feature++ {
		ring[feature] = m.rankAt(out, len(out)-window+feature)
	}
	head := 0
	for i := 0; i < n; i++ {
		pred := m.net.predict(func(feature int) float32 {
			return ring[(head+feature)%window]
		})
		out = append(out, m.classes[pred])
		ring[head] = m.rankAt(out, len(out)-1)
		head++
		if head == window {
			head = 0
		}
	}
	return out
}

// rankAt is the frozen PageRank of the bigram ending at pos.
// A missing position, or a bigram the training graph never saw, is zero.
func (m *model) rankAt(text []rune, pos int) float32 {
	if m == nil || m.graph == nil || pos < 1 || pos >= len(text) {
		return 0
	}
	id, ok := m.graph.idOf[pairKey(text[pos-1], text[pos])]
	if !ok {
		return 0
	}
	return m.ranks[id]
}
