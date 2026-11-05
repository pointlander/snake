// Copyright 2026 The Snake Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"math/rand/v2"
	"sort"
	"time"
)

// config is the PageRank, network, and forest setup.
// model is "net" or "forest". An empty model selects the network.
type config struct {
	window       int
	damping      float64
	walksPerNode int
	model        string
	hidden       int
	epochs       int
	batch        int
	lr           float64
	trees        int
	maxDepth     int
	minLeaf      int
	mtry         int
	focus        int // most recent window positions considered at every split
	splitSample  int
	seed         uint64
}

// metrics summarizes one train/validation run.
type metrics struct {
	Runes        int
	Cut          int
	Nodes        int
	Edges        int
	Recomputes   int
	Classes      int
	TrainN       int
	TrainCorrect int
	ValN         int
	ValCorrect   int
	Baseline     int
	Markov       int
	Elapsed      time.Duration
}

type rankedState struct {
	pair [2]rune
	rank float32
}

// predictor maps a window of PageRanks to the next rune.
type predictor interface {
	predict(at func(feature int) float32) uint8
	probs(at func(feature int) float32, dst []float32)
}

// model is the frozen training graph, its PageRank, and the predictor.
type model struct {
	graph   *markov
	ranks   []float32
	classes []rune
	window  int
	pred    predictor
}

// experiment streams the training prefix once. Nodes are second-order
// Markov states (consecutive rune pairs). Monte Carlo PageRank is
// recomputed whenever a state is added, then once more on the finished
// training graph. The predictor — a network or a random forest — scores
// the rune at position t from the final normalized PageRank of the previous
// window of states. The graph and the predictor stay frozen on the held-out
// suffix.
func experiment(text []rune, cut int, cfg config, logf func(string, ...any)) (metrics, *model) {
	started := time.Now()
	var out metrics
	out.Runes = len(text)
	if cut < 0 {
		cut = 0
	}
	if cut > len(text) {
		cut = len(text)
	}
	out.Cut = cut
	if cfg.window < 1 || len(text) < cfg.window+2 || cut < cfg.window+2 {
		out.Elapsed = time.Since(started)
		return out, nil
	}

	m := newMarkov()
	labels := make([]uint8, cut)
	classOf := make(map[rune]uint8)
	classList := make([]rune, 0, 128)
	stateAt := make([]int32, cut)
	for i := range stateAt {
		stateAt[i] = -1
	}
	var ranks []float32
	rng := rand.New(rand.NewPCG(cfg.seed, cfg.seed^0xd1b54a32d192ed03))

	var prev rune
	lastID := -1
	for t, r := range text[:cut] {
		if c, ok := classOf[r]; ok {
			labels[t] = c
		} else if len(classList) == 256 {
			panic("more than 256 distinct training runes")
		} else {
			c = uint8(len(classList))
			classOf[r] = c
			classList = append(classList, r)
			labels[t] = c
		}
		if t == 0 {
			prev = r
			continue
		}
		id, added := m.observe(prev, r)
		if lastID >= 0 {
			m.link(lastID, id)
		}
		if added {
			ranks = m.pageRank(rng, cfg.damping, cfg.walksPerNode)
			out.Recomputes++
			if logf != nil && (out.Recomputes%400 == 0 || out.Recomputes == 1) {
				logf("pagerank recompute %d  nodes %d  edges %d  at rune %d", out.Recomputes, len(m.out), m.edges(), t)
			}
		}
		stateAt[t] = int32(id)
		lastID = id
		prev = r
	}
	ranks = m.pageRank(rng, cfg.damping, cfg.walksPerNode)
	out.Recomputes++
	out.Nodes = len(m.out)
	out.Edges = m.edges()
	out.Classes = len(classList)
	if logf != nil {
		logf("graph  nodes %d  edges %d  recomputes %d  classes %d", out.Nodes, out.Edges, out.Recomputes, out.Classes)
		top := topStates(m.pair, ranks, 8)
		for _, s := range top {
			logf("rank %.6f  state %q", s.rank, string(s.pair[:]))
		}
	}

	first := cfg.window + 1
	examples := make([]int, 0, cut-first)
	for t := first; t < cut; t++ {
		examples = append(examples, t)
	}
	out.TrainN = len(examples)
	if out.TrainN == 0 || out.Classes == 0 {
		out.Elapsed = time.Since(started)
		return out, nil
	}

	var learned predictor
	switch cfg.model {
	case "", "net":
		learned = trainNetwork(nil, stateAt, ranks, labels, examples, len(classList), cfg, logf)
	case "forest":
		if logf != nil {
			logf("training forest on %d examples  window %d  trees %d  mtry %d  focus %d", out.TrainN, cfg.window, cfg.trees, cfg.mtry, cfg.focus)
		}
		learned = trainForest(nil, stateAt, ranks, labels, examples, len(classList), cfg, logf)
	default:
		panic("unknown model " + cfg.model)
	}

	trainCorrect := 0
	for _, t := range examples {
		origin := t - cfg.window
		pred := learned.predict(func(feature int) float32 {
			return rankOf(stateAt, ranks, origin+feature)
		})
		if pred == labels[t] {
			trainCorrect++
		}
	}
	out.TrainCorrect = trainCorrect

	freq := make([]int, len(classList))
	for _, t := range examples {
		freq[labels[t]]++
	}
	majority := 0
	for c, n := range freq {
		if n > freq[majority] {
			majority = c
		}
	}
	majorityClass := uint8(majority)

	// Continuation counts of the training chain: previous state -> next rune.
	// This is the second-order Markov ceiling for the same prediction.
	nClass := len(classList)
	follow := make([]int, len(m.out)*nClass)
	for t := 2; t < cut; t++ {
		sid := stateAt[t-1]
		follow[int(sid)*nClass+int(labels[t])]++
	}

	// The held-out walk reads the frozen training PageRank. A state that
	// never occurred in training contributes a zero.
	ring := make([]float32, cfg.window)
	for i := 0; i < cfg.window; i++ {
		ring[i] = rankOf(stateAt, ranks, cut-cfg.window+i)
	}
	head := 0 // index of the oldest sample in ring
	valCorrect := 0
	baseline := 0
	markovHit := 0
	valN := 0
	for t := cut; t < len(text); t++ {
		pred := learned.predict(func(feature int) float32 {
			return ring[(head+feature)%cfg.window]
		})
		valN++
		var class uint8
		known := false
		if c, ok := classOf[text[t]]; ok {
			class = c
			known = true
			if pred == c {
				valCorrect++
			}
			if c == majorityClass {
				baseline++
			}
		}
		if known && t >= 2 {
			if sid, ok := m.idOf[pairKey(text[t-2], text[t-1])]; ok {
				row := follow[sid*nClass : (sid+1)*nClass]
				best, bestN := 0, -1
				for cls, n := range row {
					if n > bestN {
						best, bestN = cls, n
					}
				}
				if uint8(best) == class {
					markovHit++
				}
			}
		}
		var p float32
		if id, ok := m.idOf[pairKey(text[t-1], text[t])]; ok {
			p = ranks[id]
		}
		ring[head] = p
		head++
		if head == cfg.window {
			head = 0
		}
	}
	out.ValN = valN
	out.ValCorrect = valCorrect
	out.Baseline = baseline
	out.Markov = markovHit
	out.Elapsed = time.Since(started)
	return out, &model{graph: m, ranks: ranks, classes: classList, window: cfg.window, pred: learned}
}

func rankOf(stateAt []int32, ranks []float32, pos int) float32 {
	if pos < 0 || pos >= len(stateAt) {
		return 0
	}
	id := stateAt[pos]
	if id < 0 {
		return 0
	}
	return ranks[id]
}

func topStates(pairs [][2]rune, ranks []float32, k int) []rankedState {
	all := make([]rankedState, len(ranks))
	for i, r := range ranks {
		all[i] = rankedState{pair: pairs[i], rank: r}
	}
	sort.Slice(all, func(i, j int) bool { return all[i].rank > all[j].rank })
	if k > len(all) {
		k = len(all)
	}
	return all[:k]
}
