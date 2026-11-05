// Copyright 2026 The Snake Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Command snake builds a second-order Markov graph of a text, estimates
// PageRank with the Monte Carlo complete-path method as states appear, and
// trains a two-layer feedforward network to predict the next rune from
// the normalized PageRank of the previous 256 states. The network sees
// the first 90% of the text. The remaining 10% is written to a validation
// file and scored once, with the graph and the network frozen.
//
// With -generate, the network continues -prompt. Each new rune is its
// prediction from the PageRank of the previous 256 states.
package main

import (
	"flag"
	"fmt"
	"os"
	"time"
)

func main() {
	textPath := flag.String("text", "100.txt.utf-8", "UTF-8 text to model")
	valPath := flag.String("validation", "validation.txt", "where to write the held-out text")
	trainFrac := flag.Float64("train", 0.9, "fraction of the text used to train")
	generate := flag.Bool("generate", false, "continue -prompt with the trained network")
	prompt := flag.String("prompt", "To be or not to be", "text to continue in -generate mode")
	nGen := flag.Int("n", 256, "runes to generate after the prompt")
	flag.Parse()

	data, err := os.ReadFile(*textPath)
	if err != nil {
		fmt.Fprintf(os.Stderr, "read %s: %v\n", *textPath, err)
		os.Exit(1)
	}
	text := []rune(string(data))
	if len(text) == 0 {
		fmt.Fprintf(os.Stderr, "%s is empty\n", *textPath)
		os.Exit(1)
	}
	cut := int(float64(len(text)) * *trainFrac)
	if cut < 0 {
		cut = 0
	}
	if cut > len(text) {
		cut = len(text)
	}
	heldOut := []byte(string(text[cut:]))
	if err := os.WriteFile(*valPath, heldOut, 0o644); err != nil {
		fmt.Fprintf(os.Stderr, "write %s: %v\n", *valPath, err)
		os.Exit(1)
	}
	fmt.Printf("wrote %s  %d bytes  %d runes\n", *valPath, len(heldOut), len(text)-cut)

	cfg := config{
		window:       256,
		damping:      0.85,
		walksPerNode: 8,
		hidden:       64,
		epochs:       2,
		batch:        128,
		lr:           0.05,
		seed:         1,
	}
	logf := func(format string, args ...any) {
		fmt.Printf("%s  %s\n", time.Now().Format("15:04:05"), fmt.Sprintf(format, args...))
	}
	m, trained := experiment(text, cut, cfg, logf)
	fmt.Printf("runes %d  train %d  validation %d\n", m.Runes, m.Cut, m.ValN)
	fmt.Printf("states %d  edges %d  pagerank recomputes %d  classes %d\n", m.Nodes, m.Edges, m.Recomputes, m.Classes)
	fmt.Printf("train accuracy %.4f  (%d/%d)\n", ratio(m.TrainCorrect, m.TrainN), m.TrainCorrect, m.TrainN)
	fmt.Printf("validation accuracy %.4f  (%d/%d)\n", ratio(m.ValCorrect, m.ValN), m.ValCorrect, m.ValN)
	fmt.Printf("validation majority baseline %.4f  (%d/%d)\n", ratio(m.Baseline, m.ValN), m.Baseline, m.ValN)
	fmt.Printf("validation markov ceiling %.4f  (%d/%d)\n", ratio(m.Markov, m.ValN), m.Markov, m.ValN)
	fmt.Printf("elapsed %s\n", m.Elapsed.Round(time.Millisecond))
	if *generate {
		if trained == nil {
			fmt.Fprintf(os.Stderr, "generate: model was not trained\n")
			os.Exit(1)
		}
		out := trained.generate([]rune(*prompt), *nGen)
		fmt.Printf("\n%s\n", string(out))
	}
}

func ratio(n, d int) float64 {
	if d == 0 {
		return 0
	}
	return float64(n) / float64(d)
}
