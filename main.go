package main

import (
	"bufio"
	"fmt"
	"os"
	"sort"
	"strings"
)

type pattern struct {
	posFreq   map[int]map[string]int
	labelFreq map[string]int
	base      string
}

func readLines(path string) ([]string, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	var lines []string
	sc := bufio.NewScanner(f)
	for sc.Scan() {
		line := strings.TrimSpace(sc.Text())
		if line == "" || strings.HasPrefix(line, "#") {
			continue
		}
		lines = append(lines, line)
	}
	if err := sc.Err(); err != nil {
		return nil, err
	}
	return unique(lines), nil
}

func unique(in []string) []string {
	seen := make(map[string]struct{}, len(in))
	out := make([]string, 0, len(in))
	for _, v := range in {
		if _, ok := seen[v]; ok {
			continue
		}
		seen[v] = struct{}{}
		out = append(out, v)
	}
	return out
}

func guessBase(hosts []string) string {
	if len(hosts) == 0 {
		return ""
	}
	parts := strings.Split(hosts[0], ".")
	if len(parts) < 2 {
		return hosts[0]
	}
	return strings.Join(parts[len(parts)-2:], ".")
}

func buildPattern(hosts []string) *pattern {
	base := guessBase(hosts)
	p := &pattern{
		posFreq:   make(map[int]map[string]int),
		labelFreq: make(map[string]int),
		base:      base,
	}
	for _, h := range hosts {
		if !strings.HasSuffix(h, base) {
			continue
		}
		sub := strings.TrimSuffix(h, base)
		sub = strings.TrimSuffix(sub, ".")
		if sub == "" {
			continue
		}
		labels := strings.Split(sub, ".")
		for i, lbl := range labels {
			if lbl == "" {
				continue
			}
			if _, ok := p.posFreq[i]; !ok {
				p.posFreq[i] = make(map[string]int)
			}
			p.posFreq[i][lbl]++
			p.labelFreq[lbl]++
		}
	}
	return p
}

type kv struct {
	label string
	freq  int
}

func (p *pattern) topLabels(pos, limit int) []string {
	freqs, ok := p.posFreq[pos]
	if !ok {
		return nil
	}
	tmp := make([]kv, 0, len(freqs))
	for l, f := range freqs {
		tmp = append(tmp, kv{label: l, freq: f})
	}
	sort.Slice(tmp, func(i, j int) bool { return tmp[i].freq > tmp[j].freq })
	if limit > len(tmp) {
		limit = len(tmp)
	}
	out := make([]string, 0, limit)
	for i := 0; i < limit; i++ {
		out = append(out, tmp[i].label)
	}
	return out
}

func generatePermutations(p *pattern, hosts []string, topN int) []string {
	candidates := make(map[string]struct{})
	for _, h := range hosts {
		if !strings.HasSuffix(h, p.base) {
			continue
		}
		sub := strings.TrimSuffix(h, p.base)
		sub = strings.TrimSuffix(sub, ".")
		if sub == "" {
			continue
		}
		labels := strings.Split(sub, ".")
		for i := 0; i < len(labels); i++ {
			originalLabel := labels[i]
			top := p.topLabels(i, topN)
			for _, newLabel := range top {
				if newLabel == originalLabel {
					continue
				}
				newLabels := make([]string, len(labels))
				copy(newLabels, labels)
				newLabels[i] = newLabel
				host := strings.Join(newLabels, ".") + "." + p.base
				candidates[host] = struct{}{}
			}
		}
	}
	out := make([]string, 0, len(candidates))
	for h := range candidates {
		out = append(out, h)
	}
	sort.Strings(out)
	return out
}

func main() {
	if len(os.Args) < 2 {
		fmt.Fprintln(os.Stderr, "usage: shaper input.txt > output.txt")
		os.Exit(1)
	}
	in := os.Args[1]
	hosts, err := readLines(in)
	if err != nil || len(hosts) == 0 {
		fmt.Fprintln(os.Stderr, "error reading file or file empty")
		os.Exit(1)
	}
	pattern := buildPattern(hosts)
	results := generatePermutations(pattern, hosts, 5)
	for _, h := range results {
		fmt.Println(h)
	}
}