package main

import (
	"bufio"
	"fmt"
	"os"
	"regexp"
	"sort"
	"strconv"
	"strings"
)

type pattern struct {
	posFreq   map[int]map[string]int
	labelFreq map[string]int
	lengths   map[int]int
	base      string
}

type numericPattern struct {
	prefix  string
	padding int
	nums    []int
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
		lines = append(lines, strings.ToLower(line))
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

	suffixCounts := make(map[string]int)
	for _, h := range hosts {
		parts := strings.Split(h, ".")
		if len(parts) < 3 {
			continue
		}

		s2 := strings.Join(parts[len(parts)-2:], ".")
		suffixCounts[s2]++

		if len(parts) > 3 {
			s3 := strings.Join(parts[len(parts)-3:], ".")
			suffixCounts[s3]++
		}

		if len(parts) > 4 {
			s4 := strings.Join(parts[len(parts)-4:], ".")
			suffixCounts[s4]++
		}
	}

	if len(suffixCounts) == 0 {
		parts := strings.Split(hosts[0], ".")
		if len(parts) < 2 {
			return hosts[0]
		}
		return strings.Join(parts[len(parts)-2:], ".")
	}

	maxFreq := 0
	bestBase := ""
	for suffix, freq := range suffixCounts {
		if freq > maxFreq {
			maxFreq = freq
			bestBase = suffix
		} else if freq == maxFreq && len(suffix) > len(bestBase) {
			bestBase = suffix
		}
	}
	return bestBase
}

func buildPattern(hosts []string) *pattern {
	base := guessBase(hosts)
	p := &pattern{
		posFreq:   make(map[int]map[string]int),
		labelFreq: make(map[string]int),
		lengths:   make(map[int]int),
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
		length := len(labels)
		p.lengths[length]++
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

var RE_NUMERIC = regexp.MustCompile(`^([a-zA-Z0-9\-]+?)(\d+)$`)

func expandNumericPatterns(p *pattern, rangeLimit int) {
	patterns := make(map[int]map[string]map[int][]int)

	for pos, labels := range p.posFreq {
		for lbl := range labels {
			matches := RE_NUMERIC.FindStringSubmatch(lbl)
			if matches == nil {
				continue
			}
			prefix := matches[1]
			numStr := matches[2]
			if prefix == "" {
				continue
			}
			num, err := strconv.Atoi(numStr)
			if err != nil {
				continue
			}
			padding := len(numStr)

			if _, ok := patterns[pos]; !ok {
				patterns[pos] = make(map[string]map[int][]int)
			}
			if _, ok := patterns[pos][prefix]; !ok {
				patterns[pos][prefix] = make(map[int][]int)
			}
			patterns[pos][prefix][padding] = append(patterns[pos][prefix][padding], num)
		}
	}

	for pos, prefixes := range patterns {
		for prefix, paddings := range prefixes {
			for padding, nums := range paddings {
				if len(nums) < 2 {
					continue
				}
				min, max := nums[0], nums[0]
				for _, n := range nums[1:] {
					if n < min {
						min = n
					}
					if n > max {
						max = n
					}
				}
				if max-min > rangeLimit {
					continue
				}

				for n := min; n <= max; n++ {
					newLabel := fmt.Sprintf("%s%0*d", prefix, padding, n)

					if _, ok := p.posFreq[pos][newLabel]; !ok {
						p.posFreq[pos][newLabel] = 1
					}
					if _, ok := p.labelFreq[newLabel]; !ok {
						p.labelFreq[newLabel] = 1
					}
				}
			}
		}
	}
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
	if limit > len(tmp) || limit == -1 {
		limit = len(tmp)
	}
	out := make([]string, 0, limit)
	for i := 0; i < limit; i++ {
		out = append(out, tmp[i].label)
	}
	return out
}

func (p *pattern) topLengths(limit int) []int {
	type kvl struct {
		l int
		f int
	}
	tmp := make([]kvl, 0, len(p.lengths))
	for l, f := range p.lengths {
		tmp = append(tmp, kvl{l: l, f: f})
	}
	sort.Slice(tmp, func(i, j int) bool { return tmp[i].f > tmp[j].f })
	if limit > len(tmp) || limit == -1 {
		limit = len(tmp)
	}
	out := make([]int, 0, limit)
	for i := 0; i < limit; i++ {
		out = append(out, tmp[i].l)
	}
	return out
}

func fallbackLabels(p *pattern, limit int) []string {
	tmp := make([]kv, 0, len(p.labelFreq))
	for l, f := range p.labelFreq {
		tmp = append(tmp, kv{label: l, freq: f})
	}
	sort.Slice(tmp, func(i, j int) bool { return tmp[i].freq > tmp[j].freq })
	if limit > len(tmp) || limit == -1 {
		limit = len(tmp)
	}
	out := make([]string, 0, limit)
	for i := 0; i < limit; i++ {
		out = append(out, tmp[i].label)
	}
	return out
}

func generateCombinations(p *pattern, maxPerPos int, maxTotal int) []string {
	lengths := p.topLengths(3)
	candidates := make(map[string]struct{})
	for _, length := range lengths {
		if length == 0 {
			continue
		}
		choices := make([][]string, length)
		for pos := 0; pos < length; pos++ {
			tops := p.topLabels(pos, maxPerPos)
			if len(tops) == 0 {
				tops = fallbackLabels(p, maxPerPos)
			}
			choices[pos] = tops
		}
		var build func(pos int, acc []string)
		build = func(pos int, acc []string) {
			if len(candidates) >= maxTotal {
				return
			}
			if pos == length {
				host := strings.Join(acc, ".") + "." + p.base
				candidates[host] = struct{}{}
				return
			}
			for _, lbl := range choices[pos] {
				next := append(acc, lbl)
				build(pos+1, next)
				if len(candidates) >= maxTotal {
					return
				}
			}
		}
		build(0, []string{})
		if len(candidates) >= maxTotal {
			break
		}
	}
	out := make([]string, 0, len(candidates))
	for h := range candidates {
		out = append(out, h)
	}
	sort.Strings(out)
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

func generateMultiPermutations(p *pattern, hosts []string, topN int, maxTotal int) []string {
	candidates := make(map[string]struct{})
	for _, h := range hosts {
		if len(candidates) >= maxTotal {
			break
		}
		if !strings.HasSuffix(h, p.base) {
			continue
		}
		sub := strings.TrimSuffix(h, p.base)
		sub = strings.TrimSuffix(sub, ".")
		if sub == "" {
			continue
		}
		labels := strings.Split(sub, ".")
		if len(labels) < 2 {
			continue
		}

		for i := 0; i < len(labels); i++ {
			for j := i + 1; j < len(labels); j++ {
				topI := p.topLabels(i, topN)
				topJ := p.topLabels(j, topN)
				for _, newLabelI := range topI {
					if newLabelI == labels[i] {
						continue
					}
					for _, newLabelJ := range topJ {
						if newLabelJ == labels[j] {
							continue
						}
						newLabels := make([]string, len(labels))
						copy(newLabels, labels)
						newLabels[i] = newLabelI
						newLabels[j] = newLabelJ
						host := strings.Join(newLabels, ".") + "." + p.base
						candidates[host] = struct{}{}
						if len(candidates) >= maxTotal {
							return unique(mapKeys(candidates))
						}
					}
				}
			}
		}
	}
	return unique(mapKeys(candidates))
}

func generateLengthVariations(p *pattern, hosts []string, topN int) []string {
	candidates := make(map[string]struct{})
	knownLengths := make(map[int]bool)
	for l := range p.lengths {
		knownLengths[l] = true
	}

	topPrefixes := p.topLabels(0, topN)

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
		currentLen := len(labels)

		if currentLen > 1 && knownLengths[currentLen-1] {
			newLabels := labels[1:]
			host := strings.Join(newLabels, ".") + "." + p.base
			candidates[host] = struct{}{}
		}

		if knownLengths[currentLen+1] {
			for _, prefix := range topPrefixes {
				if prefix == labels[0] {
					continue
				}
				newLabels := append([]string{prefix}, labels...)
				host := strings.Join(newLabels, ".") + "." + p.base
				candidates[host] = struct{}{}
			}
		}
	}
	return unique(mapKeys(candidates))
}

func mapKeys(m map[string]struct{}) []string {
	keys := make([]string, 0, len(m))
	for k := range m {
		keys = append(keys, k)
	}
	return keys
}

func main() {
	if len(os.Args) < 2 {
		fmt.Fprintln(os.Stderr, "[-] error uso: shaper input.txt > output.txt")
		os.Exit(1)
	}
	in := os.Args[1]
	hosts, err := readLines(in)
	if err != nil || len(hosts) == 0 {
		fmt.Fprintf(os.Stderr, "[-] error lendo %s: %v\n", in, err)
		os.Exit(1)
	}

	pattern := buildPattern(hosts)

	expandNumericPatterns(pattern, 30)

	a := generateCombinations(pattern, 10, 300000)
	b := generatePermutations(pattern, hosts, 10)
	c := generateMultiPermutations(pattern, hosts, 5, 50000)
	d := generateLengthVariations(pattern, hosts, 10)

	all := append(hosts, a...)
	all = append(all, b...)
	all = append(all, c...)
	all = append(all, d...)

	results := unique(all)
	sort.Strings(results)

	for _, h := range results {
		fmt.Println(h)
	}
	fmt.Fprintln(os.Stderr, "[+] worked :)")
}