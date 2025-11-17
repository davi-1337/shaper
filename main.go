package main

import (
	"bufio"
	"context"
	"fmt"
	"os"
	"regexp"
	"runtime"
	"sort"
	"strconv"
	"strings"
	"sync"
	"sync/atomic"
)

type SafeSet struct {
	mu    sync.RWMutex
	set   map[string]struct{}
	limit int
	count int64
}

func NewSafeSet(limit int) *SafeSet {
	return &SafeSet{
		set:   make(map[string]struct{}, limit),
		limit: limit,
		count: 0,
	}
}

func (s *SafeSet) Add(host string) bool {
	current := atomic.LoadInt64(&s.count)
	if int(current) >= s.limit {
		return true
	}

	s.mu.Lock()
	if len(s.set) >= s.limit {
		s.mu.Unlock()
		return true
	}
	if _, exists := s.set[host]; !exists {
		s.set[host] = struct{}{}
		atomic.AddInt64(&s.count, 1)
	}
	limitReached := len(s.set) >= s.limit
	s.mu.Unlock()
	return limitReached
}

func (s *SafeSet) Len() int {
	return int(atomic.LoadInt64(&s.count))
}

func (s *SafeSet) Keys() []string {
	s.mu.RLock()
	keys := make([]string, 0, len(s.set))
	for k := range s.set {
		keys = append(keys, k)
	}
	s.mu.RUnlock()
	return keys
}

type pattern struct {
	posFreq   map[int]map[string]int
	labelFreq map[string]int
	lengths   map[int]int
	base      string
	extractor *PatternExtractor
}

type PatternExtractor struct {
	keywords       map[string]int
	separators     map[string]int
	environments   []string
	services       []string
	versions       []string
	wordPairs      map[string][]string
	commonPrefixes []string
	commonSuffixes []string
}

func readLines(path string) ([]string, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	var lines []string
	sc := bufio.NewScanner(f)
	buf := make([]byte, 0, 1024*1024)
	sc.Buffer(buf, 10*1024*1024)
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
	suffixCounts := make(map[string]int, 1000)
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
		posFreq:   make(map[int]map[string]int, 100),
		labelFreq: make(map[string]int, 10000),
		lengths:   make(map[int]int, 20),
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
				p.posFreq[i] = make(map[string]int, 1000)
			}
			p.posFreq[i][lbl]++
			p.labelFreq[lbl]++
		}
	}

	p.extractor = extractPatterns(hosts, p)
	return p
}

func extractPatterns(hosts []string, p *pattern) *PatternExtractor {
	ex := &PatternExtractor{
		keywords:     make(map[string]int, 5000),
		separators:   make(map[string]int, 10),
		wordPairs:    make(map[string][]string, 2000),
		environments: []string{},
		services:     []string{},
		versions:     []string{},
	}

	envKeywords := map[string]bool{
		"dev": true, "development": true, "staging": true, "stage": true, "stg": true,
		"prod": true, "production": true, "test": true, "testing": true, "qa": true,
		"uat": true, "demo": true, "sandbox": true, "preprod": true, "beta": true,
		"alpha": true, "canary": true, "preview": true, "local": true,
	}

	serviceKeywords := map[string]bool{
		"api": true, "app": true, "web": true, "www": true, "admin": true,
		"portal": true, "dashboard": true, "cdn": true, "static": true, "assets": true,
		"mail": true, "email": true, "smtp": true, "imap": true, "ftp": true,
		"vpn": true, "ssh": true, "git": true, "gitlab": true, "jenkins": true,
		"db": true, "database": true, "redis": true, "mongo": true, "sql": true,
		"auth": true, "login": true, "sso": true, "oauth": true, "gateway": true,
		"proxy": true, "lb": true, "loadbalancer": true, "cache": true,
	}

	versionPattern := regexp.MustCompile(`^v\d+$|^\d{4}$|^v\d+\.\d+$`)

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

		for i, lbl := range labels {
			if lbl == "" {
				continue
			}

			if strings.Contains(lbl, "-") {
				ex.separators["-"]++
				parts := strings.Split(lbl, "-")
				for _, part := range parts {
					if len(part) > 1 {
						ex.keywords[part]++
						if envKeywords[part] {
							ex.environments = append(ex.environments, part)
						}
						if serviceKeywords[part] {
							ex.services = append(ex.services, part)
						}
						if versionPattern.MatchString(part) {
							ex.versions = append(ex.versions, part)
						}
					}
				}
				if len(parts) == 2 && parts[0] != "" && parts[1] != "" {
					ex.wordPairs[parts[0]] = append(ex.wordPairs[parts[0]], parts[1])
				}
			} else if strings.Contains(lbl, "_") {
				ex.separators["_"]++
				parts := strings.Split(lbl, "_")
				for _, part := range parts {
					if len(part) > 1 {
						ex.keywords[part]++
						if envKeywords[part] {
							ex.environments = append(ex.environments, part)
						}
						if serviceKeywords[part] {
							ex.services = append(ex.services, part)
						}
						if versionPattern.MatchString(part) {
							ex.versions = append(ex.versions, part)
						}
					}
				}
				if len(parts) == 2 && parts[0] != "" && parts[1] != "" {
					ex.wordPairs[parts[0]] = append(ex.wordPairs[parts[0]], parts[1])
				}
			} else {
				ex.separators[""]++
				ex.keywords[lbl]++
				if envKeywords[lbl] {
					ex.environments = append(ex.environments, lbl)
				}
				if serviceKeywords[lbl] {
					ex.services = append(ex.services, lbl)
				}
				if versionPattern.MatchString(lbl) {
					ex.versions = append(ex.versions, lbl)
				}
			}

			if i > 0 && labels[i-1] != "" {
				ex.wordPairs[labels[i-1]] = append(ex.wordPairs[labels[i-1]], lbl)
			}
		}
	}

	ex.environments = uniqueStrings(ex.environments)
	ex.services = uniqueStrings(ex.services)
	ex.versions = uniqueStrings(ex.versions)

	ex.extractPrefixesSuffixes(p.labelFreq)

	return ex
}

func uniqueStrings(input []string) []string {
	seen := make(map[string]bool, len(input))
	result := make([]string, 0, len(input))
	for _, s := range input {
		if !seen[s] {
			seen[s] = true
			result = append(result, s)
		}
	}
	return result
}

func (ex *PatternExtractor) extractPrefixesSuffixes(labelFreq map[string]int) {
	type kv struct {
		label string
		freq  int
	}
	items := make([]kv, 0, len(labelFreq))
	for l, f := range labelFreq {
		if len(l) >= 2 && f >= 2 {
			items = append(items, kv{label: l, freq: f})
		}
	}
	sort.Slice(items, func(i, j int) bool { return items[i].freq > items[j].freq })

	limit := 100
	if len(items) < limit {
		limit = len(items)
	}

	for i := 0; i < limit; i++ {
		ex.commonPrefixes = append(ex.commonPrefixes, items[i].label)
		ex.commonSuffixes = append(ex.commonSuffixes, items[i].label)
	}
}

var RE_NUMERIC = regexp.MustCompile(`^([a-zA-Z0-9\-]+?)(\d+)$`)

func expandNumericPatterns(p *pattern, rangeLimit int) {
	patterns := make(map[int]map[string]map[int][]int, 100)
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

				expansionCount := 0
				for n := min; n <= max && expansionCount <= rangeLimit; n++ {
					newLabel := fmt.Sprintf("%s%0*d", prefix, padding, n)

					if _, ok := p.posFreq[pos][newLabel]; !ok {
						p.posFreq[pos][newLabel] = 1
					}
					if _, ok := p.labelFreq[newLabel]; !ok {
						p.labelFreq[newLabel] = 1
					}
					expansionCount++
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

func generateCombinations(ctx context.Context, p *pattern, results *SafeSet, maxPerPos int) {
	lengths := p.topLengths(5)

	for _, length := range lengths {
		select {
		case <-ctx.Done():
			return
		default:
		}

		if results.Len() >= results.limit {
			return
		}
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
			select {
			case <-ctx.Done():
				return
			default:
			}

			if results.Len() >= results.limit {
				return
			}

			if pos == length {
				host := strings.Join(acc, ".") + "." + p.base
				results.Add(host)
				return
			}

			for _, lbl := range choices[pos] {
				if results.Len() >= results.limit {
					return
				}
				build(pos+1, append(acc, lbl))
			}
		}
		build(0, []string{})
	}
}

func generatePermutations(ctx context.Context, p *pattern, hostsChunk []string, results *SafeSet, topN int, maxIter int) {
	if p.extractor == nil {
		return
	}

	ex := p.extractor
	iterations := 0

	for _, h := range hostsChunk {
		select {
		case <-ctx.Done():
			return
		default:
		}

		if results.Len() >= results.limit {
			return
		}
		if iterations >= maxIter {
			return
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
		newLabels := make([]string, len(labels))

		for i := 0; i < len(labels); i++ {
			if results.Len() >= results.limit || iterations >= maxIter {
				return
			}

			originalLabel := labels[i]
			top := p.topLabels(i, topN)

			for _, newLabel := range top {
				if results.Len() >= results.limit || iterations >= maxIter {
					return
				}
				if newLabel == originalLabel {
					continue
				}
				copy(newLabels, labels)
				newLabels[i] = newLabel
				host := strings.Join(newLabels, ".") + "." + p.base
				results.Add(host)
				iterations++
			}
		}
	}

	for _, h := range hostsChunk {
		select {
		case <-ctx.Done():
			return
		default:
		}

		if results.Len() >= results.limit || iterations >= maxIter {
			return
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

		for i := 0; i < len(labels)-1; i++ {
			if results.Len() >= results.limit || iterations >= maxIter {
				return
			}

			w1, w2 := labels[i], labels[i+1]

			separators := []string{"", "-", "_"}
			for _, sep := range separators {
				if results.Len() >= results.limit || iterations >= maxIter {
					return
				}

				merged := w1 + sep + w2
				newLabels := make([]string, 0, len(labels)-1)
				newLabels = append(newLabels, labels[:i]...)
				newLabels = append(newLabels, merged)
				if i+2 < len(labels) {
					newLabels = append(newLabels, labels[i+2:]...)
				}

				if len(newLabels) > 0 {
					host := strings.Join(newLabels, ".") + "." + p.base
					results.Add(host)
					iterations++
				}
			}

			for _, sep := range separators {
				if results.Len() >= results.limit || iterations >= maxIter {
					return
				}

				merged := w2 + sep + w1
				newLabels := make([]string, 0, len(labels)-1)
				newLabels = append(newLabels, labels[:i]...)
				newLabels = append(newLabels, merged)
				if i+2 < len(labels) {
					newLabels = append(newLabels, labels[i+2:]...)
				}

				if len(newLabels) > 0 {
					host := strings.Join(newLabels, ".") + "." + p.base
					results.Add(host)
					iterations++
				}
			}
		}
	}

	if len(ex.environments) > 0 && len(ex.services) > 0 {
		separators := []string{"-", "", "_"}

		for _, env := range ex.environments {
			for _, svc := range ex.services {
				select {
				case <-ctx.Done():
					return
				default:
				}

				if results.Len() >= results.limit || iterations >= maxIter {
					return
				}

				for _, sep := range separators {
					if results.Len() >= results.limit || iterations >= maxIter {
						return
					}

					combo1 := env + sep + svc
					host1 := combo1 + "." + p.base
					results.Add(host1)
					iterations++

					combo2 := svc + sep + env
					host2 := combo2 + "." + p.base
					results.Add(host2)
					iterations++
				}
			}
		}
	}

	if len(ex.services) > 0 && len(ex.versions) > 0 {
		separators := []string{"-", "", "_"}

		for _, svc := range ex.services {
			for _, ver := range ex.versions {
				select {
				case <-ctx.Done():
					return
				default:
				}

				if results.Len() >= results.limit || iterations >= maxIter {
					return
				}

				for _, sep := range separators {
					if results.Len() >= results.limit || iterations >= maxIter {
						return
					}

					combo := svc + sep + ver
					host := combo + "." + p.base
					results.Add(host)
					iterations++
				}
			}
		}
	}

	for _, h := range hostsChunk {
		select {
		case <-ctx.Done():
			return
		default:
		}

		if results.Len() >= results.limit || iterations >= maxIter {
			return
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
		if len(labels) == 0 {
			continue
		}

		limit := 5
		if len(ex.commonPrefixes) < limit {
			limit = len(ex.commonPrefixes)
		}

		for i := 0; i < limit; i++ {
			if results.Len() >= results.limit || iterations >= maxIter {
				return
			}

			prefix := ex.commonPrefixes[i]

			newLabels := make([]string, 0, len(labels)+1)
			newLabels = append(newLabels, prefix)
			newLabels = append(newLabels, labels...)
			host := strings.Join(newLabels, ".") + "." + p.base
			results.Add(host)
			iterations++

			for _, sep := range []string{"-", "", "_"} {
				if results.Len() >= results.limit || iterations >= maxIter {
					return
				}

				newFirst := prefix + sep + labels[0]
				newLabels2 := make([]string, len(labels))
				copy(newLabels2, labels)
				newLabels2[0] = newFirst
				host2 := strings.Join(newLabels2, ".") + "." + p.base
				results.Add(host2)
				iterations++
			}
		}
	}
}

func generateLengthVariations(ctx context.Context, p *pattern, hostsChunk []string, results *SafeSet, topN int, maxIter int) {
	knownLengths := make(map[int]bool, len(p.lengths))
	for l := range p.lengths {
		knownLengths[l] = true
	}

	topPrefixes := p.topLabels(0, topN)
	iterations := 0

	for _, h := range hostsChunk {
		select {
		case <-ctx.Done():
			return
		default:
		}

		if results.Len() >= results.limit || iterations >= maxIter {
			return
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
		currentLen := len(labels)

		if currentLen > 1 && knownLengths[currentLen-1] {
			if results.Len() >= results.limit || iterations >= maxIter {
				return
			}
			host := strings.Join(labels[1:], ".") + "." + p.base
			results.Add(host)
			iterations++
		}

		if knownLengths[currentLen+1] {
			for _, prefix := range topPrefixes {
				if results.Len() >= results.limit || iterations >= maxIter {
					return
				}

				if prefix == labels[0] {
					continue
				}

				newHost := prefix + "." + strings.Join(labels, ".") + "." + p.base
				results.Add(newHost)
				iterations++
			}
		}
	}
}

func main() {
	runtime.GOMAXPROCS(12)

	if len(os.Args) < 2 {
		fmt.Fprintln(os.Stderr, "[-] erro: use shaper input.txt > output.txt")
		os.Exit(1)
	}
	in := os.Args[1]
	hosts, err := readLines(in)
	if err != nil || len(hosts) == 0 {
		fmt.Fprintf(os.Stderr, "[-] erro ao ler %s: %v\n", in, err)
		os.Exit(1)
	}

	const MAX_TOTAL = 1000000
	const RANGE_LIMIT = 30
	const TOP_N = 8
	const MAX_PER_POS = 8
	const MAX_ITER_PER_WORKER = 50000

	results := NewSafeSet(MAX_TOTAL)
	for _, h := range hosts {
		results.Add(h)
	}

	fmt.Fprintln(os.Stderr, "[+] Construindo padrão...")
	pattern := buildPattern(hosts)

	fmt.Fprintln(os.Stderr, "[+] Expandindo padrões numéricos...")
	expandNumericPatterns(pattern, RANGE_LIMIT)

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	var wg sync.WaitGroup

	wg.Add(1)
	go func() {
		defer wg.Done()
		fmt.Fprintln(os.Stderr, "[+] Iniciando combinações...")
		generateCombinations(ctx, pattern, results, MAX_PER_POS)
		fmt.Fprintln(os.Stderr, "[+] Combinações concluídas.")
	}()

	numWorkers := 12
	chunkSize := (len(hosts) + numWorkers - 1) / numWorkers

	fmt.Fprintf(os.Stderr, "[+] Iniciando permutações e variações (%d workers)...\n", numWorkers)

	for i := 0; i < numWorkers; i++ {
		start := i * chunkSize
		end := (i + 1) * chunkSize
		if end > len(hosts) {
			end = len(hosts)
		}
		if start >= end {
			continue
		}

		wg.Add(1)
		go func(chunk []string, workerID int) {
			defer wg.Done()
			generatePermutations(ctx, pattern, chunk, results, TOP_N, MAX_ITER_PER_WORKER)
			if results.Len() < results.limit {
				generateLengthVariations(ctx, pattern, chunk, results, TOP_N, MAX_ITER_PER_WORKER)
			}
			fmt.Fprintf(os.Stderr, "[+] Worker %d concluído\n", workerID)
		}(hosts[start:end], i+1)
	}

	go func() {
		for {
			if results.Len() >= results.limit {
				fmt.Fprintln(os.Stderr, "[+] Limite atingido, cancelando workers...")
				cancel()
				return
			}
			select {
			case <-ctx.Done():
				return
			default:
			}
		}
	}()

	wg.Wait()
	fmt.Fprintln(os.Stderr, "[+] Todas as tarefas concluídas.")

	finalHosts := results.Keys()
	fmt.Fprintln(os.Stderr, "[+] Ordenando resultados...")
	sort.Strings(finalHosts)

	for _, h := range finalHosts {
		fmt.Println(h)
	}

	fmt.Fprintf(os.Stderr, "[+] trabalho concluído. total: %d\n", len(finalHosts))
}
