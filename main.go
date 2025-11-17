package main

import (
	"bufio"
	"fmt"
	"os"
	"regexp"
	"runtime" // Importado para paralelismo
	"sort"
	"strconv"
	"strings"
	"sync" // Importado para paralelismo e SafeSet
)

// Estrutura de dados segura para concorrência (thread-safe) para armazenar os resultados.
// Usa um Mutex para proteger o map contra escritas simultâneas.
type SafeSet struct {
	mu    sync.Mutex
	set   map[string]struct{}
	limit int
}

// NewSafeSet cria um novo conjunto seguro com um limite.
func NewSafeSet(limit int) *SafeSet {
	return &SafeSet{
		set:   make(map[string]struct{}, limit),
		limit: limit,
	}
}

// Add adiciona um host ao conjunto.
// Retorna 'true' se o limite foi atingido (seja antes ou depois desta adição).
func (s *SafeSet) Add(host string) bool {
	s.mu.Lock()
	// Verifica o limite *antes* de adicionar, para não estourar o map
	if len(s.set) >= s.limit {
		s.mu.Unlock()
		return true // Limite já atingido
	}

	s.set[host] = struct{}{}
	// Verifica se esta adição atingiu o limite
	limitReached := len(s.set) >= s.limit
	s.mu.Unlock()
	return limitReached
}

// Len retorna o tamanho atual do conjunto de forma segura.
func (s *SafeSet) Len() int {
	s.mu.Lock()
	l := len(s.set)
	s.mu.Unlock()
	return l
}

// Keys retorna uma cópia de todas as chaves (hosts) do conjunto.
func (s *SafeSet) Keys() []string {
	s.mu.Lock()
	keys := make([]string, 0, len(s.set))
	for k := range s.set {
		keys = append(keys, k)
	}
	s.mu.Unlock()
	return keys
}

type pattern struct {
	posFreq   map[int]map[string]int
	labelFreq map[string]int
	lengths   map[int]int
	base      string
}

func readLines(path string) ([]string, error) {
	f, err := os.Open(path)
	// ... (código existente inalterado) ...
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
	// ... (código existente inalterado) ...
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
	// ... (código existente inalterado) ...
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
		// ... (código existente inalterado) ...
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
	// ... (código existente inalterado) ...
	for pos, labels := range p.posFreq {
		for lbl := range labels {
			matches := RE_NUMERIC.FindStringSubmatch(lbl)
			if matches == nil {
				continue
			}
			prefix := matches[1]
			numStr := matches[2]
			if prefix == "" {
				// Evitar padrões que são *apenas* números
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
				// Otimização: Se o range for > rangeLimit, não faz nada
				if max-min > rangeLimit {
					continue
				}

				// Otimização: Se o range for muito grande (ex: 1 a 1000)
				// e rangeLimit é 20, ainda geramos 20.
				// Vamos limitar a expansão total ao rangeLimit.
				expansionCount := 0
				for n := min; n <= max && expansionCount <= rangeLimit; n++ {
					// Formatando o número com o padding correto
					newLabel := fmt.Sprintf("%s%0*d", prefix, padding, n)

					if _, ok := p.posFreq[pos][newLabel]; !ok {
						p.posFreq[pos][newLabel] = 1 // Damos frequência baixa para não dominar
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
	// ... (código existente inalterado) ...
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
	// ... (código existente inalterado) ...
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
	// ... (código existente inalterado) ...
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

// OTIMIZAÇÃO: Modificado para usar o SafeSet e parar se o limite for atingido.
func generateCombinations(p *pattern, results *SafeSet, maxPerPos int) {
	lengths := p.topLengths(3) // Foca nos 3 comprimentos mais comuns

	for _, length := range lengths {
		if results.Len() >= results.limit {
			return // Para se o limite global foi atingido
		}
		if length == 0 {
			continue
		}
		choices := make([][]string, length)
		// ... (código existente inalterado) ...
		for pos := 0; pos < length; pos++ {
			tops := p.topLabels(pos, maxPerPos)
			if len(tops) == 0 {
				tops = fallbackLabels(p, maxPerPos) // Fallback
			}
			choices[pos] = tops
		}

		var build func(pos int, acc []string)
		build = func(pos int, acc []string) {
			// Verificação de limite em *cada* chamada recursiva
			if results.Len() >= results.limit {
				return
			}

			if pos == length {
				host := strings.Join(acc, ".") + "." + p.base
				// Adiciona ao SafeSet; se o limite for atingido, 'return'
				if results.Add(host) {
					return
				}
				return
			}

			for _, lbl := range choices[pos] {
				build(pos+1, append(acc, lbl))
				// Se a chamada interna atingiu o limite, paramos este loop também
				if results.Len() >= results.limit {
					return
				}
			}
		}
		build(0, []string{})
	}
}

// OTIMIZAÇÃO: Modificado para usar o SafeSet e aceitar um "chunk" de hosts.
func generatePermutations(p *pattern, hostsChunk []string, results *SafeSet, topN int) {
	for _, h := range hostsChunk {
		// Verificação de limite no início de cada iteração
		if results.Len() >= results.limit {
			return // Limite global atingido
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
			originalLabel := labels[i]
			top := p.topLabels(i, topN) // Pega os N mais comuns para esta *posição*

			for _, newLabel := range top {
				if newLabel == originalLabel {
					continue
				}

				copy(newLabels, labels)
				newLabels[i] = newLabel
				host := strings.Join(newLabels, ".") + "." + p.base

				// Adiciona ao SafeSet; se o limite for atingido, para tudo.
				if results.Add(host) {
					return // Limite global atingido
				}
			}
		}
	}
}

// OTIMIZAÇÃO: Modificado para usar o SafeSet e aceitar um "chunk" de hosts.
func generateLengthVariations(p *pattern, hostsChunk []string, results *SafeSet, topN int) {
	knownLengths := make(map[int]bool)
	for l := range p.lengths {
		knownLengths[l] = true
	}

	topPrefixes := p.topLabels(0, topN) // Top N labels da *primeira* posição (pos 0)

	for _, h := range hostsChunk {
		if results.Len() >= results.limit {
			return // Limite global
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

		// 1. Tenta encurtar
		if currentLen > 1 && knownLengths[currentLen-1] {
			// Pula o primeiro label (labels[0])
			host := strings.Join(labels[1:], ".") + "." + p.base
			if results.Add(host) {
				return
			}
		}

		// 2. Tenta alongar
		if knownLengths[currentLen+1] {
			for _, prefix := range topPrefixes {
				if prefix == labels[0] { // Evita adicionar o mesmo prefixo
					continue
				}

				// Criando host com novo prefixo
				newHost := prefix + "." + strings.Join(labels, ".") + "." + p.base
				if results.Add(newHost) {
					return
				}
			}
		}
	}
}

// mapKeys agora é obsoleto, pois SafeSet tem seu próprio método Keys()

func main() {
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

	const MAX_TOTAL = 500000
	const RANGE_LIMIT = 20 // Limite para expandir números
	const TOP_N = 5        // Quantas palavras "top" usar
	const MAX_PER_POS = 5  // Quantas palavras por posição nas combinações

	// OTIMIZAÇÃO: Usar o SafeSet para coletar resultados de forma concorrente
	results := NewSafeSet(MAX_TOTAL)
	for _, h := range hosts {
		results.Add(h) // Adiciona a lista inicial
	}

	fmt.Fprintln(os.Stderr, "[+] Construindo padrão...")
	pattern := buildPattern(hosts)

	// 1. Expandir padrões numéricos (rápido, pode ser síncrono)
	fmt.Fprintln(os.Stderr, "[+] Expandindo padrões numéricos...")
	expandNumericPatterns(pattern, RANGE_LIMIT)

	// OTIMIZAÇÃO: Usar um WaitGroup para gerenciar goroutines
	var wg sync.WaitGroup

	// 2. Gerar Combinações (em sua própria goroutine)
	wg.Add(1)
	go func() {
		defer wg.Done()
		fmt.Fprintln(os.Stderr, "[+] Iniciando combinações (thread 1)...")
		generateCombinations(pattern, results, MAX_PER_POS)
		fmt.Fprintln(os.Stderr, "[+] Combinações concluídas.")
	}()

	// 3. & 4. Gerar Permutações e Variações de Comprimento (em paralelo)
	numCPU := runtime.NumCPU()
	if numCPU < 1 {
		numCPU = 1
	}
	// Cálculo para dividir os hosts uniformemente entre as CPUs
	chunkSize := (len(hosts) + numCPU - 1) / numCPU

	fmt.Fprintf(os.Stderr, "[+] Iniciando permutações e variações (usando %d CPUs)...\n", numCPU)

	for i := 0; i < numCPU; i++ {
		start := i * chunkSize
		end := (i + 1) * chunkSize
		if end > len(hosts) {
			end = len(hosts)
		}
		if start >= end {
			continue // Sem trabalho para esta goroutine
		}

		wg.Add(1)
		go func(chunk []string, workerID int) {
			defer wg.Done()
			// fmt.Fprintf(os.Stderr, "[+] Worker %d: Iniciando permutações...\n", workerID)
			generatePermutations(pattern, chunk, results, TOP_N)

			// Só executa a próxima tarefa se o limite não foi atingido
			if results.Len() < results.limit {
				// fmt.Fprintf(os.Stderr, "[+] Worker %d: Iniciando variações de comprimento...\n", workerID)
				generateLengthVariations(pattern, chunk, results, TOP_N)
			}
			// fmt.Fprintf(os.Stderr, "[+] Worker %d: Concluído.\n", workerID)
		}(hosts[start:end], i+1)
	}

	// Esperar todas as goroutines (Combinações + Workers) terminarem
	wg.Wait()
	fmt.Fprintln(os.Stderr, "[+] Todas as tarefas de geração concluídas.")

	// O SafeSet já cuidou da unicidade.
	// Apenas precisamos extrair as chaves e ordená-las.
	finalHosts := results.Keys()
	fmt.Fprintln(os.Stderr, "[+] Ordenando resultados...")
	sort.Strings(finalHosts)

	// Imprime os resultados
	for _, h := range finalHosts {
		fmt.Println(h)
	}

	fmt.Fprintf(os.Stderr, "[+] trabalho concluído. total de subdomínios: %d\n", len(finalHosts))
}
