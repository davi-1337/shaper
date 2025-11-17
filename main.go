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
	extractor *PatternExtractor
}

// PatternExtractor extrai padrões inteligentes dos subdomínios
type PatternExtractor struct {
	keywords         map[string]int       // Palavras-chave com frequência
	separators       map[string]int       // Separadores observados (-, _, "")
	environments     []string             // dev, staging, prod, test, qa, uat
	services         []string             // api, app, admin, portal, dashboard
	versions         []string             // v1, v2, 2023, 2024, etc
	wordPairs        map[string][]string  // Pares de palavras que aparecem juntas
	commonPrefixes   []string             // Prefixos mais comuns
	commonSuffixes   []string             // Sufixos mais comuns
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

	// Extrai padrões inteligentes
	p.extractor = extractPatterns(hosts, p)
	return p
}

// extractPatterns analisa os hosts e extrai padrões para permutações criativas
func extractPatterns(hosts []string, p *pattern) *PatternExtractor {
	ex := &PatternExtractor{
		keywords:     make(map[string]int, 500),
		separators:   make(map[string]int),
		wordPairs:    make(map[string][]string, 200),
		environments: []string{},
		services:     []string{},
		versions:     []string{},
	}

	// Categorias conhecidas
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

	// Analisa cada label para extrair padrões
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

		// Analisa cada label
		for i, lbl := range labels {
			if lbl == "" {
				continue
			}

			// Detecta separadores dentro do label e categoriza as partes
			if strings.Contains(lbl, "-") {
				ex.separators["-"]++
				parts := strings.Split(lbl, "-")
				for _, part := range parts {
					if len(part) > 1 {
						ex.keywords[part]++
						// Categoriza cada parte
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
				// Rastreia pares
				if len(parts) == 2 && parts[0] != "" && parts[1] != "" {
					ex.wordPairs[parts[0]] = append(ex.wordPairs[parts[0]], parts[1])
				}
			} else if strings.Contains(lbl, "_") {
				ex.separators["_"]++
				parts := strings.Split(lbl, "_")
				for _, part := range parts {
					if len(part) > 1 {
						ex.keywords[part]++
						// Categoriza cada parte
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
				// Label sem separador
				ex.separators[""]++
				ex.keywords[lbl]++
				// Categoriza o label completo
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

			// Rastreia pares de labels adjacentes
			if i > 0 && labels[i-1] != "" {
				ex.wordPairs[labels[i-1]] = append(ex.wordPairs[labels[i-1]], lbl)
			}
		}
	}

	// Remove duplicatas e mantém apenas os mais frequentes
	ex.environments = uniqueStrings(ex.environments)
	ex.services = uniqueStrings(ex.services)
	ex.versions = uniqueStrings(ex.versions)

	// Extrai prefixos e sufixos mais comuns dos labels
	ex.extractPrefixesSuffixes(p.labelFreq)

	return ex
}

func uniqueStrings(input []string) []string {
	seen := make(map[string]bool)
	result := []string{}
	for _, s := range input {
		if !seen[s] {
			seen[s] = true
			result = append(result, s)
		}
	}
	return result
}

func (ex *PatternExtractor) extractPrefixesSuffixes(labelFreq map[string]int) {
	// Extrai os top N labels como potenciais prefixos/sufixos
	type kv struct {
		label string
		freq  int
	}
	items := make([]kv, 0, len(labelFreq))
	for l, f := range labelFreq {
		if len(l) >= 2 && f >= 2 { // Mínimo de 2 caracteres e frequência >= 2
			items = append(items, kv{label: l, freq: f})
		}
	}
	sort.Slice(items, func(i, j int) bool { return items[i].freq > items[j].freq })

	limit := 50
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

// OTIMIZAÇÃO: Geração criativa de permutações usando padrões extraídos
// Esta função implementa 5 estratégias de permutação:
// 1. Swapping posicional (original): Troca labels por outros comuns na mesma posição
// 2. Word Concatenation: Mescla labels adjacentes com diferentes separadores (-, _, "")
//    Exemplos: dev.api → dev-api, devapi, dev_api | api.admin → admin-api, apiadmin
// 3. Environment + Service: Combina ambientes com serviços de forma inteligente
//    Exemplos: dev+api → dev-api, api-dev, devapi, apidev
// 4. Version Combinations: Adiciona versões aos serviços
//    Exemplos: api+v1 → api-v1, apiv1, api_v1
// 5. Prefix/Suffix Additions: Adiciona prefixos/sufixos comuns
//    Exemplos: api → dev-api, staging-api
func generatePermutations(p *pattern, hostsChunk []string, results *SafeSet, topN int) {
	if p.extractor == nil {
		return
	}

	ex := p.extractor

	// 1. Permutações baseadas em swapping (modo original, mas otimizado)
	for _, h := range hostsChunk {
		if results.Len() >= results.limit {
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

		// Swap com top labels por posição
		for i := 0; i < len(labels); i++ {
			originalLabel := labels[i]
			top := p.topLabels(i, topN)

			for _, newLabel := range top {
				if newLabel == originalLabel {
					continue
				}
				copy(newLabels, labels)
				newLabels[i] = newLabel
				host := strings.Join(newLabels, ".") + "." + p.base
				if results.Add(host) {
					return
				}
			}
		}
	}

	// 2. NOVO: Word Concatenation - Mescla palavras com diferentes separadores
	for _, h := range hostsChunk {
		if results.Len() >= results.limit {
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

		// Para cada label, tenta mesclar com o próximo
		for i := 0; i < len(labels)-1; i++ {
			if results.Len() >= results.limit {
				return
			}

			w1, w2 := labels[i], labels[i+1]

			// Concatena com diferentes separadores
			separators := []string{"", "-", "_"}
			for _, sep := range separators {
				merged := w1 + sep + w2
				newLabels := make([]string, 0, len(labels)-1)
				newLabels = append(newLabels, labels[:i]...)
				newLabels = append(newLabels, merged)
				if i+2 < len(labels) {
					newLabels = append(newLabels, labels[i+2:]...)
				}

				if len(newLabels) > 0 {
					host := strings.Join(newLabels, ".") + "." + p.base
					if results.Add(host) {
						return
					}
				}
			}

			// Inverte ordem: w2 + sep + w1
			for _, sep := range separators {
				merged := w2 + sep + w1
				newLabels := make([]string, 0, len(labels)-1)
				newLabels = append(newLabels, labels[:i]...)
				newLabels = append(newLabels, merged)
				if i+2 < len(labels) {
					newLabels = append(newLabels, labels[i+2:]...)
				}

				if len(newLabels) > 0 {
					host := strings.Join(newLabels, ".") + "." + p.base
					if results.Add(host) {
						return
					}
				}
			}
		}
	}

	// 3. NOVO: Environment + Service Combinations
	// Gera combinações inteligentes tipo: api-prod, dev-admin, staging-portal
	if len(ex.environments) > 0 && len(ex.services) > 0 {
		separators := []string{"-", "", "_"}

		for _, env := range ex.environments {
			for _, svc := range ex.services {
				if results.Len() >= results.limit {
					return
				}

				for _, sep := range separators {
					// env + sep + svc (ex: dev-api, prodapi)
					combo1 := env + sep + svc
					host1 := combo1 + "." + p.base
					if results.Add(host1) {
						return
					}

					// svc + sep + env (ex: api-dev, apiprod)
					combo2 := svc + sep + env
					host2 := combo2 + "." + p.base
					if results.Add(host2) {
						return
					}
				}
			}
		}
	}

	// 4. NOVO: Version Combinations
	// Gera: api-v1, api-v2, service-2024, etc
	if len(ex.services) > 0 && len(ex.versions) > 0 {
		separators := []string{"-", "", "_"}

		for _, svc := range ex.services {
			for _, ver := range ex.versions {
				if results.Len() >= results.limit {
					return
				}

				for _, sep := range separators {
					combo := svc + sep + ver
					host := combo + "." + p.base
					if results.Add(host) {
						return
					}
				}
			}
		}
	}

	// 5. NOVO: Prefix/Suffix Additions
	// Adiciona prefixos/sufixos comuns aos labels existentes
	for _, h := range hostsChunk {
		if results.Len() >= results.limit {
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

		// Adiciona prefixos aos primeiros labels (até 3 prefixos)
		limit := 3
		if len(ex.commonPrefixes) < limit {
			limit = len(ex.commonPrefixes)
		}

		for i := 0; i < limit; i++ {
			if results.Len() >= results.limit {
				return
			}
			prefix := ex.commonPrefixes[i]

			// Adiciona como novo label no início
			newLabels := make([]string, 0, len(labels)+1)
			newLabels = append(newLabels, prefix)
			newLabels = append(newLabels, labels...)
			host := strings.Join(newLabels, ".") + "." + p.base
			if results.Add(host) {
				return
			}

			// Concatena com primeiro label
			for _, sep := range []string{"-", "", "_"} {
				newFirst := prefix + sep + labels[0]
				newLabels2 := make([]string, len(labels))
				copy(newLabels2, labels)
				newLabels2[0] = newFirst
				host2 := strings.Join(newLabels2, ".") + "." + p.base
				if results.Add(host2) {
					return
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
