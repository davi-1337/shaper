package main

import (
	"bufio"
	"fmt"
	"os"
	"regexp"
	"sort"
	"strconv"
	"strings"
	"sync" // Importado para uso futuro, embora não estritamente necessário para esta refatoração
)

// Otimização: Usaremos um pool para strings.Builder para reduzir alocações de memória
// Embora a concatenação de strings em Go seja otimizada, para geração massiva,
// um builder pode ser ligeiramente mais eficiente.
var builderPool = sync.Pool{
	New: func() interface{} {
		return &strings.Builder{}
	},
}

type pattern struct {
	posFreq   map[int]map[string]int
	labelFreq map[string]int
	lengths   map[int]int
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

// OTIMIZAÇÃO: Modificado para adicionar resultados diretamente ao map e respeitar o limite total
func generateCombinations(p *pattern, results map[string]struct{}, maxPerPos int, maxTotal int) {
	lengths := p.topLengths(3) // Foca nos 3 comprimentos mais comuns

	// Otimização: Usar um string builder do pool
	sb := builderPool.Get().(*strings.Builder)
	defer builderPool.Put(sb)

	for _, length := range lengths {
		if len(results) >= maxTotal {
			return // Para se o limite global foi atingido
		}
		if length == 0 {
			continue
		}
		choices := make([][]string, length)
		totalChoices := 1
		for pos := 0; pos < length; pos++ {
			tops := p.topLabels(pos, maxPerPos)
			if len(tops) == 0 {
				tops = fallbackLabels(p, maxPerPos) // Fallback
			}
			choices[pos] = tops
			if len(tops) > 0 {
				totalChoices *= len(tops)
			}
		}

		// Otimização: Se a combinação deste comprimento for explodir o limite,
		// podemos pular ou ser mais espertos, mas por enquanto,
		// a verificação interna da 'build' cuidará disso.

		var build func(pos int, acc []string)
		build = func(pos int, acc []string) {
			// Verificação de limite em *cada* chamada recursiva
			if len(results) >= maxTotal {
				return
			}

			if pos == length {
				// Usa o string builder para eficiência
				sb.Reset()
				for i, lbl := range acc {
					sb.WriteString(lbl)
					if i < len(acc)-1 {
						sb.WriteRune('.')
					}
				}
				sb.WriteRune('.')
				sb.WriteString(p.base)
				host := sb.String()

				results[host] = struct{}{}
				return
			}

			for _, lbl := range choices[pos] {
				// Otimização: a checagem do limite é feita na próxima chamada recursiva
				build(pos+1, append(acc, lbl))
				// Se a chamada interna atingiu o limite, paramos este loop também
				if len(results) >= maxTotal {
					return
				}
			}
		}
		build(0, []string{})
	}
}

// OTIMIZAÇÃO: Modificado para adicionar resultados diretamente ao map e respeitar o limite total
func generatePermutations(p *pattern, hosts []string, results map[string]struct{}, topN int, maxTotal int) {
	sb := builderPool.Get().(*strings.Builder)
	defer builderPool.Put(sb)

	for _, h := range hosts {
		if len(results) >= maxTotal {
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

				// Evita alocação excessiva de 'copy'
				copy(newLabels, labels)
				newLabels[i] = newLabel

				// Usa o string builder
				sb.Reset()
				for j, lbl := range newLabels {
					sb.WriteString(lbl)
					if j < len(newLabels)-1 {
						sb.WriteRune('.')
					}
				}
				sb.WriteRune('.')
				sb.WriteString(p.base)
				host := sb.String()

				results[host] = struct{}{}

				if len(results) >= maxTotal {
					return // Limite global atingido
				}
			}
		}
	}
}

// OTIMIZAÇÃO: Removida a função 'generateMultiPermutations'
// Esta função é a causa de maior explosão combinatória e lentidão,
// gerando muitos subdomínios de baixa probabilidade.
// Removê-la aumenta a velocidade e foca em resultados mais "estratégicos".

// OTIMIZAÇÃO: Modificado para adicionar resultados diretamente ao map e respeitar o limite total
func generateLengthVariations(p *pattern, hosts []string, results map[string]struct{}, topN int, maxTotal int) {
	knownLengths := make(map[int]bool)
	for l := range p.lengths {
		knownLengths[l] = true
	}

	topPrefixes := p.topLabels(0, topN) // Top N labels da *primeira* posição (pos 0)

	sb := builderPool.Get().(*strings.Builder)
	defer builderPool.Put(sb)

	for _, h := range hosts {
		if len(results) >= maxTotal {
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

		// 1. Tenta encurtar (ex: de 'a.b.c.base' para 'b.c.base')
		// Só faz se o comprimento resultante (len-1) for um comprimento conhecido
		if currentLen > 1 && knownLengths[currentLen-1] {
			sb.Reset()
			for i := 1; i < len(labels); i++ { // Começa do índice 1
				sb.WriteString(labels[i])
				if i < len(labels)-1 {
					sb.WriteRune('.')
				}
			}
			sb.WriteRune('.')
			sb.WriteString(p.base)
			host := sb.String()
			results[host] = struct{}{}
			if len(results) >= maxTotal {
				return
			}
		}

		// 2. Tenta alongar (ex: de 'b.c.base' para 'PRE.b.c.base')
		// Só faz se o comprimento resultante (len+1) for um comprimento conhecido
		if knownLengths[currentLen+1] {
			for _, prefix := range topPrefixes {
				if prefix == labels[0] { // Evita adicionar o mesmo prefixo
					continue
				}

				sb.Reset()
				sb.WriteString(prefix)
				sb.WriteRune('.')
				for i, lbl := range labels {
					sb.WriteString(lbl)
					if i < len(labels)-1 {
						sb.WriteRune('.')
					}
				}
				sb.WriteRune('.')
				sb.WriteString(p.base)
				host := sb.String()
				results[host] = struct{}{}
				if len(results) >= maxTotal {
					return
				}
			}
		}
	}
}

// Otimização: Função helper para extrair chaves de map
func mapKeys(m map[string]struct{}) []string {
	keys := make([]string, 0, len(m))
	for k := range m {
		keys = append(keys, k)
	}
	return keys
}

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

	// OTIMIZAÇÃO: Limite máximo
	const MAX_TOTAL = 500000

	// OTIMIZAÇÃO: Parâmetros estratégicos (menos palavras)
	const RANGE_LIMIT = 20 // Limite para expandir números (ex: web01, web02...)
	const TOP_N = 5        // Quantas palavras "top" usar para permutações
	const MAX_PER_POS = 5  // Quantas palavras por posição nas combinações

	// OTIMIZAÇÃO: Usar um map para coletar resultados e garantir unicidade
	results := make(map[string]struct{}, MAX_TOTAL)
	for _, h := range hosts {
		results[h] = struct{}{}
	}

	pattern := buildPattern(hosts)

	// 1. Expandir padrões numéricos (ex: web01, web02 -> web01..web20)
	expandNumericPatterns(pattern, RANGE_LIMIT)

	// 2. Gerar Combinações (ex: [dev,stg] + [web,db] -> dev.web, dev.db, stg.web, stg.db)
	generateCombinations(pattern, results, MAX_PER_POS, MAX_TOTAL)
	if len(results) >= MAX_TOTAL {
		fmt.Fprintln(os.Stderr, "[!] Atingiu o limite máximo durante as combinações.")
	}

	// 3. Gerar Permutações (ex: dev.web.base -> stg.web.base, prod.web.base)
	if len(results) < MAX_TOTAL {
		generatePermutations(pattern, hosts, results, TOP_N, MAX_TOTAL)
	}
	if len(results) >= MAX_TOTAL {
		fmt.Fprintln(os.Stderr, "[!] Atingiu o limite máximo durante as permutações.")
	}

	// 4. Gerar Variações de Comprimento (ex: a.b.base -> b.base | ex: b.base -> a.b.base)
	if len(results) < MAX_TOTAL {
		generateLengthVariations(pattern, hosts, results, TOP_N, MAX_TOTAL)
	}
	if len(results) >= MAX_TOTAL {
		fmt.Fprintln(os.Stderr, "[!] Atingiu o limite máximo durante as variações de comprimento.")
	}

	// OTIMIZAÇÃO: Não precisamos mais de 'unique(all)',
	// o map 'results' já cuidou da unicidade.
	// Apenas precisamos extrair as chaves e ordená-las.
	finalHosts := mapKeys(results)
	sort.Strings(finalHosts)

	for _, h := range finalHosts {
		fmt.Println(h)
	}

	fmt.Fprintf(os.Stderr, "[+] trabalho concluído. total de subdomínios: %d\n", len(finalHosts))
}
