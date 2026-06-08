# Hybrid Lossless Text Compression via Neural Network-Based Algorithm Selection and Large Language Model-Driven Dictionary Synthesis: A Case Study on Turkish Text

**Betül Arslan**, *Student Member, IEEE*

Department of Computer Engineering, Yıldız Technical University, Istanbul, Türkiye
E-mail: betularslan@yildiz.edu.tr

**Manuscript received:** June 2026.

---

## Abstract

This paper introduces a hybrid lossless data compression framework that synergistically combines three classical compression algorithms—Huffman coding [1], Lempel–Ziv–Welch (LZW) [3], and the Burrows–Wheeler Transform (BWT) pipeline [4]—with two artificial intelligence (AI) modalities: (i) a feedforward multilayer perceptron (MLP) for adaptive *algorithm selection*, formulated within Rice's Algorithm Selection Problem (ASP) framework [5], and (ii) a Large Language Model (LLM)-based *dictionary synthesizer* that generates source-adaptive initialization dictionaries for LZW. The proposed system is benchmarked on heterogeneous corpora including natural Turkish prose, structured logs, DNA sequences, and synthetic patterns. Empirical results demonstrate up to **85.9%** reduction over standard Huffman in repetitive data and competitive performance against industry-standard compressors (gzip, bzip2, lzma, zlib), surpassing bzip2 by **36.5%** on short Turkish texts. The MLP attains **95.2%** hold-out accuracy and **91.7% ± 1.8%** under stratified 5-fold cross-validation across 2,357 training instances. The framework is empirically validated through **100 unit tests** covering edge cases and is deployed as an interactive Streamlit application on HuggingFace Spaces. All source code, datasets, the trained model, and the AI interaction log (42 entries with 12 explicit literature cross-references) are publicly released under the MIT license.

**Index Terms** — Lossless data compression, Burrows–Wheeler transform, Huffman coding, Lempel–Ziv–Welch, Move-to-Front, algorithm selection problem, multilayer perceptron, large language models, prompt engineering, Turkish natural language processing, information theory.

---

## I. INTRODUCTION

### A. Motivation

Lossless text compression remains a fundamental problem in information theory and applied computer science, with widespread applications in archival storage, network transmission, and natural language processing pipelines. Since Shannon's seminal work [6], the theoretical lower bound for any lossless coder has been characterized by the source entropy

$$H(X) = -\sum_{c \in \Sigma} p(c) \log_2 p(c) \tag{1}$$

where $\Sigma$ denotes the source alphabet and $p(c)$ the probability mass function of symbol $c$. Practical compressors approximate this bound under various structural assumptions; however, no single algorithm uniformly dominates across all data modalities. Huffman coding [1] achieves optimality within the constraint of integer-length codewords, but its symbol-by-symbol nature limits its exploitation of higher-order correlations. LZW [3] excels on repetitive lexical patterns but suffers in low-redundancy contexts. BWT-based pipelines [4] dominate on structurally clustered data but require block-wise processing.

This data-dependent performance heterogeneity motivates the **Algorithm Selection Problem (ASP)** as formalized by Rice [5]:

> *Given a problem instance $x \in P$ and a set of candidate algorithms $A = \{a_1, a_2, \ldots, a_k\}$, select the algorithm $a^*$ that minimizes a performance metric $m(a, x)$.*

Recent advances in machine learning–based algorithm selection [7] have demonstrated that data-driven selectors can outperform any individual algorithm in heterogeneous workloads. We extend this paradigm to the lossless text compression domain with two AI integrations:

1. **A discriminative MLP-based selector** that maps a hand-crafted 11-dimensional feature vector to a categorical decision over $\{H, L, B\}$ (Huffman, LZW, BWT).
2. **An LLM-driven dictionary synthesizer** that augments the LZW initialization dictionary with corpus-adaptive lexical patterns.

### B. Contributions

The principal contributions of this work are:

1. **LLM-Augmented LZW (LZW-AI):** We formalize and implement a mechanism wherein a Large Language Model (LLaMA-3.3-70B via Groq Cloud) ingests a sample of the source text and produces a ranked list of frequent lexical units. These units are inserted as multi-byte entries into the LZW initialization dictionary, yielding empirical gains of **3–15%** over standard LZW on Turkish corpora.

2. **Three-Class MLP Algorithm Selector:** A feedforward neural network with architecture $11 \rightarrow 32 \rightarrow 16 \rightarrow 8 \rightarrow 3$ trained on 2,357 oracle-labeled instances. The model achieves $95.2\%$ hold-out accuracy and $91.7\% \pm 1.8\%$ under stratified 5-fold cross-validation.

3. **No-Regret Hybrid Manager:** A meta-algorithmic layer combining the MLP prediction with a BWT post-check, guaranteeing performance no worse than standard Huffman:
$$L_{\text{hybrid}}(x) \leq L_{\text{Huffman}}(x), \quad \forall x \in \Sigma^*. \tag{2}$$

4. **Complete bzip2 Pipeline:** Full implementation of the BWT $\rightarrow$ MTF [4], [9] $\rightarrow$ RLE $\rightarrow$ Huffman cascade with blockwise processing for unbounded input length.

5. **Unicode-Aware LZW:** Dynamic alphabet extension supporting Turkish-specific characters (`ş, ğ, ü, ö, ç, ı`) beyond the standard 0–255 ASCII range.

6. **Empirical Validation:** A test suite of **100 unit tests** covering edge cases, Turkish characters, scaling behavior, and cross-algorithm consistency.

7. **Open-Source Release:** Full reproducibility through public release of code, data, trained models, and AI interaction logs.

### C. Paper Organization

Section II surveys related work. Section III formalizes the system architecture. Section IV details the algorithmic components. Section V describes the AI integrations. Section VI presents the experimental methodology and results. Section VII discusses limitations and future directions. Section VIII concludes.

---

## II. RELATED WORK

### A. Classical Lossless Compression

Huffman's optimal prefix coding [1] established the foundation of frequency-based encoding, with the well-known bound

$$H(X) \leq L_{\text{Huffman}} < H(X) + 1 \tag{3}$$

provable via Kraft's inequality [10]. Welch [3] extended Lempel–Ziv [11] with a dictionary-based scheme that eliminates the need for explicit dictionary transmission. The Burrows–Wheeler Transform [4] introduced a reversible permutation that clusters contextually similar symbols, enabling efficient downstream compression via Move-to-Front transformation [9] and Huffman coding—the cornerstone of bzip2 [4].

### B. Algorithm Selection Problem

Rice's seminal formulation [5] established the theoretical underpinning for selecting algorithms based on instance features. Kotthoff's survey [7] catalogs machine learning–based selectors that have demonstrated superior performance on combinatorial search problems. Our work extends this paradigm to lossless text compression, a domain where—to our knowledge—no comprehensive ML-based selector has been previously deployed for Turkish text.

### C. Neural Compression

Recent neural compression research has explored two complementary directions:

1. **End-to-end neural coding** (e.g., NNCP [12], DeepZip [13]): A language model directly outputs symbol probabilities $P(c_i \mid c_{1:i-1})$, coupled with arithmetic coding to approach the conditional entropy
$$H(X) = \lim_{n \to \infty} \frac{1}{n} H(X_1, X_2, \ldots, X_n). \tag{4}$$

2. **Hybrid architectures**: Classical algorithms augmented by learned components. Our work falls within this category, providing a lightweight and interpretable hybrid suitable for resource-constrained settings.

### D. Information Theory of Natural Language

Shannon [8] empirically estimated English entropy at approximately 1.3 bits/character using human predictors. This bound, attainable only under contextual modeling, motivates our trigram-based contextual entropy analysis in Section VI-D.

---

## III. SYSTEM ARCHITECTURE

### A. Pipeline Overview

The proposed system comprises four logically distinct layers:

1. **Input Layer**: Accepts UTF-8 encoded text with arbitrary length, supporting Turkish Unicode characters.

2. **Algorithm Layer**: Implements three classical pipelines—Huffman, LZW, and BWT+MTF+RLE+Huffman—each augmented with Turkish character support.

3. **AI Layer**: Provides (a) MLP-based algorithm selection over hand-crafted features, and (b) LLM-based dictionary synthesis for LZW initialization.

4. **Hybrid Manager**: Combines selector predictions with a guard mechanism (BWT post-check) to ensure no-regret guarantees.

The system is modularized into ten core Python modules: `huffman.py`, `lzw.py`, `bwt.py`, `nn_selector.py`, `hybrid.py`, `ai_engine.py`, `entropy.py`, `next_token.py`, `benchmark.py`, and `ui_helpers.py`.

### B. Architectural Diagram

```
┌─────────────────────────────────────────────────┐
│                Interactive UI                   │
│  (Streamlit, 9 tabs, Plotly visualization)      │
└────────┬──────────┬──────────────┬──────────────┘
         │          │              │
   ┌─────▼────┐ ┌───▼──────┐ ┌─────▼─────┐
   │ Classical│ │ Groq LLM │ │ MLP       │
   │ Codecs   │ │ Service  │ │ Selector  │
   └─────┬────┘ └────┬─────┘ └─────┬─────┘
         │           │             │
   ┌─────▼───────────▼─────────────▼─────┐
   │      Hybrid Decision Manager        │
   │  (NN prediction + BWT post-check)   │
   └─────────────────┬───────────────────┘
                     │
              ┌──────▼──────┐
              │ Binary (.bin)│
              └─────────────┘
```

Fig. 1. System architecture: Three-layer integration of classical codecs, LLM dictionary synthesis, and MLP-based selection coordinated by a no-regret hybrid manager.

---

## IV. CLASSICAL COMPRESSION ALGORITHMS

### A. Huffman Coding

**Definition 1 (Huffman Tree Construction):** Given a probability distribution $\{p(c_i)\}_{i=1}^{|\Sigma|}$ over an alphabet $\Sigma$, the Huffman tree is constructed by repeated merging of the two minimum-frequency nodes in a min-heap until a single root remains. The resulting binary tree induces a prefix-free code $\phi : \Sigma \rightarrow \{0,1\}^*$ via left-edge label 0, right-edge label 1.

**Theorem 1 (Huffman Optimality [1]):** Among all prefix codes, the Huffman code minimizes the expected codeword length
$$L = \sum_{c \in \Sigma} p(c) |\phi(c)|. \tag{5}$$

We implement Huffman with a min-heap (Python `heapq`) in $O(n \log n)$ time, where $n = |\Sigma|$. Turkish Unicode characters are inherently supported via Python's native string handling.

### B. LZW with Unicode Support

The standard LZW algorithm [3] initializes the dictionary $D_0 = \{(c, i) : c = \text{chr}(i), \ 0 \leq i < 256\}$ and proceeds incrementally.

**Algorithm 1 (LZW Encode):**

```
Input: text ∈ Σ*
Output: codes ⊆ ℕ
1: D ← D_0
2: w ← ε
3: for each c ∈ text do
4:    wc ← w · c
5:    if wc ∈ keys(D) then
6:       w ← wc
7:    else
8:       output D[w]
9:       D[wc] ← |D|
10:      w ← c
11: end for
12: output D[w]
```

**Turkish Character Problem:** The standard initialization $D_0$ contains only single-byte ASCII characters. Turkish characters such as `ş` (U+015F) have code points exceeding 255, causing `KeyError` exceptions during dictionary lookup.

**Proposed Solution:** We augment $D_0$ dynamically with all unique characters appearing in the input:
$$D_0' = D_0 \cup \{(c, 256 + k) : c \in \text{unique}(x) \setminus D_0\} \tag{6}$$

where the index $k$ enumerates novel characters. This approach aligns with Salomon's universal principle [14, §6.13]: *"The dictionary should be initialized with all possible single symbols of the source alphabet."*

### C. BWT + MTF + RLE + Huffman Pipeline

The Burrows–Wheeler Transform [4] applies a reversible permutation that exploits contextual redundancy.

**Algorithm 2 (BWT Encode):**

```
Input: text T ∈ Σ^n
Output: (L, idx) where L ∈ Σ^n, idx ∈ ℕ
1: Append sentinel character: T' ← T · $
2: Construct all n+1 cyclic rotations of T'
3: Sort rotations lexicographically into matrix M
4: L ← last column of M
5: idx ← row index of T' in M
6: return (L, idx)
```

The Move-to-Front transform [9] converts contextual clusters into small-integer sequences:

**Algorithm 3 (MTF Encode):**

```
Input: L ∈ Σ^n
Output: indices ∈ ℕ^n, alphabet ∈ Σ^|Σ|
1: A ← sorted list of unique symbols in L
2: indices ← []
3: for each c ∈ L do
4:    i ← index of c in A
5:    indices.append(i)
6:    Move c to the front of A
7: end for
8: return (indices, A)
```

The output is dominated by small integers, enabling efficient downstream Run-Length Encoding and Huffman compression. This three-stage pipeline corresponds to the canonical bzip2 architecture [14, §8.5].

### D. Block-wise BWT for Unbounded Inputs

The BWT's $O(n^2 \log n)$ complexity for naive suffix array construction limits practical block size. We adopt the bzip2 block-wise strategy [14, §8.5]:

$$\text{BWT}^*(T) = \{(\text{BWT}(T_i), \text{idx}_i)\}_{i=1}^{\lceil |T|/B \rceil} \tag{7}$$

where $T_i$ denotes the $i$-th block of size $B = 8000$. This guarantees lossless reconstruction for arbitrary input lengths while maintaining tractable per-block complexity.

---

## V. ARTIFICIAL INTELLIGENCE INTEGRATION

### A. LLM-Driven Dictionary Synthesis

**Motivation:** Standard LZW initialization with single-character entries provides no source-adaptive prior. We hypothesize that domain-specific lexical patterns, when embedded as multi-byte initialization entries, accelerate pattern matching and reduce code stream length.

**Architectural Component:** A Large Language Model (LLaMA-3.3-70B accessed via the Groq Cloud API) serves as a domain-knowledge oracle. Given a sample $\tilde{T}$ of the source text, the model is prompted to produce a ranked list of $k$ frequent lexical units suitable for LZW dictionary inclusion.

**Prompt Engineering:** The system instruction is

```
System: You are a data compression expert. For the given text 
        type, predict the MOST FREQUENT words/expressions to 
        accelerate LZW pattern matching. Return JSON.
User:   Suggest k frequent words/expressions for LZW dictionary 
        initialization for this Turkish text.
        Sample: """{T̃[:300]}"""
        Format: ["word1", "word2", ...]
```

**Robust JSON Parsing:** Empirically, LLM outputs may include markdown code blocks or trailing text. We implement a robust parser that extracts the first valid JSON array via balanced bracket matching:

```python
def _parse_json(raw):
    raw = raw.replace("```json", "").replace("```", "")
    s, e = raw.find("["), raw.rfind("]")
    if s != -1 and e != -1:
        return json.loads(raw[s:e+1])
    raise ValueError("No valid JSON array found")
```

**LZW Dictionary Augmentation:** The synthesized words $W = \{w_1, \ldots, w_k\}$ are inserted into the LZW initialization dictionary:

$$D_0'' = D_0' \cup \{(w_j, 256 + |\text{Turkish chars}| + j)\}_{j=1}^{k} \tag{8}$$

### B. MLP-Based Algorithm Selector

**Feature Engineering:** We extract an 11-dimensional feature vector $\phi(T) \in \mathbb{R}^{11}$ from input text $T$:

| Index | Feature | Definition |
|-------|---------|------------|
| $\phi_1$ | Shannon entropy | $H(T) = -\sum p(c) \log_2 p(c)$ |
| $\phi_2$ | Unique character ratio | $|\Sigma_T| / |T|$ |
| $\phi_3$ | Top-3 character mass | $\sum_{i=1}^{3} p(c_{(i)})$ |
| $\phi_4$ | Whitespace ratio | $p(' ')$ |
| $\phi_5$ | Turkish character ratio | $\sum_{c \in \Sigma_{\text{TR}}} p(c)$ |
| $\phi_6$ | Mean run-length | $\bar{r}(T)$ |
| $\phi_7$ | Digit ratio | $\sum_{c \in \{0..9\}} p(c)$ |
| $\phi_8$ | Uppercase ratio | $p(\text{upper})$ |
| $\phi_9$ | Bigram entropy | $H_2(T)$ |
| $\phi_{10}$ | Max run-length ratio | $\max(r)/|T|$ |
| $\phi_{11}$ | $\log_2 |\Sigma_T|$ | Log-alphabet size |

**Network Architecture:** We employ a feedforward MLP with three hidden layers:

$$f_\theta : \mathbb{R}^{11} \xrightarrow{W_1, \text{ReLU}} \mathbb{R}^{32} \xrightarrow{W_2, \text{ReLU}} \mathbb{R}^{16} \xrightarrow{W_3, \text{ReLU}} \mathbb{R}^{8} \xrightarrow{W_4, \text{Softmax}} \Delta^{3} \tag{9}$$

where $\Delta^3$ is the 2-simplex (output probability distribution over $\{H, L, B\}$).

**Training Objective:** Cross-entropy loss with L2 regularization:

$$\mathcal{L}(\theta) = -\frac{1}{N} \sum_{i=1}^{N} \log f_\theta(y_i \mid \phi(T_i)) + \frac{\alpha}{2} \|\theta\|_2^2 \tag{10}$$

with $\alpha = 10^{-3}$. Optimization is performed via Adam with early stopping on a 15% validation split.

**Oracle Labeling Strategy:** Each training instance $T_i$ is labeled with the algorithm yielding the minimum bit count:

$$y_i = \arg\min_{a \in \{H, L, B\}} \text{bits}(a, T_i) \tag{11}$$

This *oracle* labeling [7] enables supervised learning of the optimal selection function.

### C. No-Regret Hybrid Manager

**Theorem 2 (No-Regret Guarantee):** Let $\mathcal{S}(T)$ denote the Smart Hybrid output and $L_H(T)$ the standard Huffman codeword length. Then:

$$L_\mathcal{S}(T) \leq L_H(T), \quad \forall T \in \Sigma^*. \tag{12}$$

**Proof:** By construction, the hybrid manager computes
$$L_\mathcal{S}(T) = \min(L_{f_\theta(T)}(T), L_{\text{BWT}}(T), L_H(T)) \tag{13}$$
where $f_\theta(T)$ is the MLP prediction and $L_{\text{BWT}}(T)$ is the BWT pipeline output (post-check). Since the minimization explicitly includes $L_H(T)$, equation (12) holds trivially. $\blacksquare$

---

## VI. EXPERIMENTAL EVALUATION

### A. Datasets

We evaluate on a heterogeneous corpus comprising:

| Dataset | Size (chars) | Type |
|---------|--------------|------|
| `large_turkish.txt` | 64,240 | Natural Turkish prose |
| `diverse_corpus.txt` | 37,853 | Mixed: Turkish + DNA + logs + code |
| `turkce_dogal.txt` | 5,260 | Curated Turkish test set |
| `sample.txt` | 7,400 | Highly repetitive Turkish |
| **Synthetic** | 14 categories | Augmentation (DNA, JSON, etc.) |

**Total training instances:** 2,357 (oracle-labeled).

### B. Cross-Validation and Hold-Out Performance

We employ stratified 5-fold cross-validation with a 20% hold-out set.

**Table I.** MLP Classification Performance.

| Metric | Value |
|--------|-------|
| Hold-out accuracy | $0.952$ |
| 5-fold CV mean | $0.917$ |
| 5-fold CV std. deviation | $\pm 0.018$ |
| Macro-F1 (hold-out) | $0.945$ |
| LZW recall | $1.00$ (small class) |

The narrow standard deviation ($\sigma = 0.018$) under cross-validation indicates strong generalization with minimal overfitting—consistent with the regularization and early-stopping regime described in Section V-B.

### C. Compression Performance Across Data Types

**Table II.** Compression ratio (% reduction) comparison: Standard Huffman vs. Smart Hybrid.

| Data Type | $|T|$ (chars) | Std. Huffman | Smart Hybrid | $\Delta$ (pp) |
|-----------|---------------|--------------|--------------|---------------|
| Repetitive (ABC×100) | 300 | 78.2% | 96.9% | +18.7 |
| DNA (ATCG) | 480 | 73.8% | 94.1% | +20.3 |
| JSON log | 6,800 | 60.7% | 94.7% | +34.0 |
| Turkish prose | 1,200 | 37.1% | 38.7% | +1.6 |
| Wikipedia Turkish | 20,000 | 39.8% | 58.3% | +18.5 |
| Turkish news | 800 | 42.0% | 54.3% | +12.3 |

Dramatic improvements on structurally repetitive data validate the BWT+MTF+RLE+Huffman pipeline's effectiveness; modest gains on natural prose reflect Huffman's near-optimality for high-entropy character sequences.

### D. Comparison with Industry-Standard Compressors

**Table III.** Benchmark on 730-character Turkish academic text. Our `bwt_rle_huffman` implementation is compared against gzip, zlib, bzip2, and lzma at maximum compression level.

| Compressor | Size (bytes) | Reduction | Time (ms) |
|------------|--------------|-----------|-----------|
| gzip -9 | 95 | 87.0% | 0.06 |
| zlib -9 | 91 | 87.5% | 0.01 |
| bzip2 -9 | 148 | 79.7% | 0.80 |
| lzma -9 | 124 | 83.0% | 6.79 |
| **`bwt_rle_huffman`** | **94** | **87.1%** | 0.56 |

Our BWT-based implementation surpasses bzip2 by **36.5%** on this benchmark, primarily due to lower per-block overhead in short texts. While gzip and zlib remain marginally superior due to mature LZ77+Huffman engineering, our system demonstrates competitive performance.

### E. Contextual Entropy Analysis (n-gram)

To quantify the gap between iid Shannon entropy and contextual entropy, we evaluate unigram, bigram, and trigram entropies on the Turkish corpus:

**Table IV.** Contextual entropy estimates on Turkish text.

| Model | Bits/character | Reduction from raw (8 bpc) |
|-------|----------------|----------------------------|
| Raw UTF-8 | 8.00 | – |
| Unigram (iid) | 4.68 | 41.5% |
| Bigram (Markov-1) | 3.58 | 55.3% |
| Trigram (Markov-2) | 2.74 | 65.8% |

The substantial gap between iid (4.68 bpc) and trigram (2.74 bpc) entropy motivates future investigation into context-aware coding (e.g., arithmetic coding with n-gram models). Shannon's classical estimate for English (~1.3 bpc) [8] suggests further improvement is achievable through deeper conditional modeling.

### F. Feature Importance Analysis

We compute permutation feature importance [15] to quantify the relative contribution of each input feature to MLP predictions:

**Table V.** Permutation feature importance.

| Rank | Feature | Importance |
|------|---------|------------|
| 1 | Unique character ratio | 0.247 |
| 2 | Bigram entropy | 0.183 |
| 3 | $\log_2 |\Sigma_T|$ | 0.096 |
| 4 | Mean run-length | 0.082 |
| 5 | Turkish character ratio | 0.058 |
| 6 | Shannon entropy | 0.042 |

The dominance of unique character ratio and bigram entropy is consistent with established compression theory: low alphabet diversity and high lexical repetition strongly indicate suitability for BWT-based or dictionary-based compression.

### G. Unit Test Coverage

The test suite comprises 100 unit tests organized as follows:

- **Reversibility** (72 tests): Encoding/decoding round-trip identity for Huffman, LZW, BWT, MTF across edge cases (empty strings, single characters, Turkish Unicode, repetitive patterns).
- **Scaling** (14 tests): Linear behavior of Huffman; verification of block-wise BWT correctness on inputs exceeding the block size $B$.
- **Performance** (11 tests): Lower-bound compression ratios on representative inputs.
- **Block BWT** (2 tests): Verification on 11,400-character inputs spanning two blocks.
- **Cross-algorithm consistency** (1 test): Independence of decoded outputs across algorithm choice.

All 100 tests pass in 0.11 seconds.

---

## VII. DISCUSSION

### A. Strengths

1. **Theoretical Grounding.** Each algorithmic component is derived from established literature (Shannon [6], Huffman [1], Welch [3], Burrows–Wheeler [4]) and verified against standard texts (Sayood [16], Cover & Thomas [10], Salomon [14]).

2. **No-Regret Guarantee.** Theorem 2 ensures that, despite imperfect MLP predictions, the Smart Hybrid never underperforms standard Huffman.

3. **Unicode-Awareness.** The system handles Turkish-specific characters seamlessly, a non-trivial extension of standard implementations that often assume ASCII-only inputs.

4. **Reproducibility.** Open-source release of code, data, models, and AI interaction logs supports independent verification and extension.

### B. Limitations

1. **iid Entropy Bound.** Our Shannon-based entropy estimates assume character independence. Real linguistic entropy is substantially lower (Section VI-E), suggesting that contextual coders (arithmetic coding + n-gram models) could yield further gains.

2. **MLP Operates at Selection Level.** The neural component performs *algorithm selection* rather than direct symbol-level prediction. End-to-end neural compressors [12], [13] operate at the symbol level and achieve sub-1-bpc compression on English, but at significantly higher computational cost.

3. **BWT Block Size.** The 8,000-character block size is a pragmatic compromise between compression quality and the $O(n^2 \log n)$ suffix array construction cost. bzip2's typical 100KB–900KB blocks could be adopted with more efficient suffix array algorithms (e.g., DC3 [17]) but were deemed unnecessary for our evaluation scope.

4. **LLM API Dependency.** The dictionary synthesis component requires external API access (Groq Cloud). The remaining system components operate fully offline.

### C. Future Work

1. **Neural Arithmetic Coding.** Integration of a Turkish-pretrained LLM with arithmetic coding to approximate the conditional entropy bound (4).

2. **Multimodal Extension.** Application of similar AI-augmented hybrid frameworks to image (via DCT [18]) and audio compression.

3. **Block Size Adaptivity.** Dynamic selection of BWT block size based on entropy estimates, potentially exploiting the trade-off between compression ratio and decode latency.

4. **Comparative Evaluation.** Benchmarking against modern compressors (Brotli, Zstandard) over a broader corpus including diverse natural languages.

---

## VIII. CONCLUSION

We have presented a hybrid lossless text compression framework that combines classical algorithms with two AI integrations: a multilayer perceptron for adaptive algorithm selection and a Large Language Model for source-adaptive LZW dictionary synthesis. Empirical evaluation demonstrates substantial improvements on structurally repetitive data (up to 85.9% reduction over standard Huffman) and competitive performance with industry-standard compressors (surpassing bzip2 by 36.5% on short Turkish text). The MLP attains 95.2% hold-out accuracy with strong cross-validation consistency, and a no-regret guarantee ensures performance no worse than standard Huffman in all scenarios. Full reproducibility is supported through public release of code, models, and detailed AI interaction logs.

This work demonstrates that lightweight AI integration—operating at the level of algorithm selection rather than symbol prediction—can yield meaningful compression gains in low-resource settings while maintaining strict performance guarantees. Future work will focus on closing the gap to neural compression bounds via context-aware coding.

---

## ACKNOWLEDGMENT

The author thanks the Yıldız Technical University Department of Computer Engineering for institutional support during the project. The author is grateful to Groq Inc. for providing free access to the LLaMA-3.3-70B model via the Groq Cloud API.

---

## REFERENCES

[1] D. A. Huffman, "A method for the construction of minimum-redundancy codes," *Proc. IRE*, vol. 40, no. 9, pp. 1098–1101, Sep. 1952.

[2] J. Ziv and A. Lempel, "A universal algorithm for sequential data compression," *IEEE Trans. Inf. Theory*, vol. IT-23, no. 3, pp. 337–343, May 1977.

[3] T. A. Welch, "A technique for high-performance data compression," *Computer*, vol. 17, no. 6, pp. 8–19, Jun. 1984.

[4] M. Burrows and D. J. Wheeler, "A block-sorting lossless data compression algorithm," Digital Equipment Corporation, SRC Research Report 124, May 1994.

[5] J. R. Rice, "The algorithm selection problem," *Advances in Computers*, vol. 15, pp. 65–118, 1976.

[6] C. E. Shannon, "A mathematical theory of communication," *Bell Syst. Tech. J.*, vol. 27, no. 3, pp. 379–423, Jul. 1948.

[7] L. Kotthoff, "Algorithm selection for combinatorial search problems: A survey," *AI Magazine*, vol. 35, no. 3, pp. 48–60, Sep. 2014.

[8] C. E. Shannon, "Prediction and entropy of printed English," *Bell Syst. Tech. J.*, vol. 30, no. 1, pp. 50–64, Jan. 1951.

[9] J. L. Bentley, D. D. Sleator, R. E. Tarjan, and V. K. Wei, "A locally adaptive data compression scheme," *Commun. ACM*, vol. 29, no. 4, pp. 320–330, Apr. 1986.

[10] T. M. Cover and J. A. Thomas, *Elements of Information Theory*, 2nd ed. Hoboken, NJ, USA: Wiley-Interscience, 2006.

[11] J. Ziv and A. Lempel, "Compression of individual sequences via variable-rate coding," *IEEE Trans. Inf. Theory*, vol. IT-24, no. 5, pp. 530–536, Sep. 1978.

[12] F. Bellard, "NNCP: Lossless data compression with neural networks," 2019. [Online]. Available: https://bellard.org/nncp/

[13] M. Goyal, K. Tatwawadi, S. Chandak, and I. Ochoa, "DeepZip: Lossless data compression using recurrent neural networks," in *Proc. Data Compression Conf. (DCC)*, 2019, pp. 575–584.

[14] D. Salomon, *Data Compression: The Complete Reference*, 4th ed. London, U.K.: Springer, 2007.

[15] L. Breiman, "Random forests," *Machine Learning*, vol. 45, no. 1, pp. 5–32, Oct. 2001.

[16] K. Sayood, *Introduction to Data Compression*, 4th ed. Burlington, MA, USA: Morgan Kaufmann, 2017.

[17] J. Kärkkäinen and P. Sanders, "Simple linear work suffix array construction," in *Proc. ICALP*, 2003, pp. 943–955.

[18] N. Ahmed, T. Natarajan, and K. R. Rao, "Discrete cosine transform," *IEEE Trans. Comput.*, vol. C-23, no. 1, pp. 90–93, Jan. 1974.

[19] D. J. C. MacKay, *Information Theory, Inference, and Learning Algorithms*. Cambridge, U.K.: Cambridge Univ. Press, 2003.

[20] C. M. Bishop, *Pattern Recognition and Machine Learning*. New York, NY, USA: Springer, 2006.

[21] I. Goodfellow, Y. Bengio, and A. Courville, *Deep Learning*. Cambridge, MA, USA: MIT Press, 2016.

[22] T. Hastie, R. Tibshirani, and J. Friedman, *The Elements of Statistical Learning*, 2nd ed. New York, NY, USA: Springer, 2009.

[23] F. Pedregosa *et al.*, "Scikit-learn: Machine learning in Python," *J. Mach. Learn. Res.*, vol. 12, pp. 2825–2830, Oct. 2011.

[24] S. Kullback and R. A. Leibler, "On information and sufficiency," *Ann. Math. Statist.*, vol. 22, no. 1, pp. 79–86, Mar. 1951.

---

## APPENDIX

### A. Reproducibility Materials

The complete source code, datasets, trained MLP model (`nn_model.pkl`), AI interaction log (`ai_diary.json` containing 42 entries with 12 explicit literature cross-references), and Docker container specification are publicly available at:

- **GitHub repository:** https://github.com/betullarslan-cpu/ai-veri-sikistirma
- **HuggingFace Spaces (interactive demonstration):** https://huggingface.co/spaces/tien23/ai-veri-sikistirma

License: MIT.

### B. Computational Environment

Experiments were conducted on a MacBook with Python 3.11. The MLP training requires approximately 10 seconds. Compression benchmarks utilize Python's standard library implementations of gzip, zlib, bzip2, and lzma.

### C. Code Organization

```
project/
├── app.py                # Streamlit interface (9 tabs)
├── core/
│   ├── huffman.py        # Huffman encode/decode
│   ├── lzw.py            # LZW with Turkish Unicode support
│   ├── bwt.py            # BWT + MTF + RLE + Huffman + blockwise
│   ├── nn_selector.py    # MLP + feature importance + confusion matrix
│   ├── hybrid.py         # No-regret hybrid manager
│   ├── ai_engine.py      # Groq LLM integration
│   ├── entropy.py        # Shannon entropy
│   ├── next_token.py     # n-gram contextual entropy
│   ├── benchmark.py      # gzip/bzip2/zlib/lzma comparison
│   └── ui_helpers.py     # Streamlit visualization helpers
├── data/                 # 4 corpora (Turkish + synthetic)
├── tests/                # 100 unit tests
├── ai_diary.json         # 42 AI interaction entries
└── nn_model.pkl          # Trained MLP weights
```

---

*Manuscript prepared June 8, 2026.*
*© 2026 Betül Arslan. All rights reserved.*
