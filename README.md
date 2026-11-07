# Transformers from Scratch in C++

[![CI](https://github.com/ChRotsides/TransformerExampleInCpp/actions/workflows/ci.yml/badge.svg)](https://github.com/ChRotsides/TransformerExampleInCpp/actions/workflows/ci.yml)

An educational implementation of the building blocks of the Transformer ("Attention Is All You Need", Vaswani et al., 2017) in plain C++17. There is no Eigen, libtorch or any other numerical library: every matrix product, softmax and normalisation is written out by hand in `std::vector<double>` loops, so the code is meant to be read.

This is a **forward-pass-only** implementation. The weights are randomly initialised from a seed and there is no training or backpropagation, so the model produces structurally correct but meaningless predictions. The point is the architecture and the mechanics of each component, which are covered by unit tests.

## What is implemented

| Component | Files | Notes |
|---|---|---|
| Tokenizer | `Common/Tokenizer` | Whitespace word tokenizer, ids by first appearance, save/load of the vocabulary |
| Embedding | `Common/Embedding` | Seeded `N(0, 1)` lookup table, save/load |
| Positional encoding | `Encoding/PositionalEncoding` | Sinusoidal, as in the paper |
| Multi-head attention | `Common/SelfAttention` | Scaled dot-product, `Q/K/V/O` projections, optional causal mask, cross-attention via `attend()` |
| Feed-forward network | `Common/FeedForwardNetwork` | Linear (d to 4d), ReLU, Linear (4d to d) |
| Layer normalisation | `Common/LayerNorm` | Per-feature scale (`gamma`) and shift (`beta`) |
| Encoder | `Encoding/EncoderLayer`, `Encoding/Encoder` | Post-norm: attention + residual + LayerNorm, FFN + residual + LayerNorm |
| Decoder | `Decoding/DecoderLayer`, `Decoding/Decoder` | Masked self-attention, cross-attention over encoder output, FFN, each with residual + LayerNorm |
| Output layer | `Common/FinalLayer` | Linear projection to vocabulary size followed by a stable softmax |
| Full model | `Transformer` | Wires everything together; shared source/target embedding table |

## Architecture

```
source tokens                                   target tokens (shifted right)
     |                                                |
 Embedding + Positional Encoding            Embedding + Positional Encoding
     |                                                |
 +---------------------------+              +----------------------------------+
 | Encoder layer  (x N)      |              | Decoder layer  (x N)             |
 |  self-attention           |              |  masked self-attention           |
 |  add & LayerNorm          |   memory     |  add & LayerNorm                 |
 |  feed-forward             | -----------> |  cross-attention (K, V = memory) |
 |  add & LayerNorm          |              |  add & LayerNorm                 |
 +---------------------------+              |  feed-forward                    |
                                            |  add & LayerNorm                 |
                                            +----------------------------------+
                                                          |
                                              Linear -> softmax over vocabulary
                                                          |
                                         (target_len x vocab_size) probabilities
```

## How attention is computed

For one head with head dimension `d_head = model_dim / num_heads`, given a query sequence `X_q` and a key/value sequence `X_kv` (the same sequence for self-attention, the encoder output for cross-attention):

```
Q = X_q  W_q          (n_q x d_head)
K = X_kv W_k          (n_k x d_head)
V = X_kv W_v          (n_k x d_head)

scores  = Q K^T / sqrt(d_head)            (n_q x n_k)
scores[i][j] = -infinity   if causal and j > i
weights = softmax(scores) along each row  (each row sums to 1)
head    = weights V                       (n_q x d_head)

output  = concat(head_1 ... head_H) W_o   (n_q x model_dim)
```

The softmax subtracts the row maximum before exponentiating, so large scores cannot overflow, and masked (`-infinity`) entries get exactly zero weight.

## Layout

```
include/            headers (Common/, Encoding/, Decoding/, Transformer.h)
src/                implementations
examples/demo.cpp   one forward pass through the full model
tests/              doctest unit tests
third_party/doctest vendored single-header test framework (MIT)
CMakeLists.txt      library + demo + tests, C++17, -Wall -Wextra
```

## Build, run and test

Requires a C++17 compiler and CMake 3.10 or newer.

```bash
cmake -S . -B build
cmake --build build
ctest --test-dir build --output-on-failure   # run the unit tests
./build/transformer_demo                     # run the example
```

Without CMake, a single compiler invocation also works (tested with MinGW g++ on Windows):

```bash
g++ -std=c++17 -O2 -Iinclude examples/demo.cpp src/*/*.cpp src/Transformer.cpp -o transformer_demo
g++ -std=c++17 -O2 -Iinclude -Ithird_party/doctest tests/*.cpp src/*/*.cpp src/Transformer.cpp -o transformer_tests
./transformer_tests
```

### Tests

The suite has 29 test cases (559 assertions). It checks, among other things:

- softmax rows sum to 1, are shift-invariant, stay finite for scores around 1000 and give `-infinity` entries zero probability
- attention output shape, row-stochastic attention weights per head, uniform weights for identical tokens, and the causal mask (no weight on future positions, and earlier outputs unchanged when a later token changes)
- cross-attention with different query and memory lengths
- positional encoding values at known positions (for example `PE(1, 2) = sin(0.01)` for `model_dim = 4`)
- layer norm gives mean 0 and variance 1 (up to epsilon)
- embedding, tokenizer, feed-forward and output layer behaviour, including error cases
- the full `Transformer` forward pass is deterministic for a fixed seed, returns one probability distribution per target position, and predictions at earlier positions ignore later target tokens

It passes on Linux (g++ 13, also under AddressSanitizer and UBSan) and with MinGW g++ on Windows.

## Example output

`./build/transformer_demo` runs a 2-layer, 4-head, `model_dim = 16` model with seed 42 on a toy sentence pair. The weights are random, so the predicted tokens are arbitrary; the output demonstrates shapes, valid probability rows and the causal attention pattern.

```
source tokens : the cat sat on the mat  (6 tokens)
target tokens : <start> le chat est assis sur le tapis  (8 tokens)
vocab size    : 12
config        : d_model=16 layers=2 heads=4 seed=42

encoder output : 6 x 16
output probs   : 8 x 12

Per-position next-token distribution (random weights, no training):
  pos 0  input=<start>   argmax=est       p=0.2711  row_sum=1.0000
  pos 1  input=le        argmax=est       p=0.2480  row_sum=1.0000
  pos 2  input=chat      argmax=est       p=0.2483  row_sum=1.0000
  pos 3  input=est       argmax=est       p=0.2780  row_sum=1.0000
  pos 4  input=assis     argmax=est       p=0.3125  row_sum=1.0000
  pos 5  input=sur       argmax=est       p=0.2847  row_sum=1.0000
  pos 6  input=le        argmax=est       p=0.2594  row_sum=1.0000
  pos 7  input=tapis     argmax=est       p=0.2892  row_sum=1.0000

Causal self-attention weights (head 0) over the target embeddings:
  1.0000 0.0000 0.0000 0.0000 0.0000 0.0000 0.0000 0.0000
  0.1759 0.8241 0.0000 0.0000 0.0000 0.0000 0.0000 0.0000
  0.0157 0.5459 0.4384 0.0000 0.0000 0.0000 0.0000 0.0000
  0.6915 0.0281 0.0577 0.2228 0.0000 0.0000 0.0000 0.0000
  0.3249 0.0743 0.1107 0.1942 0.2959 0.0000 0.0000 0.0000
  0.0666 0.0850 0.0366 0.6118 0.0439 0.1560 0.0000 0.0000
  0.0442 0.2070 0.2008 0.0333 0.1400 0.1676 0.2070 0.0000
  0.0242 0.1846 0.1655 0.0214 0.0402 0.1112 0.1846 0.2683
```

## Limitations and next steps

- No training: there is no loss-driven weight update, backpropagation or optimiser, so the weights stay at their random initial values. Adding backward passes (or a small autograd) is the natural next step.
- No autoregressive generation loop (greedy or sampling) yet; `Transformer::forward` computes teacher-forced probabilities for a given target sequence.
- Single sequence at a time (no batching), no padding mask, no dropout, no weight tying.
- Post-norm layers as in the original paper; pre-norm, GELU and learned positional embeddings are not implemented.
- Naive `std::vector<std::vector<double>>` loops with no SIMD, blocking or threading, so it is far from fast. It is written for clarity.
- Word-level whitespace tokenizer; no subword (BPE) vocabulary.

## License

PolyForm Noncommercial License 1.0.0, see [LICENSE](LICENSE). The code is free to use for noncommercial and research purposes. Commercial use requires a separate license: please contact lhampis@gmail.com.

The vendored test framework in `third_party/doctest` is MIT licensed (see `third_party/doctest/LICENSE.txt`).
