// Transformer.h
#ifndef TRANSFORMER_H
#define TRANSFORMER_H

#include <vector>

#include "Common/Embedding.h"
#include "Common/FinalLayer.h"
#include "Common/Types.h"
#include "Decoding/Decoder.h"
#include "Encoding/Encoder.h"
#include "Encoding/PositionalEncoding.h"

// Encoder-decoder transformer wired from the components in this repository.
// Source and target share one embedding table. Weights are random (seeded) and
// there is no training code, so outputs are structurally valid but not meaningful.
class Transformer {
public:
    Transformer(int vocab_size, int model_dim, int num_layers, int num_heads, int max_seq_len,
                unsigned seed = 42);

    // Runs the full forward pass and returns next-token probabilities for every target
    // position: (target_len x vocab_size), each row sums to 1.
    Matrix forward(const std::vector<int>& source_tokens, const std::vector<int>& target_tokens) const;

    // Encoder output only: (source_len x model_dim).
    Matrix encode(const std::vector<int>& source_tokens) const;

private:
    Embedding embedding;
    PositionalEncoding positional;
    Encoder encoder;
    Decoder decoder;
    FinalLayer finalLayer;
};

#endif // TRANSFORMER_H
