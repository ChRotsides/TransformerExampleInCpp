// Transformer.cpp
#include "Transformer.h"

Transformer::Transformer(int vocab_size, int model_dim, int num_layers, int num_heads, int max_seq_len,
                         unsigned seed)
    : embedding(vocab_size, model_dim, seed),
      positional(max_seq_len, model_dim),
      encoder(num_layers, model_dim, num_heads, seed + 100),
      decoder(num_layers, model_dim, num_heads, seed + 200),
      finalLayer(model_dim, vocab_size, seed + 300) {}

Matrix Transformer::encode(const std::vector<int>& source_tokens) const {
    return encoder.forward(positional.apply(embedding.tokenToEmbeddings(source_tokens)));
}

Matrix Transformer::forward(const std::vector<int>& source_tokens,
                            const std::vector<int>& target_tokens) const {
    const Matrix memory = encode(source_tokens);
    const Matrix target_in = positional.apply(embedding.tokenToEmbeddings(target_tokens));
    return finalLayer.forward(decoder.forward(target_in, memory));
}
