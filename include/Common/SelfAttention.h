// SelfAttention.h
#ifndef SELF_ATTENTION_H
#define SELF_ATTENTION_H

#include <vector>

#include "Common/Types.h"

// Multi-head scaled dot-product attention.
//
//   Q = X_q W_q,  K = X_kv W_k,  V = X_kv W_v           (per head: d_head = model_dim / num_heads)
//   head_h = softmax(Q_h K_h^T / sqrt(d_head) [+ causal mask]) V_h
//   output = concat(head_1 .. head_H) W_o
//
// With query == key/value source this is self-attention; with the encoder output as
// key/value source it is encoder-decoder (cross) attention.
class SelfAttention {
public:
    explicit SelfAttention(int model_dim, int num_heads = 1, unsigned seed = 42);

    // Self-attention over a whole sequence (seq_len x model_dim) -> (seq_len x model_dim).
    // If causal is true, position i may only attend to positions <= i.
    // If attention_weights is non-null it receives one (seq_len x seq_len) matrix per head.
    Matrix forward(const Matrix& input, bool causal = false,
                   std::vector<Matrix>* attention_weights = nullptr) const;

    // General form: queries come from `query_input`, keys/values from `memory`.
    // Returns (query_len x model_dim); weights are (query_len x memory_len) per head.
    Matrix attend(const Matrix& query_input, const Matrix& memory, bool causal = false,
                  std::vector<Matrix>* attention_weights = nullptr) const;

    int numHeads() const { return num_heads; }

private:
    int model_dim, num_heads, head_dim;
    Matrix W_q, W_k, W_v, W_o; // each (model_dim x model_dim); applied as x -> W x

    Matrix project(const Matrix& input, const Matrix& weights) const;
};

#endif // SELF_ATTENTION_H
