// demo.cpp - runs one forward pass of the untrained encoder-decoder transformer.
#include <algorithm>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

#include "Common/SelfAttention.h"
#include "Common/Tokenizer.h"
#include "Transformer.h"

namespace {
int argmax(const Vector& v) {
    return static_cast<int>(std::distance(v.begin(), std::max_element(v.begin(), v.end())));
}
} // namespace

int main() {
    const std::string source_text = "the cat sat on the mat";
    const std::string target_text = "<start> le chat est assis sur le tapis";

    Tokenizer tok;
    const std::vector<int> source = tok.tokenize(source_text);
    const std::vector<int> target = tok.tokenize(target_text);

    const int vocab_size = tok.getVocabSize();
    const int model_dim = 16, num_layers = 2, num_heads = 4, max_seq_len = 64;
    const unsigned seed = 42;

    std::cout << "source tokens : " << source_text << "  (" << source.size() << " tokens)\n";
    std::cout << "target tokens : " << target_text << "  (" << target.size() << " tokens)\n";
    std::cout << "vocab size    : " << vocab_size << "\n";
    std::cout << "config        : d_model=" << model_dim << " layers=" << num_layers
              << " heads=" << num_heads << " seed=" << seed << "\n\n";

    Transformer model(vocab_size, model_dim, num_layers, num_heads, max_seq_len, seed);
    const Matrix memory = model.encode(source);
    const Matrix probs = model.forward(source, target);

    std::cout << "encoder output : " << memory.size() << " x " << memory[0].size() << "\n";
    std::cout << "output probs   : " << probs.size() << " x " << probs[0].size() << "\n\n";

    std::cout << "Per-position next-token distribution (random weights, no training):\n";
    std::cout << std::fixed << std::setprecision(4);
    for (size_t i = 0; i < probs.size(); ++i) {
        double sum = 0.0;
        for (double p : probs[i]) sum += p;
        const int best = argmax(probs[i]);
        std::cout << "  pos " << i << "  input=" << std::setw(8) << std::left << tok.detokenize({target[i]})
                  << std::right << "  argmax=" << std::setw(8) << std::left << tok.detokenize({best})
                  << std::right << "  p=" << probs[i][best] << "  row_sum=" << sum << "\n";
    }

    // Show the attention pattern of a standalone causal self-attention layer.
    std::cout << "\nCausal self-attention weights (head 0) over the target embeddings:\n";
    SelfAttention attn(model_dim, num_heads, seed);
    std::vector<Matrix> weights;
    Embedding emb(vocab_size, model_dim, seed);
    attn.forward(emb.tokenToEmbeddings(target), /*causal=*/true, &weights);
    for (const auto& row : weights[0]) {
        std::cout << " ";
        for (double w : row) std::cout << " " << std::setw(6) << w;
        std::cout << "\n";
    }
    return 0;
}
