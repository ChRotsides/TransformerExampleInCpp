// PositionalEncoding.cpp
#include "Encoding/PositionalEncoding.h"

#include <cmath>
#include <stdexcept>

PositionalEncoding::PositionalEncoding(int max_seq_len, int model_dim)
    : max_seq_len(max_seq_len), model_dim(model_dim) {
    if (max_seq_len <= 0 || model_dim <= 0)
        throw std::invalid_argument("PositionalEncoding: dimensions must be positive");
    encoding_matrix.assign(max_seq_len, Vector(model_dim, 0.0));
    for (int pos = 0; pos < max_seq_len; ++pos) {
        // i is the even feature index (2k), so the exponent 2k / model_dim equals i / model_dim.
        for (int i = 0; i < model_dim; i += 2) {
            const double angle = pos / std::pow(10000.0, static_cast<double>(i) / model_dim);
            encoding_matrix[pos][i] = std::sin(angle);
            if (i + 1 < model_dim) encoding_matrix[pos][i + 1] = std::cos(angle);
        }
    }
}

const Vector& PositionalEncoding::getEncoding(int pos) const {
    if (pos < 0 || pos >= max_seq_len)
        throw std::out_of_range("PositionalEncoding::getEncoding: position out of range");
    return encoding_matrix[pos];
}

Matrix PositionalEncoding::apply(const Matrix& embeddings) const {
    if (static_cast<int>(embeddings.size()) > max_seq_len)
        throw std::out_of_range("PositionalEncoding::apply: sequence longer than max_seq_len");
    Matrix out(embeddings.size(), Vector(model_dim));
    for (size_t i = 0; i < embeddings.size(); ++i) {
        if (static_cast<int>(embeddings[i].size()) != model_dim)
            throw std::invalid_argument("PositionalEncoding::apply: embedding width does not match model_dim");
        for (int j = 0; j < model_dim; ++j) out[i][j] = embeddings[i][j] + encoding_matrix[i][j];
    }
    return out;
}
