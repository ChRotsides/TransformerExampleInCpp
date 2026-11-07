// Embedding.cpp
#include "Common/Embedding.h"

#include <fstream>
#include <iomanip>
#include <iostream>
#include <random>
#include <sstream>
#include <stdexcept>

Embedding::Embedding(int vocab_size, int embed_dim, unsigned seed)
    : vocab_size(vocab_size), embed_dim(embed_dim) {
    if (vocab_size <= 0 || embed_dim <= 0)
        throw std::invalid_argument("Embedding: vocab_size and embed_dim must be positive");
    std::mt19937 gen(seed);
    std::normal_distribution<double> dist(0.0, 1.0);
    embedding_matrix.assign(vocab_size, Vector(embed_dim));
    for (auto& row : embedding_matrix)
        for (auto& x : row) x = dist(gen);
}

void Embedding::printEmbeddingMatrix() const {
    std::cout << "Embedding matrix:" << std::endl;
    for (const auto& row : embedding_matrix) {
        for (double x : row) std::cout << std::fixed << std::setprecision(2) << std::setw(6) << x << " ";
        std::cout << std::endl;
    }
}

void Embedding::saveEmbeddingMatrix(const std::string& filename) const {
    std::ofstream file(filename);
    if (!file.is_open()) throw std::runtime_error("Embedding: cannot open for writing: " + filename);
    file << std::setprecision(17);
    for (const auto& row : embedding_matrix) {
        for (double x : row) file << x << ' ';
        file << '\n';
    }
}

void Embedding::loadEmbeddingMatrix(const std::string& filename) {
    std::ifstream file(filename);
    if (!file.is_open()) throw std::runtime_error("Embedding: cannot open for reading: " + filename);

    Matrix loaded(vocab_size, Vector(embed_dim, 0.0));
    std::string line;
    int row = 0;
    while (row < vocab_size && std::getline(file, line)) {
        std::istringstream iss(line);
        int col = 0;
        double value;
        while (col < embed_dim && iss >> value) loaded[row][col++] = value;
        if (col != embed_dim)
            throw std::runtime_error("Embedding: row " + std::to_string(row + 1) + " has too few values");
        ++row;
    }
    if (row != vocab_size) throw std::runtime_error("Embedding: file has too few rows");
    embedding_matrix = std::move(loaded);
}

Vector Embedding::getEmbedding(int token_id) const {
    if (token_id < 0 || token_id >= vocab_size)
        throw std::out_of_range("Embedding: token id out of range");
    return embedding_matrix[token_id];
}

Matrix Embedding::tokenToEmbeddings(const std::vector<int>& tokens) const {
    Matrix out;
    out.reserve(tokens.size());
    for (int t : tokens) out.push_back(getEmbedding(t));
    return out;
}
