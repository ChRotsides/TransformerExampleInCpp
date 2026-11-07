// Embedding.h
#ifndef EMBEDDING_H
#define EMBEDDING_H

#include <string>
#include <vector>

#include "Common/Types.h"

// Lookup table mapping token ids to dense vectors (vocab_size x embed_dim).
class Embedding {
public:
    Embedding(int vocab_size, int embed_dim, unsigned seed = 42);

    Vector getEmbedding(int token_id) const;
    Matrix tokenToEmbeddings(const std::vector<int>& tokens) const;

    void saveEmbeddingMatrix(const std::string& filename) const;
    void loadEmbeddingMatrix(const std::string& filename);
    void printEmbeddingMatrix() const;

    int vocabSize() const { return vocab_size; }
    int embedDim() const { return embed_dim; }

private:
    int vocab_size;
    int embed_dim;
    Matrix embedding_matrix;
};

#endif // EMBEDDING_H
