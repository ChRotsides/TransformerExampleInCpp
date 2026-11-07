// Tokenizer.h
#ifndef SELF_TOKENIZER_H
#define SELF_TOKENIZER_H

#include <string>
#include <unordered_map>
#include <vector>

#include "Common/Types.h"

// Whitespace word tokenizer. Ids are assigned in order of first appearance.
class Tokenizer {
public:
    Tokenizer() = default;

    // Splits on whitespace, adding unseen words to the vocabulary.
    std::vector<int> tokenize(const std::string& text);
    // Joins words with single spaces; unknown ids are rendered as "<unk>".
    std::string detokenize(const std::vector<int>& tokens) const;

    int getVocabSize() const { return static_cast<int>(word_to_token_id.size()); }
    // Returns the id of an existing word, or -1.
    int tokenId(const std::string& word) const;
    Vector oneHotEncode(int token_id) const;

    void printWordToTokenIdMap() const;
    void saveTokenMap(const std::string& filename) const;
    void loadTokenMap(const std::string& filename);

private:
    std::unordered_map<std::string, int> word_to_token_id;
    std::unordered_map<int, std::string> token_id_to_word;
};

#endif // SELF_TOKENIZER_H
