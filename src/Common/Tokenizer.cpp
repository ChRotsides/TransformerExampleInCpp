// Tokenizer.cpp
#include "Common/Tokenizer.h"

#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>

std::vector<int> Tokenizer::tokenize(const std::string& text) {
    std::vector<int> tokens;
    std::istringstream iss(text);
    std::string word;
    while (iss >> word) {
        auto it = word_to_token_id.find(word);
        if (it == word_to_token_id.end()) {
            const int id = static_cast<int>(word_to_token_id.size());
            it = word_to_token_id.emplace(word, id).first;
            token_id_to_word[id] = word;
        }
        tokens.push_back(it->second);
    }
    return tokens;
}

int Tokenizer::tokenId(const std::string& word) const {
    auto it = word_to_token_id.find(word);
    return it == word_to_token_id.end() ? -1 : it->second;
}

std::string Tokenizer::detokenize(const std::vector<int>& tokens) const {
    std::string text;
    for (size_t i = 0; i < tokens.size(); ++i) {
        auto it = token_id_to_word.find(tokens[i]);
        if (i > 0) text += ' ';
        text += (it == token_id_to_word.end()) ? "<unk>" : it->second;
    }
    return text;
}

Vector Tokenizer::oneHotEncode(int token_id) const {
    if (token_id < 0 || token_id >= getVocabSize())
        throw std::out_of_range("Tokenizer: token id out of range");
    Vector v(getVocabSize(), 0.0);
    v[token_id] = 1.0;
    return v;
}

void Tokenizer::printWordToTokenIdMap() const {
    std::cout << "Word to Token ID Map:" << std::endl;
    for (const auto& pair : word_to_token_id) std::cout << pair.first << " -> " << pair.second << std::endl;
}

void Tokenizer::saveTokenMap(const std::string& filename) const {
    std::ofstream file(filename);
    if (!file.is_open()) throw std::runtime_error("Tokenizer: cannot open for writing: " + filename);
    for (const auto& pair : word_to_token_id) file << pair.first << ' ' << pair.second << '\n';
}

void Tokenizer::loadTokenMap(const std::string& filename) {
    std::ifstream file(filename);
    if (!file.is_open()) throw std::runtime_error("Tokenizer: cannot open for reading: " + filename);
    std::string word;
    int token_id;
    while (file >> word >> token_id) {
        word_to_token_id[word] = token_id;
        token_id_to_word[token_id] = word;
    }
}
