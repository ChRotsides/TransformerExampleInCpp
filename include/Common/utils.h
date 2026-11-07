// utils.h
#ifndef UTILS_H
#define UTILS_H

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <vector>

#include "Common/Types.h"

class Utils {
public:
    static bool containsNaN(const std::vector<double>& vec) {
        return std::any_of(vec.begin(), vec.end(), [](double x) { return std::isnan(x); });
    }

    static bool allFinite(const Matrix& m) {
        for (const auto& row : m)
            for (double x : row)
                if (!std::isfinite(x)) return false;
        return true;
    }

    // Numerically stable softmax: the maximum is subtracted before exponentiating,
    // so large scores cannot overflow. Entries equal to -infinity (masked positions)
    // get probability 0 as long as at least one entry is finite.
    static std::vector<double> softmax(const std::vector<double>& scores) {
        if (scores.empty()) throw std::invalid_argument("softmax: empty input");
        const double maxScore = *std::max_element(scores.begin(), scores.end());
        if (std::isinf(maxScore) && maxScore < 0)
            throw std::invalid_argument("softmax: all entries are -infinity");
        std::vector<double> out(scores.size());
        double sum = 0.0;
        for (size_t i = 0; i < scores.size(); ++i) {
            out[i] = std::exp(scores[i] - maxScore);
            sum += out[i];
        }
        for (double& v : out) v /= sum;
        return out;
    }
};

#endif // UTILS_H
