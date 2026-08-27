#pragma once

#include <vector>
#include <concepts>
#include <cstdint>
#include <span>
#include <cmath>

namespace VectorDB {

// может использоваться только типы с плавающей точкой
template<typename T>
concept FloatingPoint = std::floating_point<T>;

// структура для хранения 1 векторного документа
struct VectorItem {
    int32_t id;
    std::vector<float> values;
};

// структура где хранится id и процент схожести
struct SearchResult {
    int32_t id;
    float similarity;
};

class VectorMath {
public:
    // вычисление косинусного сходства 
    // Благодаря флагу -ffast-math компилятор векторизует этот цикл с помощью AVX инструкций
    template<FloatingPoint T>
    static float cosine_similarity(std::span<const T> a, std::span<const T> b) {
        float numerator = 0.0f; //числитель
        float norm_a = 0.0f; // сумма квадратов 1 вектора
        float norm_b = 0.0f;// сумма квадратов 2 вектора

        for (size_t i = 0; i < a.size(); ++i) {
            numerator += a[i] * b[i];
            norm_a += a[i] * a[i];
            norm_b += b[i] * b[i];
        }

        if (norm_a == 0.0f || norm_b == 0.0f) return 0.0f;
        return numerator / (std::sqrt(norm_a) * std::sqrt(norm_b));
    }
};

} 
