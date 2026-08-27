#include "vectordb/vectorizer.hpp"
#include <algorithm>
#include <numeric>
#include <cmath>
#include <fstream>
#include <iostream>

namespace VectorDB {

OnnxVectorizer::OnnxVectorizer(const VectorizerConfig& config) : config_(config) {
    try {
        // создаем окружение ONNX
        env_ = std::make_unique<Ort::Env>(ORT_LOGGING_LEVEL_WARNING, "Vectorizer");
        
        // настройки сессии
        Ort::SessionOptions session_options;
        session_options.SetIntraOpNumThreads(4);  // количество потоков для параллельных вычислений
        session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);  // максимальная оптимизация графа
        
        // загружаем модель
        std::cout << "Загрузка ONNX модели: " << config.model_path << std::endl;
        session_ = std::make_unique<Ort::Session>(*env_, config.model_path.c_str(), session_options);
        
        // получаем имена входов и выходов
        Ort::AllocatorWithDefaultOptions allocator;
        
        // сохраняем имена всех входных узлов
        size_t num_input_nodes = session_->GetInputCount();
        for (size_t i = 0; i < num_input_nodes; i++) {
            auto name = session_->GetInputNameAllocated(i, allocator);
            input_names_.push_back(name.get());
        }
        
        // сохраняем имена всех выходных узлов
        size_t num_output_nodes = session_->GetOutputCount();
        for (size_t i = 0; i < num_output_nodes; i++) {
            auto name = session_->GetOutputNameAllocated(i, allocator);
            output_names_.push_back(name.get());
        }
        
        // создаем информацию о памяти для CPU
        memory_info_ = std::make_unique<Ort::MemoryInfo>(
            Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault)
        );
        
        std::cout << "ONNX модель загружена. Входов: " << num_input_nodes 
                  << ", Выходов: " << num_output_nodes << std::endl;
        std::cout << "Размерность эмбеддинга: " << config.embedding_dim << std::endl;
        
    } catch (const Ort::Exception& e) {
        std::cout << "Ошибка загрузки ONNX модели: " << e.what() << std::endl;
        throw;
    }
}

OnnxVectorizer::~OnnxVectorizer() = default;

std::vector<int64_t> OnnxVectorizer::tokenize(const std::string& text) {
    // простая токенизация по словам с маппингом на ID
    std::vector<int64_t> tokens;
    
    // добавляем CLS токен (101 для BERT) - маркер начала предложения
    tokens.push_back(101);
    
    // разбиваем текст на слова
    std::string current_word;
    for (char c : text) {
        if (c == ' ' || c == '\n' || c == '\t') {
            if (!current_word.empty()) {
                // хешируем слово в ID токена (временное решение)
                int64_t token_id = 1000 + (std::hash<std::string>{}(current_word) % 20000);
                tokens.push_back(token_id);
                current_word.clear();
            }
        } else {
            current_word += std::tolower(c);  // приводим к нижнему регистру
        }
    }
    
    // обрабатываем последнее слово
    if (!current_word.empty()) {
        int64_t token_id = 1000 + (std::hash<std::string>{}(current_word) % 20000);
        tokens.push_back(token_id);
    }
    
    // добавляем SEP токен (102 для BERT) - маркер конца предложения
    tokens.push_back(102);
    
    // обрезаем до max_length, оставляя место для финального SEP токена
    if (tokens.size() > static_cast<size_t>(config_.max_length)) {
        tokens.resize(config_.max_length - 1);
        tokens.push_back(102);
    }
    
    return tokens;
}

std::vector<float> OnnxVectorizer::mean_pooling(
    const std::vector<float>& last_hidden_state,
    const std::vector<int64_t>& attention_mask,
    int64_t seq_length,
    int64_t hidden_size) {
    
    std::vector<float> pooled(hidden_size, 0.0f);
    
    // усредняем эмбеддинги только реальных токенов (где attention_mask = 1)
    float mask_sum = 0.0f;
    for (int64_t i = 0; i < seq_length; i++) {
        if (attention_mask[i] == 1) {
            mask_sum += 1.0f;
            for (int64_t j = 0; j < hidden_size; j++) {
                pooled[j] += last_hidden_state[i * hidden_size + j];
            }
        }
    }
    
    // нормализуем на количество токенов
    if (mask_sum > 0) {
        for (int64_t j = 0; j < hidden_size; j++) {
            pooled[j] /= mask_sum;
        }
    }
    
    return pooled;
}

std::vector<float> OnnxVectorizer::l2_normalize(const std::vector<float>& vec) {
    // вычисляем L2 норму вектора
    float sum = 0.0f;
    for (float v : vec) {
        sum += v * v;
    }
    
    // делим каждый элемент на норму для получения единичного вектора
    float norm = std::sqrt(sum);
    if (norm > 0) {
        std::vector<float> normalized(vec.size());
        for (size_t i = 0; i < vec.size(); i++) {
            normalized[i] = vec[i] / norm;
        }
        return normalized;
    }
    
    return vec;
}

std::vector<float> OnnxVectorizer::post_process(
    const std::vector<float>& output,
    const std::vector<int64_t>& attention_mask) {
    
    int64_t seq_length = attention_mask.size();
    int64_t hidden_size = config_.embedding_dim;
    
    // применяем mean pooling для агрегации эмбеддингов токенов
    auto pooled = mean_pooling(output, attention_mask, seq_length, hidden_size);
    
    // нормализуем финальный эмбеддинг для использования в косинусном сходстве
    return l2_normalize(pooled);
}

std::vector<float> OnnxVectorizer::vectorize(const std::string& text) {
    // векторизуем один текст через батч-версию
    auto batch = vectorizeBatch({text});
    return batch.empty() ? std::vector<float>(config_.embedding_dim, 0.0f) : batch[0];
}

std::vector<std::vector<float>> OnnxVectorizer::vectorizeBatch(const std::vector<std::string>& texts) {
    if (texts.empty()) return {};
    
    try {
        size_t batch_size = texts.size();
        int64_t max_len = config_.max_length;
        
        // подготавливаем входные тензоры для всего батча
        std::vector<int64_t> all_input_ids(batch_size * max_len, 0);  // ID токенов
        std::vector<int64_t> all_attention_mask(batch_size * max_len, 0);  // маска внимания
        std::vector<int64_t> all_token_type_ids(batch_size * max_len, 0);  // типы токенов (segment IDs)
        
        // токенизируем каждый текст в батче
        for (size_t i = 0; i < batch_size; i++) {
            auto tokens = tokenize(texts[i]);
            size_t seq_len = std::min(tokens.size(), static_cast<size_t>(max_len));
            
            for (size_t j = 0; j < seq_len; j++) {
                all_input_ids[i * max_len + j] = tokens[j];
                all_attention_mask[i * max_len + j] = 1;  // 1 для реальных токенов
                all_token_type_ids[i * max_len + j] = 0;  // 0 для первого сегмента
            }
        }
        
        // размерности для тензоров [batch_size, sequence_length]
        std::vector<int64_t> input_shape = {static_cast<int64_t>(batch_size), max_len};
        
        // создаем Ort::Value тензоры из подготовленных данных
        Ort::Value input_tensor = Ort::Value::CreateTensor<int64_t>(
            *memory_info_, all_input_ids.data(), all_input_ids.size(),
            input_shape.data(), input_shape.size()
        );
        
        Ort::Value attention_tensor = Ort::Value::CreateTensor<int64_t>(
            *memory_info_, all_attention_mask.data(), all_attention_mask.size(),
            input_shape.data(), input_shape.size()
        );
        
        Ort::Value token_type_tensor = Ort::Value::CreateTensor<int64_t>(
            *memory_info_, all_token_type_ids.data(), all_token_type_ids.size(),
            input_shape.data(), input_shape.size()
        );
        
        // подготавливаем входы для инференса
        std::vector<Ort::Value> ort_inputs;
        ort_inputs.push_back(std::move(input_tensor));
        ort_inputs.push_back(std::move(attention_tensor));
        ort_inputs.push_back(std::move(token_type_tensor));
        
        // конвертируем имена в C-строки для ONNX Runtime API
        std::vector<const char*> input_names_c;
        for (const auto& name : input_names_) {
            input_names_c.push_back(name.c_str());
        }
        
        std::vector<const char*> output_names_c;
        for (const auto& name : output_names_) {
            output_names_c.push_back(name.c_str());
        }
        
        // запускаем инференс модели
        auto output_tensors = session_->Run(
            Ort::RunOptions{nullptr},
            input_names_c.data(), ort_inputs.data(), ort_inputs.size(),
            output_names_c.data(), output_names_c.size()
        );
        
        // обрабатываем выходы модели
        std::vector<std::vector<float>> embeddings;
        
        if (!output_tensors.empty()) {
            // получаем сырые данные из выходного тензора
            float* output_data = output_tensors[0].GetTensorMutableData<float>();
            auto output_shape = output_tensors[0].GetTensorTypeAndShapeInfo().GetShape();
            
            // output_shape: [batch_size, sequence_length, hidden_size]
            int64_t seq_len = output_shape[1];
            int64_t hidden_size = output_shape[2];
            
            // обрабатываем каждый элемент батча
            for (size_t i = 0; i < batch_size; i++) {
                // извлекаем эмбеддинг для i-го текста
                std::vector<float> token_embeddings;
                size_t offset = i * seq_len * hidden_size;  // смещение до начала данных i-го текста
                
                // копируем все скрытые состояния для i-го текста
                for (int64_t s = 0; s < seq_len; s++) {
                    for (int64_t h = 0; h < hidden_size; h++) {
                        token_embeddings.push_back(output_data[offset + s * hidden_size + h]);
                    }
                }
                
                // применяем mean pooling и нормализацию
                std::vector<int64_t> attn_mask_slice(
                    all_attention_mask.begin() + i * max_len,
                    all_attention_mask.begin() + (i + 1) * max_len
                );
                
                auto embedding = post_process(token_embeddings, attn_mask_slice);
                embeddings.push_back(embedding);
            }
        }
        
        return embeddings;
        
    } catch (const Ort::Exception& e) {
        std::cout << "Ошибка инференса ONNX: " << e.what() << std::endl;
        return {};  // возвращаем пустой вектор в случае ошибки
    }
}

} 