#pragma once

#include <string>
#include <vector>
#include <onnxruntime_cxx_api.h>
#include <nlohmann/json.hpp>

namespace VectorDB {

using json = nlohmann::json;

struct VectorizerConfig {
    std::string model_path;
    int64_t max_length = 128;
    int64_t embedding_dim = 384;
};

class OnnxVectorizer {
private:
    std::unique_ptr<Ort::Env> env_;
    std::unique_ptr<Ort::Session> session_;
    std::unique_ptr<Ort::MemoryInfo> memory_info_;
    
    VectorizerConfig config_;
    
    std::vector<std::string> input_names_;
    std::vector<std::string> output_names_;
    
    std::vector<int64_t> tokenize(const std::string& text);
    std::vector<float> mean_pooling(const std::vector<float>& last_hidden_state,
                                     const std::vector<int64_t>& attention_mask,
                                     int64_t seq_length,
                                     int64_t hidden_size);
    std::vector<float> l2_normalize(const std::vector<float>& vec);
    std::vector<float> post_process(const std::vector<float>& output,
                                    const std::vector<int64_t>& attention_mask);

public:
    explicit OnnxVectorizer(const VectorizerConfig& config);
    ~OnnxVectorizer();
    
    std::vector<float> vectorize(const std::string& text);
    std::vector<std::vector<float>> vectorizeBatch(const std::vector<std::string>& texts);
    
    int64_t getEmbeddingDim() const { return config_.embedding_dim; }
};

} 