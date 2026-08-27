#pragma once

#include <vector>
#include <string>
#include <span>
#include <expected>
#include <optional>
#include <cstdint>
#include "schema_generated.h"

namespace VectorDB {

struct SearchResult {
    int32_t id;
    float similarity;
    RecordType record_type;
    std::string chat_id;
    std::string message_id;
    std::string book_id;
    int32_t chunk_index = 0;
};

struct VectorItem {
    int32_t id;
    std::vector<float> values;
    RecordType record_type;
    std::string chat_id;
    std::string message_id;
    std::string book_id;
    int32_t chunk_index = 0;
};

class StorageManager {
public:
    explicit StorageManager(size_t dims);
    
    bool add_vector(int32_t id, std::vector<float> vec, 
                    std::string chat_id = "", std::string message_id = "",
                    std::string book_id = "", int32_t chunk_index = 0,
                    RecordType record_type = RecordType_CHAT);
    
    bool add_chat_vector(int32_t id, std::vector<float> vec,
                         std::string chat_id, std::string message_id);
    
    bool add_book_vector(int32_t id, std::vector<float> vec,
                         std::string book_id, int32_t chunk_index);
    
    std::expected<void, std::string> save_to_file(std::string_view filename) const;
    
    std::expected<void, std::string> load_from_file(std::string_view filename);
    
    std::vector<SearchResult> search_top_k(std::span<const float> query, size_t k) const;
    
    std::vector<SearchResult> search_similar_context(std::span<const float> query, 
                                                      size_t k, float threshold = 0.5f) const;
    
    std::vector<SearchResult> search_by_type(std::span<const float> query, size_t k,
                                             RecordType type) const;
    
    size_t size() const { return storage.size(); }
    size_t get_dimensions() const { return dimensions; }

private:
    std::vector<VectorItem> storage;
    size_t dimensions;
};

} // namespace VectorDB







