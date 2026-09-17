#include "vectordb/storage_manager.hpp"
#include "schema_generated.h"
#include <fstream>
#include <thread>
#include <algorithm>
#include <iostream>

namespace VectorDB {

// конструктор (запоминает размерность векторов)
StorageManager::StorageManager(size_t dims) : dimensions(dims) {
    std::cerr << "[DEBUG] StorageManager created with dimensions: " << dims << std::endl;
}



// универсальное добавление вектора (общая логика для чатов и книг)
bool StorageManager::add_vector(int32_t id, std::vector<float> vec, 
                                std::string chat_id, std::string message_id,
                                std::string book_id, int32_t chunk_index,
                                RecordType record_type) {
    std::cerr << "[DEBUG] add_vector called - id: " << id 
              << ", type: " << static_cast<uint32_t>(record_type)
              << ", vec_size: " << vec.size() 
              << ", chat_id: " << chat_id 
              << ", message_id: " << message_id
              << ", book_id: " << book_id
              << ", chunk_index: " << chunk_index << std::endl;
    
    // проверяем размерность 
    if (vec.size() != dimensions) {
        std::cerr << "[ERROR] Vector dimension mismatch! Expected: " << dimensions 
                  << ", got: " << vec.size() << std::endl;
        return false;
    }
    
    // складываем в общее хранилище
    storage.push_back(VectorItem{id, std::move(vec), record_type, chat_id, message_id, book_id, chunk_index});
    std::cerr << "[DEBUG] Vector added successfully. Total vectors: " << storage.size() << std::endl;
    return true;
}

// добавление вектора чата 
bool StorageManager::add_chat_vector(int32_t id, std::vector<float> vec,
                                     std::string chat_id, std::string message_id) {
    return add_vector(id, std::move(vec), chat_id, message_id, "", 0, RecordType_CHAT);
}

// добавление вектора книги 
bool StorageManager::add_book_vector(int32_t id, std::vector<float> vec,
                                     std::string book_id, int32_t chunk_index) {
    return add_vector(id, std::move(vec), "", "", book_id, chunk_index, RecordType_BOOK);
}



// сохраняем всё хранилище в бинарный файл
std::expected<void, std::string> StorageManager::save_to_file(std::string_view filename) const {
    std::cerr << "[DEBUG] save_to_file called. Filename: " << filename 
              << ", vectors count: " << storage.size() << std::endl;
    
    flatbuffers::FlatBufferBuilder builder(1024);  // начинаем с буфера в 1KB
    std::vector<flatbuffers::Offset<VectorDB::Document>> doc_offsets;
    doc_offsets.reserve(storage.size());  // резервируем место под все документы

    // сериализуем каждый вектор
    for (size_t i = 0; i < storage.size(); ++i) {
        const auto& doc = storage[i];
        
        auto vec_offset = builder.CreateVector(doc.values);  // массив float
        auto chat_id_offset = builder.CreateString(doc.chat_id);
        auto message_id_offset = builder.CreateString(doc.message_id);
        auto book_id_offset = builder.CreateString(doc.book_id);
        
        // собираем объект Document
        auto doc_offset = VectorDB::CreateDocument(
            builder, doc.id, vec_offset, doc.record_type,
            chat_id_offset, message_id_offset,
            book_id_offset, doc.chunk_index);
        doc_offsets.push_back(doc_offset);
    }

    // корневой объект — storage с массивом документов
    auto docs_vector_offset = builder.CreateVector(doc_offsets);
    auto storage_offset = VectorDB::CreateStorage(builder, docs_vector_offset);
    builder.Finish(storage_offset);

    // пишем в файл
    std::ofstream outfile(std::string(filename), std::ios::binary);
    if (!outfile) {
        std::cerr << "[ERROR] Failed to open file for writing: " << filename << std::endl;
        return std::unexpected("Не удалось открыть файл для записи: " + std::string(filename));
    }
    
    outfile.write(reinterpret_cast<const char*>(builder.GetBufferPointer()), builder.GetSize());
    
    // проверяем что запись реально прошла
    if (!outfile) {
        std::cerr << "[ERROR] Failed to write data to file!" << std::endl;
        return std::unexpected("Ошибка при физической записи данных на диск.");
    }
    
    std::cerr << "[DEBUG] File saved successfully! Size: " << builder.GetSize() << " bytes" << std::endl;
    return {};
}

// загружаем хранилище из бинарного файла
std::expected<void, std::string> StorageManager::load_from_file(std::string_view filename) {
    std::cerr << "[DEBUG] load_from_file called. Filename: " << filename << std::endl;
    
    // узнаём размер
    std::ifstream infile(std::string(filename), std::ios::binary | std::ios::ate);
    if (!infile) {
        std::cerr << "[ERROR] Failed to open file for reading: " << filename << std::endl;
        return std::unexpected("Не удалось открыть файл для чтения: " + std::string(filename));
    }

    std::streamsize size = infile.tellg();  // размер файла
    std::cerr << "[DEBUG] File size: " << size << " bytes" << std::endl;
    infile.seekg(0, std::ios::beg);  // возвращаемся в начало

    // читаем весь файл в буфер
    std::vector<uint8_t> buffer(size);
    if (!infile.read(reinterpret_cast<char*>(buffer.data()), size)) {
        std::cerr << "[ERROR] Failed to read data from file!" << std::endl;
        return std::unexpected("Ошибка при чтении данных из файла!");
    }
    
    std::cerr << "[DEBUG] Successfully read " << infile.gcount() << " bytes" << std::endl;

    // минимальная проверка буфера
    if (buffer.size() < 4) {
        std::cerr << "[ERROR] Buffer too small: " << buffer.size() << " bytes" << std::endl;
        return std::unexpected("Буфер слишком маленький, возможно файл повреждён");
    }
    
    // FlatBuffers (проверяем целостность)
    flatbuffers::Verifier verifier(buffer.data(), buffer.size());
    if (!VectorDB::VerifyStorageBuffer(verifier)) {
        std::cerr << "[ERROR] Buffer verification failed!" << std::endl;
        return std::unexpected("Буфер повреждён или имеет неверный формат");
    }
    std::cerr << "[DEBUG] Buffer verification passed!" << std::endl;

    // получаем корневой объект
    auto storage_fb = VectorDB::GetStorage(buffer.data());
    if (!storage_fb) {
        std::cerr << "[ERROR] GetStorage returned nullptr!" << std::endl;
        return std::unexpected("Не удалось получить корневой объект Storage");
    }
    
    auto docs_fb = storage_fb->documents();
    if (!docs_fb) {
        std::cerr << "[ERROR] documents() returned nullptr!" << std::endl;
        return std::unexpected("Отсутствуют документы в хранилище");
    }
    
    std::cerr << "[DEBUG] Documents count: " << docs_fb->size() << std::endl;

    // десериализуем все документы во временное хранилище
    std::vector<VectorItem> temp_storage;
    if (docs_fb->size() > 0) {
        temp_storage.reserve(docs_fb->size());
        
        for (flatbuffers::uoffset_t i = 0; i < docs_fb->size(); ++i) {
            auto doc_fb = docs_fb->Get(i);
            if (!doc_fb || !doc_fb->values()) continue;  // битый документ пропускаем
            
            // проверяем размерность на первом элементе
            if (i == 0) {
                size_t file_dims = doc_fb->values()->size();
                    if (dimensions != file_dims) {
                        std::cerr << "[DEBUG] Adjusting StorageManager dimensions from " 
                        << dimensions << " to " << file_dims << " (based on file data)" << std::endl;
                        dimensions = file_dims; // Меняем размерность на лету!
                }
            }
            
            // копируем данные из flatbuffer в обычные структуры
            std::vector<float> vec(doc_fb->values()->begin(), doc_fb->values()->end());
            std::string chat_id = doc_fb->chat_id() ? doc_fb->chat_id()->str() : "";
            std::string message_id = doc_fb->message_id() ? doc_fb->message_id()->str() : "";
            std::string book_id = doc_fb->book_id() ? doc_fb->book_id()->str() : "";
            
            temp_storage.push_back(VectorItem{
                doc_fb->id(),
                std::move(vec),
                doc_fb->record_type(),
                chat_id,
                message_id,
                book_id,
                doc_fb->chunk_index()
            });
        }
    }
    
    std::cerr << "[DEBUG] Successfully parsed " << temp_storage.size() << " documents" << std::endl;
    storage = std::move(temp_storage);  // заменяем основное хранилище
    std::cerr << "[DEBUG] Storage loaded successfully! Total vectors: " << storage.size() << std::endl;
    return {};
}


// поиск top-k ближайших по косинусному сходству (через косинусное сходство)
std::vector<SearchResult> StorageManager::search_top_k(std::span<const float> query, size_t k) const {
    std::cerr << "[DEBUG] search_top_k - query_size: " << query.size() 
              << ", k: " << k 
              << ", storage_size: " << storage.size() 
              << ", dimensions: " << dimensions << std::endl;


    if (query.size() != dimensions || storage.empty() || k == 0) {
        return {};
    }

    size_t actual_k = std::min(k, storage.size());  // не можем вернуть больше чем есть
    std::vector<SearchResult> results;
    results.reserve(storage.size());

    // считаем сходство со всеми векторами (полный перебор, без индекса)
    for (size_t i = 0; i < storage.size(); ++i) {
        float sim = 0.0f;
        for (size_t j = 0; j < dimensions; ++j) {
            sim += query[j] * storage[i].values[j];  // скалярное произведение
        }
        results.push_back(SearchResult{
            storage[i].id, sim, storage[i].record_type,
            storage[i].chat_id, storage[i].message_id,
            storage[i].book_id, storage[i].chunk_index
        });
    }

    // частичная сортировка только первые k элементов
    std::partial_sort(results.begin(), results.begin() + actual_k, results.end(),
        [](const SearchResult& a, const SearchResult& b) {
            return a.similarity > b.similarity; 
        }
    );
    results.resize(actual_k);  // отрезаем остальные

    return results;
}

// поиск с фильтрацией по порогу сходства
std::vector<SearchResult> StorageManager::search_similar_context(
    std::span<const float> query, size_t k, float threshold) const {
    
    auto results = search_top_k(query, k);
    std::vector<SearchResult> filtered;
    
    // оставляем только те что выше порога
    for (const auto& r : results) {
        if (r.similarity >= threshold) {
            filtered.push_back(r);
        }
    }
    
    return filtered;
}

// поиск с фильтрацией по типу записи (чат / книга)
std::vector<SearchResult> StorageManager::search_by_type(
    std::span<const float> query, size_t k, RecordType type) const {
    
    std::cerr << "[DEBUG] search_by_type - type: " << static_cast<uint32_t>(type)
              << ", k: " << k << std::endl;
    
    // сначала ищем по всем, потом фильтруем по типу
    auto results = search_top_k(query, storage.size());
    std::vector<SearchResult> filtered;
    
    for (const auto& r : results) {
        if (r.record_type == type) {
            filtered.push_back(r);
            if (filtered.size() >= k) break;  // выход
        }
    }
    
    std::cerr << "[DEBUG] search_by_type returned " << filtered.size() << " results" << std::endl;
    return filtered;
}

} // namespace VectorDB