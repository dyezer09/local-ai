// src/text_receiver.cpp
#include "../include/vectordb/text_receiver.hpp"
#include <fstream>
#include <filesystem>
#include <chrono>
#include <iomanip>
#include <flatbuffers/flatbuffers.h>

namespace TextReceiver {

namespace fs = std::filesystem;

// ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ ДЛЯ РАБОТЫ С ФАЙЛАМИ

bool fileExists(const std::string& filename) {
    return fs::exists(filename);
}

bool removeFile(const std::string& filename) {
    try {
        if (fs::exists(filename)) {
            fs::remove(filename);
            return true;
        }
        return false;
    } catch (const std::exception& e) {
        std::cerr << "❌ Ошибка удаления файла: " << e.what() << std::endl;
        return false;
    }
}

std::vector<uint8_t> readBinaryFile(const std::string& filename) {
    std::ifstream infile(filename, std::ios::binary | std::ios::ate);
    if (!infile) {
        std::cerr << "❌ Не удалось открыть файл: " << filename << std::endl;
        return {};
    }
    
    size_t file_size = infile.tellg();
    infile.seekg(0, std::ios::beg);
    
    std::vector<uint8_t> buffer(file_size);
    infile.read(reinterpret_cast<char*>(buffer.data()), file_size);
    infile.close();
    
    return buffer;
}

bool writeBinaryFile(const std::string& filename, const std::vector<uint8_t>& data) {
    std::ofstream outfile(filename, std::ios::binary);
    if (!outfile) {
        std::cerr << "❌ Не удалось создать файл: " << filename << std::endl;
        return false;
    }
    
    outfile.write(reinterpret_cast<const char*>(data.data()), data.size());
    outfile.close();
    return true;
}

// ПАРСИНГ FLATBUFFERS

bool parseFlatBuffersFromBuffer(const std::vector<uint8_t>& buffer, StorageData& storage) {
    std::string error_msg;
    return parseFlatBuffersFromBuffer(buffer, storage, error_msg);
}

bool parseFlatBuffersFromBuffer(const std::vector<uint8_t>& buffer, StorageData& storage, std::string& error_msg) {
    storage.clear();
    
    if (buffer.empty()) {
        error_msg = "Buffer is empty";
        return false;
    }
    
    flatbuffers::Verifier verifier(buffer.data(), buffer.size());
    if (!VectorDB::VerifyStorageBuffer(verifier)) {
        error_msg = "FlatBuffers verification failed";
        std::cerr << "❌ " << error_msg << std::endl;
        return false;
    }
    
    auto storage_fb = VectorDB::GetStorage(buffer.data());
    if (!storage_fb) {
        error_msg = "Failed to get Storage object";
        std::cerr << "❌ " << error_msg << std::endl;
        return false;
    }
    
    auto documents = storage_fb->documents();
    if (!documents || documents->size() == 0) {
        error_msg = "No documents in storage";
        std::cerr << "⚠️ " << error_msg << std::endl;
        return true;
    }
    
    for (const auto* doc : *documents) {
        if (!doc) continue;
        
        VectorData vec_data;
        
        vec_data.id = doc->id();
        
        auto values = doc->values();
        if (values && values->size() > 0) {
            vec_data.embedding.reserve(values->size());
            for (size_t i = 0; i < values->size(); ++i) {
                vec_data.embedding.push_back((*values)[i]);
            }
        } else {
            std::cerr << "⚠️  Документ " << vec_data.id << " имеет пустой вектор" << std::endl;
            continue;
        }
        
        vec_data.record_type = doc->record_type();
        
        if (doc->chat_id()) {
            vec_data.chat_id = doc->chat_id()->str();
        }
        if (doc->message_id()) {
            vec_data.message_id = doc->message_id()->str();
        }
        if (doc->book_id()) {
            vec_data.book_id = doc->book_id()->str();
        }
        vec_data.chunk_index = doc->chunk_index();
        
        storage.vectors.push_back(std::move(vec_data));
    }
    
    storage.total_vectors = storage.vectors.size();
    
    return true;
}


// ЧТЕНИЕ ИЗ ФАЙЛА


bool readFlatBuffersFromFile(const std::string& filename, StorageData& storage) {
    std::string error_msg;
    return readFlatBuffersFromFile(filename, storage, error_msg);
}

bool readFlatBuffersFromFile(const std::string& filename, StorageData& storage, std::string& error_msg) {
    if (!fileExists(filename)) {
        error_msg = "File does not exist: " + filename;
        std::cerr << "❌ " << error_msg << std::endl;
        return false;
    }
    
    auto buffer = readBinaryFile(filename);
    if (buffer.empty()) {
        error_msg = "File is empty or could not be read: " + filename;
        std::cerr << "❌ " << error_msg << std::endl;
        return false;
    }
    
    std::cout << "📦 Прочитано " << buffer.size() << " байт из файла: " << filename << std::endl;
    
    return parseFlatBuffersFromBuffer(buffer, storage, error_msg);
}

// ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ

std::string recordTypeToString(VectorDB::RecordType type) {
    switch (type) {
        case VectorDB::RecordType_CHAT:
            return "CHAT";
        case VectorDB::RecordType_BOOK:
            return "BOOK";
        default:
            return "UNKNOWN";
    }
}

void printStorageData(const StorageData& storage) {
    std::cout << "════════════════════════════════════════════════════════════" << std::endl;
    std::cout << "📊 Данные из FlatBuffers" << std::endl;
    std::cout << "════════════════════════════════════════════════════════════" << std::endl;
    std::cout << "   Всего векторов: " << storage.total_vectors << std::endl;
    std::cout << "────────────────────────────────────────────────────────────" << std::endl;
    
    for (size_t i = 0; i < storage.vectors.size(); ++i) {
        const auto& vec = storage.vectors[i];
        std::cout << "📄 Вектор #" << (i + 1) << ":" << std::endl;
        std::cout << "   ID: " << vec.id << std::endl;
        std::cout << "   Тип: " << recordTypeToString(vec.record_type) << std::endl;
        std::cout << "   Размерность: " << vec.embedding.size() << std::endl;
        
        if (vec.record_type == VectorDB::RecordType_CHAT) {
            std::cout << "   Chat ID: " << vec.chat_id << std::endl;
            std::cout << "   Message ID: " << vec.message_id << std::endl;
        } else if (vec.record_type == VectorDB::RecordType_BOOK) {
            std::cout << "   Book ID: " << vec.book_id << std::endl;
            std::cout << "   Chunk Index: " << vec.chunk_index << std::endl;
        }
        
        if (!vec.embedding.empty()) {
            std::cout << "   Первые 5 значений: ";
            for (size_t j = 0; j < std::min<size_t>(5, vec.embedding.size()); ++j) {
                std::cout << std::fixed << std::setprecision(4) << vec.embedding[j];
                if (j < 4) std::cout << ", ";
            }
            std::cout << " ..." << std::endl;
        }
        std::cout << std::endl;
    }
    std::cout << "════════════════════════════════════════════════════════════" << std::endl;
}

json createResponse(bool success, const std::string& message) {
    json response;
    response["status"] = success ? "success" : "error";
    response["message"] = message;
    return response;
}

bool validateStorageData(const StorageData& storage) {
    if (storage.vectors.empty()) {
        std::cerr << "❌ Нет векторов в StorageData" << std::endl;
        return false;
    }
    
    if (!storage.vectors.empty()) {
        size_t dims = storage.vectors[0].embedding.size();
        for (size_t i = 1; i < storage.vectors.size(); ++i) {
            if (storage.vectors[i].embedding.size() != dims) {
                std::cerr << "❌ Несовпадение размерности векторов: "
                          << storage.vectors[i].embedding.size() 
                          << " != " << dims << std::endl;
                return false;
            }
        }
    }
    
    return true;
}

} // namespace TextReceiver