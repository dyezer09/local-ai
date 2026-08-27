#pragma once

#include <string>
#include <vector>
#include <iostream>
#include <nlohmann/json.hpp>  // Только для ответов (статус OK/ERROR)
#include "schema_generated.h"

namespace TextReceiver {

using json = nlohmann::json;

// ============================================================
// СТРУКТУРЫ ДАННЫХ ИЗ FLATBUFFERS
// ============================================================

// Один вектор с метаданными
struct VectorData {
    int32_t id;
    std::vector<float> embedding;
    VectorDB::RecordType record_type;
    std::string chat_id;
    std::string message_id;
    std::string book_id;
    int32_t chunk_index;
    
    VectorData() 
        : id(0)
        , record_type(VectorDB::RecordType_CHAT)
        , chunk_index(0) {}
};

// Контейнер для всех векторов
struct StorageData {
    std::vector<VectorData> vectors;
    size_t total_vectors;
    
    StorageData() : total_vectors(0) {}
    
    void clear() {
        vectors.clear();
        total_vectors = 0;
    }
    
    bool empty() const {
        return vectors.empty();
    }
    
    size_t size() const {
        return vectors.size();
    }
};

// ============================================================
// ОСНОВНЫЕ ФУНКЦИИ
// ============================================================

// Чтение и парсинг FlatBuffers из файла
bool readFlatBuffersFromFile(const std::string& filename, StorageData& storage);

// Чтение и парсинг FlatBuffers из файла с сообщением об ошибке
bool readFlatBuffersFromFile(const std::string& filename, StorageData& storage, std::string& error_msg);

// Парсинг FlatBuffers из бинарного буфера
bool parseFlatBuffersFromBuffer(const std::vector<uint8_t>& buffer, StorageData& storage);

// Парсинг FlatBuffers из бинарного буфера с сообщением об ошибке
bool parseFlatBuffersFromBuffer(const std::vector<uint8_t>& buffer, StorageData& storage, std::string& error_msg);

// ============================================================
// ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ
// ============================================================

// Печать данных (для отладки)
void printStorageData(const StorageData& storage);

// Создание JSON ответа (только для HTTP статуса)
json createResponse(bool success, const std::string& message);

// Конвертация RecordType в строку
std::string recordTypeToString(VectorDB::RecordType type);

// Проверка валидности StorageData
bool validateStorageData(const StorageData& storage);

// ============================================================
// ФУНКЦИИ ДЛЯ РАБОТЫ С ФАЙЛАМИ (утилиты)
// ============================================================

// Проверка существования файла
bool fileExists(const std::string& filename);

// Удаление файла
bool removeFile(const std::string& filename);

// Чтение бинарного файла в буфер
std::vector<uint8_t> readBinaryFile(const std::string& filename);

// Запись бинарного файла (для отладки)
bool writeBinaryFile(const std::string& filename, const std::vector<uint8_t>& data);

} // namespace TextReceiver