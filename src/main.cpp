#include "../include/vectordb/storage_manager.hpp"
#include "../generated/schema_generated.h"
#include "../include/vectordb/text_receiver.hpp"     
#include "../include/vectordb/vectorizer.hpp"             
#include <httplib.h>                     
#include <iostream>
#include <fstream>
#include <chrono>                        
#include <cassert>
#include <atomic>                         
#include <thread>                         
#include <mutex>                        
#include <filesystem>
#include <iomanip>

using namespace TextReceiver;

namespace fs = std::filesystem;

std::atomic<int32_t> g_next_auto_id{1000};
std::atomic<bool> g_server_running{true};
std::mutex g_db_mutex;

std::unique_ptr<VectorDB::StorageManager> g_db;
std::unique_ptr<VectorDB::OnnxVectorizer> g_vectorizer;
std::string g_filename = "vector_storage.bin";
size_t g_current_dims = 384;

// Пути для файлового обмена с Node.js
std::string g_data_file = "/home/den/project/vectorDB/data/chat_data.bin";
std::string g_ready_file = g_data_file + ".ready";


// определение размерности из файла
size_t peek_flatbuffers_dimensions(const std::string& filename) {
    std::ifstream infile(filename, std::ios::binary | std::ios::ate);  
    if (!infile) {
        std::cerr << "[DEBUG peek] File not found, using default dims: 384" << std::endl;
        return 384;  
    }
    
    size_t file_size = infile.tellg();
    infile.seekg(0, std::ios::beg);
    
    std::vector<uint8_t> buffer(file_size);
    infile.read(reinterpret_cast<char*>(buffer.data()), file_size);
    infile.close();
    
    flatbuffers::Verifier verifier(buffer.data(), buffer.size());
    if (!VectorDB::VerifyStorageBuffer(verifier)) {
        std::cerr << "[ERROR peek] Buffer verification failed!" << std::endl;
        return 384;
    }
    
    auto storage_fb = VectorDB::GetStorage(buffer.data());
    if (!storage_fb) {
        std::cerr << "[ERROR peek] GetStorage returned nullptr!" << std::endl;
        return 384;
    }
    
    auto docs = storage_fb->documents();
    if (!docs || docs->size() == 0) {
        std::cerr << "[DEBUG peek] No documents in storage" << std::endl;
        return 384;
    }
    
    auto first_doc = docs->Get(0);
    if (!first_doc || !first_doc->values()) {
        std::cerr << "[ERROR peek] First document invalid" << std::endl;
        return 384;
    }
    
    size_t dims = first_doc->values()->size();
    std::cerr << "[DEBUG peek] Detected dimensions: " << dims << std::endl;
    return dims;
}

// 
bool processFlatBuffersFromFile(const std::string& filename) {
    StorageData storage;
    std::string error_msg;
    
    if (!readFlatBuffersFromFile(filename, storage, error_msg)) {
        std::cerr << "❌ Ошибка чтения FlatBuffers: " << error_msg << std::endl;
        return false;
    }
    
    // ВЫВЕДИТЕ ОТЛАДОЧНУЮ ИНФОРМАЦИЮ
    std::cout << "📊 Получено " << storage.vectors.size() << " векторов" << std::endl;
    for (const auto& vec : storage.vectors) {
        std::cout << "   - ID: " << vec.id 
                  << ", размерность: " << vec.embedding.size()
                  << ", тип: " << recordTypeToString(vec.record_type) << std::endl;
    }
    
    // Читаем FlatBuffers из файла через text_receiver
    if (!readFlatBuffersFromFile(filename, storage, error_msg)) {
        std::cerr << "❌ Ошибка чтения FlatBuffers: " << error_msg << std::endl;
        return false;
    }
    
    if (storage.vectors.empty()) {
        std::cerr << "⚠️ Нет данных в файле" << std::endl;
        return false;
    }
    
    // Выводим полученные данные
    printStorageData(storage);
    
    // Блокируем БД
    std::lock_guard<std::mutex> lock(g_db_mutex);
    
    // Проверяем размерность
    size_t embedding_dim = storage.vectors[0].embedding.size();
    if (g_db->size() == 0) {
        g_current_dims = embedding_dim;
        g_db = std::make_unique<VectorDB::StorageManager>(g_current_dims);
        std::cout << "📐 Установлена размерность: " << g_current_dims << std::endl;
    } else if (embedding_dim != g_db->get_dimensions()) {
        std::cerr << "❌ Несовпадение размерности! Ожидаем: " 
                  << g_db->get_dimensions() << ", получено: " << embedding_dim << std::endl;
        return false;
    }
    
    // Добавляем каждый вектор в БД
    int added_count = 0;
    for (const auto& vec_data : storage.vectors) {
        int32_t new_id = g_next_auto_id.fetch_add(1);
        
        bool added = false;
        if (vec_data.record_type == VectorDB::RecordType_CHAT) {
            added = g_db->add_vector(
                new_id,
                vec_data.embedding,
                vec_data.chat_id,
                vec_data.message_id,
                "",  // book_id
                0,   // chunk_index
                VectorDB::RecordType_CHAT
            );
        } else if (vec_data.record_type == VectorDB::RecordType_BOOK) {
            added = g_db->add_vector(
                new_id,
                vec_data.embedding,
                "",  // chat_id
                "",  // message_id
                vec_data.book_id,
                vec_data.chunk_index,
                VectorDB::RecordType_BOOK
            );
        } else {
            // По умолчанию как CHAT
            added = g_db->add_vector(
                new_id,
                vec_data.embedding,
                vec_data.chat_id,
                vec_data.message_id,
                "",
                0,
                VectorDB::RecordType_CHAT
            );
        }
        
        if (added) {
            added_count++;
            std::cout << "✅ Добавлен вектор ID: " << new_id 
                      << " | Тип: " << recordTypeToString(vec_data.record_type);
            if (vec_data.record_type == VectorDB::RecordType_CHAT) {
                std::cout << " | Chat: " << vec_data.chat_id 
                          << " | Msg: " << vec_data.message_id;
            } else if (vec_data.record_type == VectorDB::RecordType_BOOK) {
                std::cout << " | Book: " << vec_data.book_id 
                          << " | Chunk: " << vec_data.chunk_index;
            }
            std::cout << std::endl;
        }
    }
    
    // Сохраняем базу
    if (added_count > 0) {
        auto save_result = g_db->save_to_file(g_filename);
        if (save_result.has_value()) {
            std::cout << "💾 База сохранена: " << g_filename 
                      << " (всего: " << g_db->size() << ")" << std::endl;
        }
    }
    
    std::cout << "📊 Добавлено " << added_count << " векторов. Всего в БД: " << g_db->size() << std::endl;
    return added_count > 0;
}

// ============================================================
// ФУНКЦИЯ: Мониторинг файлов в фоновом потоке
// ============================================================
void handleFlatBuffersFile() {
    std::cout << "📂 Мониторинг файла: " << g_data_file << std::endl;
    std::cout << "📂 Индикатор готовности: " << g_ready_file << std::endl;
    
    // Создаем директорию если её нет
    fs::path data_dir = fs::path(g_data_file).parent_path();
    if (!data_dir.empty() && !fs::exists(data_dir)) {
        fs::create_directories(data_dir);
        std::cout << "📁 Создана директория: " << data_dir << std::endl;
    }
    
    int processed_count = 0;
    
    while (g_server_running) {
        // Проверяем наличие файла-индикатора
        if (fs::exists(g_ready_file)) {
            std::cout << "\n📁 Обнаружен новый файл-индикатор!" << std::endl;
            
            try {
                // Читаем и обрабатываем данные через text_receiver
                bool success = processFlatBuffersFromFile(g_data_file);
                
                // Удаляем обработанные файлы
                removeFile(g_ready_file);
                removeFile(g_data_file);
                
                if (success) {
                    processed_count++;
                    std::cout << "🗑️  Файлы удалены. Обработано: " << processed_count << std::endl;
                }
                
            } catch (const std::exception& e) {
                std::cerr << "❌ Ошибка обработки файла: " << e.what() << std::endl;
                // Пытаемся очистить файлы чтобы не зациклиться
                removeFile(g_ready_file);
                removeFile(g_data_file);
            }
        }
        
        // Небольшая задержка, чтобы не нагружать процессор
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
}

// ============================================================
// ФУНКЦИЯ: HTTP сервер (только эндпоинты для FlatBuffers)
// ============================================================
void start_http_server() {
    try {
        httplib::Server server;
        
        // ============================================================
        // ЭНДПОИНТ: Прием FlatBuffers через файл (триггер)
        // ============================================================
        server.Post("/receive_flatbuffers", [](const httplib::Request& req, httplib::Response& res) {
            try {
                json response;
                response["status"] = "success";
                response["message"] = "FlatBuffers file processing triggered";
                response["data_file"] = g_data_file;
                response["ready_file"] = g_ready_file;
                
                res.set_content(response.dump(), "application/json");
                
                std::cout << "\n📨 Получен запрос на обработку FlatBuffers файла" << std::endl;
                
            } catch (const std::exception& e) {
                std::cerr << "❌ /receive_flatbuffers error: " << e.what() << std::endl;
                json err;
                err["status"] = "error";
                err["message"] = e.what();
                res.set_content(err.dump(), "application/json");
            }
        });
        
        // ============================================================
        // ЭНДПОИНТ: Векторизация книги (оставляем для обратной совместимости)
        // ============================================================
        server.Post("/vectorize_book", [](const httplib::Request& req, httplib::Response& res) {
            try {
                json request_data = json::parse(req.body);
                
                if (!request_data.contains("texts") || !request_data["texts"].is_array()) {
                    json err;
                    err["status"] = "error";
                    err["message"] = "Missing texts array";
                    res.set_content(err.dump(), "application/json");
                    return;
                }
                
                std::string book_id = request_data.value("book_id", "unknown");
                std::vector<std::string> texts;
                std::vector<int32_t> chunk_indices;
                
                for (const auto& t : request_data["texts"]) {
                    texts.push_back(t.get<std::string>());
                }
                
                if (request_data.contains("chunk_indices") && request_data["chunk_indices"].is_array()) {
                    for (const auto& ci : request_data["chunk_indices"]) {
                        chunk_indices.push_back(ci.get<int32_t>());
                    }
                }
                
                if (!g_vectorizer || !g_db) {
                    json err;
                    err["status"] = "error";
                    err["message"] = "Vectorizer not ready";
                    res.set_content(err.dump(), "application/json");
                    return;
                }
                
                std::cout << "\n📖 Векторизация книги ID=" << book_id
                          << " (" << texts.size() << " чанков)" << std::endl;
                
                auto start_vec = std::chrono::high_resolution_clock::now();
                auto embeddings = g_vectorizer->vectorizeBatch(texts);
                auto end_vec = std::chrono::high_resolution_clock::now();
                auto vec_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end_vec - start_vec).count();
                
                std::cout << "   ✅ Векторизация: " << vec_ms << " мс" << std::endl;
                
                std::lock_guard<std::mutex> lock(g_db_mutex);
                
                json positions = json::array();
                
                for (size_t i = 0; i < embeddings.size(); ++i) {
                    if (embeddings[i].empty()) continue;
                    
                    int32_t new_id = g_next_auto_id.fetch_add(1);
                    int32_t chunk_idx = (i < chunk_indices.size()) ? chunk_indices[i] : static_cast<int32_t>(i);
                    
                    bool added = g_db->add_vector(
                        new_id, 
                        embeddings[i],
                        "", "",
                        book_id, 
                        chunk_idx,
                        VectorDB::RecordType_BOOK
                    );
                    
                    if (added) {
                        json pos;
                        pos["chunkIndex"] = chunk_idx;
                        pos["vectorId"] = new_id;
                        pos["position"] = static_cast<int32_t>(g_db->size() - 1);
                        positions.push_back(pos);
                    }
                }
                
                auto start_save = std::chrono::high_resolution_clock::now();
                g_db->save_to_file(g_filename);
                auto end_save = std::chrono::high_resolution_clock::now();
                auto save_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end_save - start_save).count();
                
                std::cout << "   💾 Сохранение: " << save_ms << " мс"
                          << " (всего в БД: " << g_db->size() << ")" << std::endl;
                
                json response;
                response["status"] = "success";
                response["book_id"] = book_id;
                response["vectors_added"] = positions.size();
                response["total_in_db"] = g_db->size();
                response["positions"] = positions;
                
                res.set_content(response.dump(), "application/json");
                
            } catch (const std::exception& e) {
                std::cerr << "❌ /vectorize_book error: " << e.what() << std::endl;
                json err;
                err["status"] = "error";
                err["message"] = e.what();
                res.set_content(err.dump(), "application/json");
            }
        });
        
        // ============================================================
        // ЭНДПОИНТ: Поиск похожих (оставляем для обратной совместимости)
        // ============================================================
        server.Post("/search_similar", [](const httplib::Request& req, httplib::Response& res) {
            try {
                json request_data = json::parse(req.body);
                
                if (!request_data.contains("text") || !g_vectorizer || !g_db) {
                    json error;
                    error["status"] = "error";
                    error["message"] = "Missing text or vectorizer not ready";
                    res.set_content(error.dump(), "application/json");
                    return;
                }
                
                std::string text = request_data["text"].get<std::string>();
                int top_k = request_data.value("top_k", 3);
                std::string search_type = request_data.value("search_type", "all");
                
                auto embedding = g_vectorizer->vectorize(text);
                
                std::lock_guard<std::mutex> lock(g_db_mutex);
                
                std::vector<VectorDB::SearchResult> results;
                
                if (search_type == "chats") {
                    results = g_db->search_by_type(embedding, top_k + 1, VectorDB::RecordType_CHAT);
                } else if (search_type == "books") {
                    results = g_db->search_by_type(embedding, top_k + 1, VectorDB::RecordType_BOOK);
                } else {
                    results = g_db->search_top_k(embedding, top_k + 1);
                }
                
                json response;
                response["status"] = "success";
                response["query"] = text;
                response["search_type"] = search_type;
                response["results"] = json::array();
                
                int added = 0;
                for (const auto& r : results) {
                    if (r.similarity > 0.999f && added == 0) {
                        continue;
                    }
                    
                    json item;
                    item["similarity"] = r.similarity;
                    item["record_type"] = static_cast<uint32_t>(r.record_type);
                    item["chat_id"] = r.chat_id;
                    item["message_id"] = r.message_id;
                    item["book_id"] = r.book_id;
                    item["chunk_index"] = r.chunk_index;
                    item["vector_id"] = r.id;
                    response["results"].push_back(item);
                    
                    added++;
                    if (added >= top_k) break;
                }
                
                std::cout << "\n" << std::string(60, '-') << std::endl;
                std::cout << "🔍 Поиск похожих: \"" << text.substr(0, 50);
                if (text.length() > 50) std::cout << "...";
                std::cout << "\"" << std::endl;
                std::cout << "   Тип поиска: " << search_type;
                std::cout << " | Найдено: " << response["results"].size() << std::endl;
                std::cout << std::string(60, '-') << std::endl;
                
                for (size_t i = 0; i < response["results"].size(); ++i) {
                    auto& r = response["results"][i];
                    uint32_t rtype = r["record_type"].get<uint32_t>();
                    
                    std::cout << "   [" << (i + 1) << "] ";
                    std::cout << "Сходство: " << std::fixed << std::setprecision(4) << r["similarity"].get<float>();
                    
                    if (rtype == static_cast<uint32_t>(VectorDB::RecordType_CHAT)) {
                        std::cout << " | Тип: ЧАТ";
                        std::cout << " | Chat ID: " << r["chat_id"].get<std::string>();
                        std::cout << " | Msg ID: " << r["message_id"].get<std::string>();
                    } else if (rtype == static_cast<uint32_t>(VectorDB::RecordType_BOOK)) {
                        std::cout << " | Тип: КНИГА";
                        std::cout << " | Book ID: " << r["book_id"].get<std::string>();
                        std::cout << " | Chunk: " << r["chunk_index"].get<int32_t>();
                    } else {
                        std::cout << " | Тип: НЕИЗВЕСТНЫЙ (" << rtype << ")";
                    }
                    std::cout << std::endl;
                }
                std::cout << std::string(60, '-') << std::endl;
                
                res.set_content(response.dump(), "application/json");
                
            } catch (const std::exception& e) {
                json error;
                error["status"] = "error";
                error["message"] = e.what();
                res.set_content(error.dump(), "application/json");
            }
        });
        
        // ============================================================
        // ЭНДПОИНТ: Health check
        // ============================================================
        server.Get("/health", [](const httplib::Request&, httplib::Response& res) {
            json health;
            health["status"] = "ok";
            health["vectors_count"] = g_db ? g_db->size() : 0;
            health["dimensions"] = g_db ? g_db->get_dimensions() : 0;
            health["exchange_file"] = g_data_file;
            res.set_content(health.dump(), "application/json");
        });
        
        std::cout << "\n📡 HTTP сервер запущен на порту 8081" << std::endl;
        std::cout << "════════════════════════════════════════════════════════════" << std::endl;
        std::cout << "   POST /receive_flatbuffers  - Прием FlatBuffers (триггер)" << std::endl;
        std::cout << "   POST /vectorize_book       - Векторизация книги" << std::endl;
        std::cout << "   POST /search_similar       - Поиск похожих" << std::endl;
        std::cout << "   GET  /health               - Статус сервера" << std::endl;
        std::cout << "════════════════════════════════════════════════════════════" << std::endl;
        
        server.listen("0.0.0.0", 8081);
        
    } catch (const std::exception& e) {
        std::cout << "❌ Ошибка HTTP сервера: " << e.what() << std::endl;
    }
}

// ============================================================
// MAIN
// ============================================================
int main() {
    try {
        std::cout << "════════════════════════════════════════════════════════════" << std::endl;
        std::cout << "🚀 C++ Векторный сервер (FlatBuffers файловый режим)" << std::endl;
        std::cout << "════════════════════════════════════════════════════════════" << std::endl;
        
        const std::string filename = "vector_storage.bin";
        g_filename = filename;
        
        // Инициализируем векторизатор 
        std::cout << "\n🔧 Инициализация векторизатора..." << std::endl;
        VectorDB::VectorizerConfig vec_config;
        vec_config.model_path = "/home/den/project/vectorDB/models/all-MiniLM-L6-v2.onnx";
        vec_config.embedding_dim = 384;
        vec_config.max_length = 128;
        
        g_vectorizer = std::make_unique<VectorDB::OnnxVectorizer>(vec_config);
        g_current_dims = g_vectorizer->getEmbeddingDim();
        std::cout << "✅ Векторизатор готов (размерность: " << g_current_dims << ")" << std::endl;
        
        // Инициализируем БД
        std::cout << "\n🔧 Инициализация БД..." << std::endl;
        
        bool file_exists = false;
        {
            std::ifstream test_file(filename, std::ios::binary);
            file_exists = test_file.good();
        }
        
        if (file_exists) {
            g_current_dims = peek_flatbuffers_dimensions(filename);
        }
        
        g_db = std::make_unique<VectorDB::StorageManager>(g_current_dims);
        
        if (file_exists) {
            auto load_result = g_db->load_from_file(filename);
            if (load_result.has_value()) {
                g_next_auto_id.store(1000 + static_cast<int32_t>(g_db->size()));
                std::cout << "✅ База загружена: " << g_db->size() << " векторов" << std::endl;
            } else {
                std::cout << "⚠️ Ошибка загрузки базы: " << load_result.error() << std::endl;
                std::cout << "⚠️ Создана новая пустая база данных." << std::endl;
            }
        } else {
            std::cout << "⚠️ Файл не найден. Создана новая пустая база данных." << std::endl;
        }
        
        // ============================================================
        // ЗАПУСКАЕМ ФОНОВЫЙ ПОТОК ДЛЯ МОНИТОРИНГА ФАЙЛОВ
        // ============================================================
        std::cout << "\n🔧 Запуск файлового монитора..." << std::endl;
        std::thread file_monitor_thread(handleFlatBuffersFile);
        file_monitor_thread.detach();
        std::cout << "✅ Файловый монитор запущен" << std::endl;
        
        // ============================================================
        // ЗАПУСКАЕМ HTTP СЕРВЕР
        // ============================================================
        std::cout << "\n🔧 Запуск HTTP сервера..." << std::endl;
        std::thread server_thread(start_http_server);
        server_thread.detach();  
        std::this_thread::sleep_for(std::chrono::milliseconds(200));
        
        // ============================================================
        // ИНТЕРАКТИВНАЯ КОНСОЛЬ
        // ============================================================
        std::cout << "\n════════════════════════════════════════════════════════════" << std::endl;
        std::cout << "💻 Интерактивная консоль" << std::endl;
        std::cout << "════════════════════════════════════════════════════════════" << std::endl;
        std::cout << "Команды:" << std::endl;
        std::cout << "  A  - Добавить вектор вручную" << std::endl;
        std::cout << "  Q  - Поиск похожих векторов" << std::endl;
        std::cout << "  V  - Векторизация текста" << std::endl;
        std::cout << "  S  - Статистика БД" << std::endl;
        std::cout << "  C  - Очистить БД" << std::endl;
        std::cout << "  H  - Помощь" << std::endl;
        std::cout << "  E  - Выход" << std::endl;
        std::cout << "════════════════════════════════════════════════════════════" << std::endl;
        std::cout << "Введите команду: ";
        
        char command;
        while (std::cin >> command) {
            switch (command) {
                case 'A':
                case 'a': {
                    int32_t id;
                    size_t count;
                    
                    std::cout << "-> [Добавление] Введите ID и размерность вектора: ";
                    std::cin >> id >> count;
                    assert(count > 0);
                    
                    std::vector<float> vec(count);
                    std::cout << "-> [Добавление] Введите " << count << " чисел через пробел: ";
                    for (size_t i = 0; i < count; ++i) std::cin >> vec[i];
                    
                    std::lock_guard<std::mutex> lock(g_db_mutex);
                    
                    if (g_db->size() == 0 && count != g_current_dims) {
                        g_current_dims = count;
                        g_db = std::make_unique<VectorDB::StorageManager>(count);
                    }
                    
                    bool added = g_db->add_vector(id, std::move(vec));
                    if (!added) {
                        std::cout << "❌ Неправильный размер вектора!" << std::endl;
                    } else {
                        g_db->save_to_file(filename);
                        std::cout << "✅ Добавлено! Всего векторов: " << g_db->size() << std::endl;
                    }
                    break;
                }
                
                case 'Q':
                case 'q': {
                    size_t count, k;
                    
                    std::cout << "-> [Поиск] Введите размерность вектора: ";
                    std::cin >> count;
                    
                    std::vector<float> query(count);
                    std::cout << "-> [Поиск] Введите " << count << " чисел через пробел: ";
                    for (size_t i = 0; i < count; ++i) std::cin >> query[i];
                    
                    std::cout << "-> [Поиск] Сколько результатов вывести (K): ";
                    std::cin >> k;
                    
                    std::lock_guard<std::mutex> lock(g_db_mutex);
                    
                    if (count != g_db->get_dimensions()) {
                        std::cout << "❌ Несовпадение размерности!" << std::endl;
                    } else {
                        auto start = std::chrono::high_resolution_clock::now();
                        auto results = g_db->search_top_k(query, k);
                        auto end = std::chrono::high_resolution_clock::now();
                        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
                        
                        std::cout << "\n" << std::string(60, '─') << std::endl;
                        std::cout << "📊 Результаты поиска (Top-" << results.size() << ", " << duration << " мкс):" << std::endl;
                        std::cout << std::string(60, '─') << std::endl;
                        
                        for (size_t i = 0; i < results.size(); ++i) {
                            std::cout << "[" << (i + 1) << "] ";
                            std::cout << "Сходство: " << std::fixed << std::setprecision(4) << results[i].similarity;
                            std::cout << " | ID: " << results[i].id;
                            if (!results[i].chat_id.empty()) {
                                std::cout << " | Chat: " << results[i].chat_id;
                            }
                            if (!results[i].message_id.empty()) {
                                std::cout << " | Msg: " << results[i].message_id;
                            }
                            std::cout << std::endl;
                        }
                        std::cout << std::string(60, '─') << std::endl;
                    }
                    break;
                }
                
                case 'V':
                case 'v': {
                    std::cin.ignore();
                    std::string text;
                    std::cout << "-> [Векторизация] Введите текст: ";
                    std::getline(std::cin, text);
                    
                    if (!text.empty() && g_vectorizer) {
                        auto start = std::chrono::high_resolution_clock::now();
                        auto embedding = g_vectorizer->vectorize(text);
                        auto end = std::chrono::high_resolution_clock::now();
                        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
                        
                        std::cout << "✅ Вектор создан за " << duration << " мкс (размерность: " << embedding.size() << ")" << std::endl;
                        std::cout << "   Первые 5 значений: ";
                        for (size_t i = 0; i < std::min(size_t(5), embedding.size()); ++i) {
                            std::cout << std::fixed << std::setprecision(4) << embedding[i] << " ";
                        }
                        std::cout << "..." << std::endl;
                        
                        std::lock_guard<std::mutex> lock(g_db_mutex);
                        if (g_db->size() > 0) {
                            auto results = g_db->search_top_k(embedding, 3);
                            std::cout << "\n🔎 ТОП-3 похожих в базе:" << std::endl;
                            for (size_t i = 0; i < results.size(); ++i) {
                                std::cout << "   [" << (i + 1) << "] ";
                                std::cout << "Сходство: " << results[i].similarity;
                                if (!results[i].chat_id.empty()) {
                                    std::cout << " | Chat: " << results[i].chat_id;
                                }
                                if (!results[i].message_id.empty()) {
                                    std::cout << " | Msg: " << results[i].message_id;
                                }
                                std::cout << std::endl;
                            }
                        }
                    }
                    break;
                }
                
                case 'S':
                case 's': {
                    std::lock_guard<std::mutex> lock(g_db_mutex);
                    std::cout << "\n📊 СТАТИСТИКА БАЗЫ ДАННЫХ" << std::endl;
                    std::cout << std::string(40, '─') << std::endl;
                    std::cout << "   Всего векторов: " << g_db->size() << std::endl;
                    std::cout << "   Размерность: " << g_db->get_dimensions() << std::endl;
                    std::cout << "   Следующий ID: " << g_next_auto_id.load() << std::endl;
                    std::cout << "   Файл: " << g_filename << std::endl;
                    std::cout << std::string(40, '─') << std::endl;
                    break;
                }
                
                case 'C':
                case 'c': {
                    std::cout << "⚠️  Вы уверены? (y/n): ";
                    char confirm;
                    std::cin >> confirm;
                    if (confirm == 'y' || confirm == 'Y') {
                        std::lock_guard<std::mutex> lock(g_db_mutex);
                        g_db = std::make_unique<VectorDB::StorageManager>(g_current_dims);
                        g_next_auto_id.store(1000);
                        g_db->save_to_file(filename);
                        std::cout << "✅ БД очищена" << std::endl;
                    }
                    break;
                }
                
                case 'H':
                case 'h': {
                    std::cout << "\n📖 Доступные команды:" << std::endl;
                    std::cout << "  A - Добавить вектор вручную" << std::endl;
                    std::cout << "  Q - Поиск похожих векторов" << std::endl;
                    std::cout << "  V - Векторизация текста" << std::endl;
                    std::cout << "  S - Статистика БД" << std::endl;
                    std::cout << "  C - Очистить БД" << std::endl;
                    std::cout << "  H - Помощь" << std::endl;
                    std::cout << "  E - Выход" << std::endl;
                    break;
                }
                
                case 'E':
                case 'e': {
                    std::cout << "👋 Выход..." << std::endl;
                    g_server_running = false;
                    return 0;
                }
                
                default:
                    std::cout << "❌ Неизвестная команда. Используйте 'H' для помощи." << std::endl;
            }
            
            std::cout << "\nВведите команду (H - помощь): ";
        }
        
    } catch (const std::exception& e) {
        std::cout << "❌ Критическая ошибка: " << e.what() << std::endl;
        return 1;
    }
    
    g_server_running = false;
    return 0;
}