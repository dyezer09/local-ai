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
std::atomic<bool> g_server_running = true;
std::mutex g_db_mutex;

std::unique_ptr<VectorDB::StorageManager> g_db;
std::unique_ptr<VectorDB::OnnxVectorizer> g_vectorizer;
std::string g_filename = "vector_storage.bin";
size_t g_current_dims = 384;

// Пути для файлового обмена с Node.js
std::string g_data_file = "/home/den/project/vectorDB/data/chat_data.bin";// менять при переходе на ноут
std::string g_ready_file = g_data_file + ".ready";



// импорт векторов, которые от Node.js
bool processFlatBuffersFromFile(const std::string& filename) {
    StorageData storage;
    std::string error_msg;
    
    // Читаем FlatBuffers из файла 
    if (!readFlatBuffersFromFile(filename, storage, error_msg)) {
        std::cerr << "❌ Ошибка чтения FlatBuffers: " << error_msg << std::endl;
        return false;
    }
    
    if (storage.vectors.empty()) {
        std::cerr << "⚠️ Нет данных в файле" << std::endl;
        return false;
    }
    
    std::cout << "🔍 Из файла получено векторов: " << storage.vectors.size() << std::endl;
    printStorageData(storage);
    
    std::lock_guard<std::mutex> lock(g_db_mutex);
    
    // проверяем размерность по первому вектору
    size_t embedding_dim = storage.vectors[0].embedding.size();
    if (embedding_dim != g_db->get_dimensions()) {
        std::cerr << "❌ Несовпадение размерности! База ожидает: " 
                  << g_db->get_dimensions() << ", файл принес: " << embedding_dim << std::endl;
        return false;
    }
    
    int added_count = 0;
    for (const auto& vec_data : storage.vectors) {
        int32_t new_id = g_next_auto_id.fetch_add(1);
        
        bool added = g_db->add_vector(
            new_id,
            vec_data.embedding,
            vec_data.chat_id,
            vec_data.message_id,
            vec_data.book_id,
            vec_data.chunk_index,
            vec_data.record_type
        );
        
        if (added) {
            added_count++;
        }
    }
    
    // Сохраняем базу на диск если что-то добавилось
    if (added_count > 0) {
        auto save_result = g_db->save_to_file(g_filename);
        if (save_result.has_value()) {
            std::cout << "💾 База успешно обновлена и сохранена на диск." << std::endl;
        }
    }
    
    std::cout << "✅ Успешно импортировано: " << added_count << " векторов. Всего в БД: " << g_db->size() << std::endl;
    return added_count > 0;
}


// Мониторинг файлов в фоновом потоке
void handleFlatBuffersFile() {
    std::cout << "📂 Мониторинг файла: " << g_data_file << std::endl;
    std::cout << "📂 Индикатор готовности: " << g_ready_file << std::endl;
    
    // создаем директорию если её нет
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
                // читаем и обрабатываем данные 
                bool success = processFlatBuffersFromFile(g_data_file);
                
                removeFile(g_ready_file);
                removeFile(g_data_file);
                
                if (success) {
                    processed_count++;
                    std::cout << "🗑️  Файлы удалены. Обработано: " << processed_count << std::endl;
                }
                
            } catch (const std::exception& e) {
                std::cerr << "❌ Ошибка обработки файла: " << e.what() << std::endl;
                removeFile(g_ready_file);
                removeFile(g_data_file);
            }
        }
        
        //  задержка
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
}


//  HTTP сервер 
void start_http_server() {
    try {
        httplib::Server server;
        

        // Прием текстов с векторизацией 
        server.Post("/receive_texts", [](const httplib::Request& req, httplib::Response& res) {
            try {
                json body = json::parse(req.body);
                
                // проверка наличие массива texts
                if (!body.contains("texts") || !body["texts"].is_array()) {
                    json err;
                    err["status"] = "error";
                    err["message"] = "Missing 'texts' array in request";
                    res.set_content(err.dump(), "application/json");
                    res.status = 400;
                    return;
                }
                
                // извлечение данных
                std::vector<std::string> texts;
                for (const auto& t : body["texts"]) {
                    texts.push_back(t.get<std::string>());
                }
                
                std::string chat_id = body.value("chat_id", "unknown");
                std::string message_id = body.value("message_id", "unknown");
                std::string timestamp = body.value("timestamp", "");
                
                // логи
                std::cout << "\n══════════════════════════════════════════════════════" << std::endl;
                std::cout << "📥 ПОЛУЧЕН ТЕКСТ ДЛЯ ВЕКТОРИЗАЦИИ" << std::endl;
                std::cout << "   Chat ID: " << chat_id << std::endl;
                std::cout << "   Message ID: " << message_id << std::endl;
                std::cout << "   Timestamp: " << timestamp << std::endl;
                std::cout << "   Количество текстов: " << texts.size() << std::endl;
                
                for (size_t i = 0; i < texts.size(); i++) {
                    std::cout << "   Текст " << i + 1 << ": " 
                            << texts[i].substr(0, 100) 
                            << (texts[i].length() > 100 ? "..." : "") 
                            << " (" << texts[i].length() << " символов)" << std::endl;
                }
                std::cout << "══════════════════════════════════════════════════════" << std::endl;
                
                // Проверяем готовность векторизатора и БД
                if (!g_vectorizer || !g_db) {
                    json err;
                    err["status"] = "error";
                    err["message"] = "Vectorizer or DB not ready";
                    res.set_content(err.dump(), "application/json");
                    res.status = 503;
                    return;
                }
                
                // векторизуем тексты через ONNX Runtime 
                std::cout << "   ⏳ Векторизация " << texts.size() << " текстов..." << std::endl;
                auto start_vec = std::chrono::high_resolution_clock::now();
                auto embeddings = g_vectorizer->vectorizeBatch(texts);
                auto end_vec = std::chrono::high_resolution_clock::now();
                auto vec_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end_vec - start_vec).count();
                std::cout << "   ⚡ Векторизация: " << vec_ms << " мс" << std::endl;
                
                // сохраняем в БД (потом чё-то сделать т.к не очень быстро из-за mutex)
                std::lock_guard<std::mutex> lock(g_db_mutex);
                
                json positions = json::array();
                int vectors_added = 0;
                
                for (size_t i = 0; i < embeddings.size(); ++i) {
                    if (embeddings[i].empty()) continue;
                    
                    int32_t new_id = g_next_auto_id.fetch_add(1);
                    
                    bool added = g_db->add_vector(
                        new_id, 
                        embeddings[i],
                        chat_id,        
                        message_id,     
                        "",                       // book_id для чата пустой
                        static_cast<int32_t>(i),  // chunk_index
                        VectorDB::RecordType_CHAT // Логически корректный тип записи
                    );
                    
                    if (added) {
                        json pos;
                        pos["chunkIndex"] = i;
                        pos["vectorId"] = new_id;
                        pos["position"] = static_cast<int32_t>(g_db->size() - 1);
                        positions.push_back(pos);
                        vectors_added++;
                    }
                }
                
                // обновление базы
                long long save_ms = 0;
                if (vectors_added > 0) {
                    std::cout << "   💾 Сохранение в базу данных..." << std::endl;
                    auto start_save = std::chrono::high_resolution_clock::now();
                    g_db->save_to_file(g_filename);
                    auto end_save = std::chrono::high_resolution_clock::now();
                    save_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end_save - start_save).count();
                    std::cout << "   ⚡ Сохранение: " << save_ms << " мс" << std::endl;
                }
                
                std::cout << "   📊 Всего в БД: " << g_db->size() << " векторов" << std::endl;
                
                // форматируем ответт
                json response;
                response["status"] = "success";
                response["message"] = "Texts vectorized and stored";
                response["chat_id"] = chat_id;
                response["vectors_added"] = vectors_added;
                response["total_in_db"] = g_db->size();
                response["positions"] = positions;
                response["vectorization_ms"] = vec_ms;
                response["save_ms"] = save_ms;
                
                res.set_content(response.dump(), "application/json");
                
            } catch (const std::exception& e) {
                std::cerr << "💥 /receive_texts error: " << e.what() << std::endl;
                json err;
                err["status"] = "error";
                err["message"] = e.what();
                res.set_content(err.dump(), "application/json");
                res.status = 500;
            }
        });


        // приём FlatBuffers 
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
        
        //  векторизация книги 
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
        

        // поиск похожих 
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
        
        // состояние сервера
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


int main() {
    try {
        std::cout << "════════════════════════════════════════════════════════════" << std::endl;
        std::cout << "🚀 C++ Векторный сервер (FlatBuffers файловый режим)" << std::endl;
        std::cout << "════════════════════════════════════════════════════════════" << std::endl;
        
        const std::string filename = "vector_storage.bin";
        g_filename = filename;
        
        // инициализируем векторизатор 
        std::cout << "\n🔧 Инициализация векторизатора..." << std::endl;
        VectorDB::VectorizerConfig vec_config;
        vec_config.model_path = "/home/den/project/vectorDB/models/all-MiniLM-L6-v2.onnx";
        vec_config.embedding_dim = 384;
        vec_config.max_length = 128;
        
        g_vectorizer = std::make_unique<VectorDB::OnnxVectorizer>(vec_config);
        g_current_dims = g_vectorizer->getEmbeddingDim();
        std::cout << "✅ Векторизатор готов (размерность: " << g_current_dims << ")" << std::endl;
        
        // инициализируем БД
        std::cout << "\n📂 Инициализация БД..." << std::endl;
        
        // создаётся база с размерность 384
        g_db = std::make_unique<VectorDB::StorageManager>(g_current_dims);
        
        // Просто пробуем загрузить файл. 
        // Если файл есть — g_db сам прочитает его и обновит g_current_dims, если там 768 или что-то еще.
        if (std::filesystem::exists(filename)) {
            auto load_result = g_db->load_from_file(filename);
            if (load_result.has_value()) {
                // Синхронизируем глобальную переменную с тем, что реально прочиталось из файла
                g_current_dims = g_db->get_dimensions(); 
                g_next_auto_id.store(1000 + static_cast<int32_t>(g_db->size()));
                std::cout << "✅ База загружена: " << g_db->size() << " векторов (размерность: " << g_current_dims << ")" << std::endl;
            } else {
                std::cout << "⚠️ Ошибка загрузки базы: " << load_result.error() << std::endl;
                std::cout << "ℹ️ Создана новая пустая база данных." << std::endl;
            }
        } else {
            std::cout << "ℹ️ Файл не найден. Создана новая пустая база данных." << std::endl;
        }
        

        std::cout << "\n🔧 Запуск файлового монитора..." << std::endl;
        std::thread file_monitor_thread(handleFlatBuffersFile);
        file_monitor_thread.detach();
        std::cout << "✅ Файловый монитор запущен" << std::endl;
        

        // запуск http сервера
        std::cout << "\n🔧 Запуск HTTP сервера..." << std::endl;
        start_http_server(); 
        
    } catch (const std::exception& e) {
        std::cout << "❌ Критическая ошибка: " << e.what() << std::endl;
        return 1;
    }
    
    g_server_running = false;
    return 0;
}