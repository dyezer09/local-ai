import express from 'express';
import fs from 'fs';
import path from 'path';
import crypto from 'crypto';
import { fileURLToPath } from 'url';
import fsp from 'fs/promises';
import Database from 'better-sqlite3';
import flatbuffers from 'flatbuffers';
import { VectorDB } from '../generated/schema.js';
import { getAllChats, getChatMessages, deleteChatFromDB, createChatInDB, addMessageToChat, updateSystemPrompt } from './db.js';

// импортируем логгеры
import { logger, dbLogger, llamaLogger, httpLogger } from './logger.js';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const app = express();
const PORT = 3000;

const url = 'http://192.168.0.101:8090/v1/chat/completions';

app.use(express.json());
app.use(express.static(path.join(__dirname)));

const dbPath = path.resolve(__dirname, '../db/chats_storage.db');
const db = new Database(dbPath);

const booksDbPath = path.resolve(__dirname, '../db/books.db');
const booksDb = new Database(booksDbPath);
db.pragma('journal_mode = DELETE');
booksDb.pragma('foreign_keys = ON');

logger.info('Databases connected', {
    chatsDb: dbPath,
    booksDb: booksDbPath
});

// Middleware для логирования всех http-запросов
app.use((req, res, next) => {
  const start = Date.now();
  
  // логируем после завершения запроса
  res.on('finish', () => {
    const duration = Date.now() - start;
    httpLogger.info(`${req.method} ${req.originalUrl}`, {
        method: req.method,
        url: req.originalUrl,
        status: res.statusCode,
        durationMs: duration,
        ip: req.ip
    });
  });
  
  next();
});

// получение пути к последнему чату
let chatsNames = getAllChats();
let chatHistory = [];

if (chatsNames.length > 0) {
    let lastChat = chatsNames[0];
    // console.log('Последний чат:', lastChat.title); 
    dbLogger.debug('Loaded last chat on startup', { 
      chatId: lastChat.id, 
      chatTitle: lastChat.title 
    });
    
    chatHistory = getChatMessages(lastChat.id);
    // console.log('История последнего чата:', chatHistory);
    dbLogger.debug('Loaded chat history on startup', { 
      chatId: lastChat.id, 
      messageCount: chatHistory.length 
    });
} else { 
    // console.log('В базе данных еще нет чатов. Ждем создания первого чата...');
    dbLogger.info('No chats in database on startup');
}

//обрабатываем запрос клиента (браузера) на получение всех имён сессий
app.get('/api/chats', (req, res) => {
    try {
        const chatsList = getAllChats();
        
        dbLogger.debug('Fetched all chats', { count: chatsList.length });
        
        const formattedChats = chatsList.map(chat => ({
            id: chat.id,                
            name: chat.title,               
            lastMessage: 'Нет сообщений', 
            unreadCount: 0,
            systemPrompt: chat.system_prompt
        }));

        res.json({ chats: formattedChats });
    } catch (error) {
        // console.error('Ошибка в маршруте /api/chats:', error.message); 
        dbLogger.error('Failed to get chats list', { 
          error: error.message, 
          stack: error.stack 
        });
        res.status(500).json({ error: 'Не удалось получить список чатов' });
    }
});

// удаление сесии
app.delete('/api/chat/:id', (req, res) => {
    try {
        const chatId = req.params.id;
        
        dbLogger.info('Attempting to delete chat', { chatId });
        
        const isDeleted = deleteChatFromDB(chatId);
        
        if (isDeleted) {
            dbLogger.info('Chat deleted successfully', { chatId });
            res.json({ success: true });
        } else {
            dbLogger.warn('Chat not found for deletion', { chatId });
            res.status(404).json({ error: 'Чат не найден в базе данных' });
        }
    } catch (error) {
        // console.error('Ошибка в маршруте DELETE /api/chat:', error.message); 
        dbLogger.error('Failed to delete chat', { 
          chatId: req.params.id,
          error: error.message, 
          stack: error.stack 
        });
        res.status(500).json({ error: 'Не удалось удалить чат' });
    }
});

//выдаёт содержимое по имени сессии  
app.get('/api/chat/:id', (req, res) => {
    try {
        const chatId = req.params.id;
        const messages = getChatMessages(chatId);
        
        dbLogger.debug('Fetched chat messages', { 
          chatId, 
          messageCount: messages.length 
        });
        
        // Форматируем для фронтенда role и clean_content
        const formattedMessages = messages.map(msg => ({
            role: msg.role,
            content: msg.clean_content,
            id: msg.id,
            parent_id: msg.parent_id
        }));
        
        res.json(formattedMessages);
    } catch (error) {
        // console.error('Ошибка при получении сообщений чата:', error.message); 
        dbLogger.error('Failed to get chat messages', { 
          chatId: req.params.id,
          error: error.message, 
          stack: error.stack 
        });
        res.status(500).json({ error: 'Не удалось загрузить историю сообщений' });
    }
});

// Функция для примерного подсчета токенов (1 токен ≈ 3.5 символов)
function estimateTokens(text) {
    return Math.ceil(text.length / 3.5);
}

// Функция подсчета токенов во всех сообщениях
function countMessagesTokens(messages) {
    let totalTokens = 0;
    for (const msg of messages) {
        totalTokens += estimateTokens(msg.role);
        totalTokens += estimateTokens(msg.content);
        // служебный 4 символа 
        totalTokens += 4;
    }
    return totalTokens;
}


// Валидация входящего сообщения
const validateMessage = (message, res) => {
    if (typeof messageCheck === 'function') {
        if (messageCheck(message, res) === 0) return false;
    }
    return true;
};

// gолучение parentId (последнее сообщение, если не указан)
const resolveParentId = (chatId, providedParentId) => {
    if (providedParentId) return providedParentId;
    
    const messages = getChatMessages(chatId);
    if (messages.length > 0) {
        return messages[messages.length - 1].id;
    }
    return null;
};

// сохранение сообщения пользователя
const saveUserMessage = (chatId, message, parentId) => {
    addMessageToChat(chatId, 'user', message, parentId, null);
    
    dbLogger.debug('User message saved', { 
        chatId, 
        messageLength: message.length 
    });
};

// получение системного промпта чата
const getSystemPrompt = (chatId) => {
    const chats = getAllChats();
    const currentChat = chats.find(c => c.id === chatId);
    return currentChat?.system_prompt;
};

// формирование сообщений для AI с RAG контекстом
const buildMessagesForAI = (chatId) => {
    const chatHistory = getChatMessages(chatId);
    
    const messagesForAI = chatHistory.map(msg => {
        const aiMsg = {
            role: msg.role,
            content: msg.clean_content
        };
        
        if (msg.role === 'user' && msg.rag_context) {
            aiMsg.content = `Context: ${msg.rag_context}\n\nQuestion: ${msg.clean_content}`;
        }
        
        return aiMsg;
    });
    
    return messagesForAI;
};

// добавление системного промпта в начало истории
const addSystemPromptToHistory = (messages, systemPrompt) => {
    if (systemPrompt) {
        messages.unshift({
            role: 'system',
            content: systemPrompt
        });
    }
    return messages;
};

// обрезка истории под лимит токенов
const trimHistoryForTokenLimit = (messages, maxTokens, chatId) => {
    const tokensBefore = countMessagesTokens(messages);
    
    let trimmedMessages = [...messages];
    let removedCount = 0;
    
    while (countMessagesTokens(trimmedMessages) > maxTokens && trimmedMessages.length > 1) {
        // системный промпт не удаляем
        if (trimmedMessages[0].role === 'system') {
            if (trimmedMessages.length > 2) {
                trimmedMessages.splice(1, 1); // удаляем второе сообщение
                removedCount++;
            } else {
                break; 
            }
        } else {
            trimmedMessages.shift(); // удаляем самое старое
            removedCount++;
        }
    }
    
    if (removedCount > 0) {
        const tokensAfter = countMessagesTokens(trimmedMessages);
        llamaLogger.debug('History trimmed for token limit', {
            chatId,
            removedMessages: removedCount,
            remainingMessages: trimmedMessages.length,
            estimatedTokensBefore: tokensBefore,
            estimatedTokensAfter: tokensAfter,
            maxTokens: maxTokens
        });
    }
    
    return trimmedMessages;
};

// отправка запроса к Llama 
const callLlamaAPI = async (url, payload) => {
    try {
        const response = await fetch(url, { 
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload),
            signal: AbortSignal.timeout(30000) // таймаут 30 секунд
        });
        
        if (!response.ok) {
            const errorBody = await response.text();
            throw { 
                status: response.status, 
                statusText: response.statusText, 
                errorBody,
                isApiError: true
            };
        }
        
        const data = await response.json();
        return data;
    } catch (error) {
        // Логируем ошибку и пробрасываем дальше
        llamaLogger.error('Llama API call failed', {
            error: error.message,
            url: url
        });
        throw error;
    }
};

// логирование ответа Llama
const logLlamaResponse = (chatId, payload, aiResponse, usage, duration) => {
    const promptTokens = usage.prompt_tokens || 0;
    const completionTokens = usage.completion_tokens || 0;
    const totalTokens = usage.total_tokens || (promptTokens + completionTokens);
    
    llamaLogger.info('Llama response received', {
        chatId,
        model: payload.model,
        promptTokens,
        completionTokens,
        totalTokens,
        durationMs: duration,
        responseLength: aiResponse.length
    });
    
    return { promptTokens, completionTokens, totalTokens };
};


// получение текста чанка книги по book_id и chunk_index
const getBookChunk = (bookId, chunkIndex) => {
    try {
        const chunk = booksDb.prepare(
            'SELECT c.*, b.title as book_title FROM chunks c JOIN books b ON c.book_id = b.id WHERE c.book_id = ? AND c.chunk_index = ?'
        ).get(bookId, chunkIndex);
        
        return chunk || null;
    } catch (error) {
        logger.error('Failed to get book chunk', { bookId, chunkIndex, error: error.message });
        return null;
    }
};

//отправка текстов напрямую в C++ сервер
async function sendTextsToCpp(chatId, messageId, userMessage, aiResponse) {
    const builder = new flatbuffers.Builder(4096);
    
    // Создаем строки
    const chatIdOffset = builder.createString(String(chatId));
    const messageIdOffset = builder.createString(String(messageId));
    
    // Получаем эмбеддинги (заглушка)
    const userEmbedding = getEmbedding(userMessage);
    const aiEmbedding = getEmbedding(aiResponse);
    
    // Создаем документы
    const userDoc = createDocument(
        builder,
        messageId + '_user',
        userEmbedding,
        VectorDB.RecordType.CHAT,
        String(chatId),
        String(messageId),
        '',
        0
    );
    
    const aiDoc = createDocument(
        builder,
        messageId + '_ai',
        aiEmbedding,
        VectorDB.RecordType.CHAT,
        String(chatId),
        String(messageId),
        '',
        0
    );
    
    // Создаем вектор документов
    const documentsOffset = VectorDB.Storage.createDocumentsVector(
        builder,
        [userDoc, aiDoc]
    );
    
    // Строим Storage
    VectorDB.Storage.startStorage(builder);
    VectorDB.Storage.addDocuments(builder, documentsOffset);
    const storage = VectorDB.Storage.endStorage(builder);
    
    builder.finish(storage);
    
    const buffer = builder.asUint8Array();
    
    // запись в файл
    const filePath = path.join(__dirname, '../../data/chat_data.bin');
    const tempPath = filePath + '.tmp';
    const readyPath = filePath + '.ready';
    
    // Создаем директорию если её нет
    const dir = path.dirname(filePath);
    if (!fs.existsSync(dir)) {
        fs.mkdirSync(dir, { recursive: true });
    }
    
    // Атомарная запись: сначала во временный файл
    fs.writeFileSync(tempPath, Buffer.from(buffer));
    // Переименовываем (атомарная операция)
    fs.renameSync(tempPath, filePath);
    // Создаем файл-индикатор, что данные готовы
    fs.writeFileSync(readyPath, Date.now().toString());
    
    console.log('\n' + '════════════════════════════════════════════════════════════');
    console.log('📡 C++ ОТПРАВКА: FlatBuffers через файл');
    console.log('════════════════════════════════════════════════════════════');
    console.log(`📤 Данные записаны в файл: ${filePath}`);
    console.log(`   Размер: ${buffer.length} байт`);
    console.log(`   Chat ID: ${chatId}`);
    console.log(`   Message ID: ${messageId}`);
    console.log(`   Документов: 2 (user + ai)`);
    console.log(`   Файл-индикатор: ${readyPath}`);
    console.log('────────────────────────────────────────────────────────────');

    try {
        
        console.log('✅ Файл успешно записан, C++ может его читать');
        console.log('════════════════════════════════════════════════════════════\n');
        
        return { success: true, filePath };
        
    } catch (error) {
        console.error('❌ ОШИБКА:', error.message);
        return null;
    }
}

// функция создания документа
function createDocument(builder, id, values, recordType, chatId, messageId, bookId, chunkIndex) {
    const idOffset = builder.createString(String(id));
    const chatIdOffset = builder.createString(String(chatId || ''));
    const messageIdOffset = builder.createString(String(messageId || ''));
    const bookIdOffset = builder.createString(String(bookId || ''));
    
    const valuesOffset = VectorDB.Document.createValuesVector(
        builder,
        values
    );
    
    VectorDB.Document.startDocument(builder);
    VectorDB.Document.addId(builder, idOffset);
    VectorDB.Document.addValues(builder, valuesOffset);
    VectorDB.Document.addRecordType(builder, recordType);
    VectorDB.Document.addChatId(builder, chatIdOffset);
    VectorDB.Document.addMessageId(builder, messageIdOffset);
    VectorDB.Document.addBookId(builder, bookIdOffset);
    VectorDB.Document.addChunkIndex(builder, chunkIndex || 0);
    
    return VectorDB.Document.endDocument(builder);
}

// заглушка для эмбеддингов
function getEmbedding(text) {
    const embeddingSize = 768;
    return Array.from({ length: embeddingSize }, () => Math.random() * 2 - 1);
}





// поиск похожих сообщений через C++ сервер
async function searchSimilarMessages(text, topK = 3) {
    const cppServerUrl = 'http://localhost:8081/search_similar';
    
    const payload = {
        text: text,
        top_k: topK
    };

    console.log('\n' + '════════════════════════════════════════════════════════════');
    console.log('🔍 ПОИСК ПОХОЖИХ: Отправка запроса в C++ сервер');
    console.log('════════════════════════════════════════════════════════════');
    console.log(`📤 Запрос: "${text.substring(0, 100)}${text.length > 100 ? '...' : ''}"`);

    try {
        const response = await fetch(cppServerUrl, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload)
        });
        
        const result = await response.json();
        
        if (result.status === 'success') {
            console.log('✅ Найдены похожие:');
            
            const filtered = result.results.filter(r => r.similarity < 0.999);
            
            for (let i = 0; i < Math.min(filtered.length, topK); i++) {
                const r = filtered[i];
                const recordType = r.record_type;
                // для чата
                if (recordType === 1) {
                    const cleanMessageId = String(r.message_id).replace(/_ai$/, '').replace(/_user$/, '');
                    
                    console.log(`   [${i + 1}] Сходство: ${r.similarity.toFixed(4)} | ТИП: ЧАТ | Chat: ${r.chat_id} | Msg: ${cleanMessageId}`);
                    
                    let foundMsg = null;
                    if (r.chat_id && r.message_id) {
                        const chatMessages = getChatMessages(r.chat_id);
                        if (chatMessages) {
                            foundMsg = chatMessages.find(m => String(m.id) === cleanMessageId);
                        }
                    }
                    
                    if (foundMsg) {
                        const role = foundMsg.role === 'user' ? '👤' : '🤖';
                        const preview = foundMsg.clean_content.substring(0, 100);
                        console.log(`      ${role} "${preview}${foundMsg.clean_content.length > 100 ? '...' : ''}"`);
                    } else {
                        console.log(`      [текст не найден в chats_storage.db]`);
                    }
                    
                } else if (recordType === 2) {
                    //для книги
                    const chunk = getBookChunk(r.book_id, r.chunk_index);
                    
                    console.log(`   [${i + 1}] Сходство: ${r.similarity.toFixed(4)} | ТИП: КНИГА | Book: ${r.book_id} | Chunk: ${r.chunk_index}`);
                    
                    if (chunk) {
                        const preview = chunk.text.substring(0, 100);
                        console.log(`      📖 "${chunk.book_title}"`);
                        console.log(`      "${preview}${chunk.text.length > 100 ? '...' : ''}"`);
                    } else {
                        console.log(`      [чанк не найден в books.db]`);
                    }
                    
                } else {
                    console.log(`   [${i + 1}] Сходство: ${r.similarity.toFixed(4)} | ТИП: НЕИЗВЕСТНЫЙ (${recordType})`);
                }
            }
            
            console.log('════════════════════════════════════════════════════════════\n');
            
            return { ...result, results: filtered.slice(0, topK) };
        } else {
            console.log('❌ Ошибка поиска:', result.message);
            console.log('════════════════════════════════════════════════════════════\n');
            return null;
        }
        
    } catch (error) {
        console.error('❌ ОШИБКА подключения к C++ серверу:', error.message);
        console.log('════════════════════════════════════════════════════════════\n');
        return null;
    }
}



// сохранение ответа ии и получение ID сообщения пользователя
const saveAIResponse = (chatId, aiResponse) => {
    const updatedMessages = getChatMessages(chatId);
    const userMessageId = updatedMessages[updatedMessages.length - 1].id;
    
    addMessageToChat(chatId, 'assistant', aiResponse, userMessageId, null);
    
    dbLogger.debug('AI response saved', { 
        chatId, 
        responseLength: aiResponse.length 
    });
    
    return userMessageId;
};

// логирование ошибок Llama
const logLlamaError = (chatId, error) => {
    if (error.message?.includes('ECONNREFUSED') || error.message?.includes('fetch failed')) {
        llamaLogger.fatal('Cannot connect to Llama server', {
            chatId,
            apiUrl: url,
            error: error.message
        });
    } else if (error.message?.includes('Timeout') || error.name === 'AbortError') {
        llamaLogger.error('Llama request timed out', {
            chatId,
            error: error.message,
            stack: error.stack
        });
    } else {
        llamaLogger.error('Failed to get Llama response', {
            chatId,
            error: error.message,
            stack: error.stack
        });
    }
};


// обработчик для нового сообщения
app.post('/api/chat/:id/message', async (req, res) => {
    try {
        const chatId = req.params.id;
        const { message, parentId } = req.body;

        // валидация
        if (!validateMessage(message, res)) return;

        // определяем parentId
        const actualParentId = resolveParentId(chatId, parentId);

        // сохраняем сообщение пользователя
        saveUserMessage(chatId, message, actualParentId);

        // получаем системный промпт
        const systemPrompt = getSystemPrompt(chatId);

        // формируем историю для ии
        let messagesForAI = buildMessagesForAI(chatId);
        
        // добавляем системный промпт
        messagesForAI = addSystemPromptToHistory(messagesForAI, systemPrompt);

        // логируем подготовку запроса
        const tokensBefore = countMessagesTokens(messagesForAI);
        llamaLogger.debug('Preparing Llama request', {
            chatId,
            messageCount: messagesForAI.length,
            estimatedTokensBefore: tokensBefore,
            hasSystemPrompt: !!systemPrompt
        });

        // обрезаем историю под лимит токенов
        const MAX_TOKENS = 6000;
        const trimmedMessages = trimHistoryForTokenLimit(messagesForAI, MAX_TOKENS, chatId);

        // формируем payload
        const payload = { 
            model: 'my-model', 
            messages: trimmedMessages 
        };

        // отправляем запрос к Llama
        const llamaStartTime = Date.now();
        const data = await callLlamaAPI(url, payload);
        const llamaDuration = Date.now() - llamaStartTime;

        // извлекаем ответ нейросети
        const aiResponse = data.choices[0].message.content;

        // логируем ответ
        const { promptTokens, completionTokens, totalTokens } = logLlamaResponse(
            chatId, payload, aiResponse, data.usage || {}, llamaDuration
        );

        // сохраняем ответ ии и получаем ID сообщения пользователя
        const userMessageId = saveAIResponse(chatId, aiResponse);

        // отправляем тексты с метаданными в C++ сервер (асинхронно)
        sendTextsToCpp(chatId, userMessageId, message, aiResponse);

        // ищем похожие сообщения для контекста
        const similarResults = await searchSimilarMessages(message, 3);

        // если нашли похожие — выводим их в консоль
        if (similarResults && similarResults.results && similarResults.results.length > 0) {
            console.log('\n📋 ПОХОЖИЕ СООБЩЕНИЯ ДЛЯ КОНТЕКСТА:');
            for (let i = 0; i < similarResults.results.length; i++) {
                const r = similarResults.results[i];
                const recordType = r.record_type;
                
                if (recordType === 1) {
                    // ─── ЧАТ ───────────────────────────────────
                    const cleanMessageId = String(r.message_id).replace(/_ai$/, '').replace(/_user$/, '');
                    
                    let foundMsg = null;
                    const chatMessages = getChatMessages(r.chat_id);
                    if (chatMessages) {
                        foundMsg = chatMessages.find(m => String(m.id) === cleanMessageId);
                    }
                    
                    if (foundMsg) {
                        const role = foundMsg.role === 'user' ? '👤' : '🤖';
                        const preview = foundMsg.clean_content.substring(0, 150);
                        console.log(`   ${i + 1}. [${r.similarity.toFixed(3)}] ${role} ${preview}${foundMsg.clean_content.length > 150 ? '...' : ''}`);
                    } else {
                        console.log(`   ${i + 1}. [${r.similarity.toFixed(3)}] Chat: ${r.chat_id} | Msg: ${cleanMessageId} | [текст не найден]`);
                    }
                    
                } else if (recordType === 2) {
                    // ─── КНИГА ─────────────────────────────────
                    const chunk = getBookChunk(r.book_id, r.chunk_index);
                    
                    if (chunk) {
                        const preview = chunk.text.substring(0, 150);
                        console.log(`   ${i + 1}. [${r.similarity.toFixed(3)}] 📖 "${chunk.book_title}"`);
                        console.log(`      ${preview}${chunk.text.length > 150 ? '...' : ''}`);
                    } else {
                        console.log(`   ${i + 1}. [${r.similarity.toFixed(3)}] Book: ${r.book_id} | Chunk: ${r.chunk_index} | [чанк не найден]`);
                    }
                    
                } else {
                    console.log(`   ${i + 1}. [${r.similarity.toFixed(3)}] Тип: НЕИЗВЕСТНЫЙ (${recordType})`);
                }
            }
            console.log('────────────────────────────────────────────────────────\n');
        }

        // отправляем ответ клиенту
        res.json({ 
            role: 'assistant', 
            content: aiResponse,
            usage: {
                prompt_tokens: promptTokens,
                completion_tokens: completionTokens,
                total_tokens: totalTokens
            }
        });
        
    } catch (error) {
        // логируем ошибку
        logLlamaError(req.params.id, error);
        
        // отправляем ошибку клиенту
        res.status(500).json({ error: 'Ошибка при получении ответа от нейросети' });
    }
});

// проверка и валидация сообщения 
function messageCheck(message, res) {
    //запрет отправки пустого сообщения  
    if (!message || message.trim() === '') {
        void res.status(400).json({ 
            error: 'Сообщение не может быть пустым' 
        });
        return 0;
    }
    // завершение сессии
    if (message.trim().toLowerCase() === 'exit') {
        void res.json({ status: 'server_stopping' });
        setTimeout(saveLogsAndExit, 1000);
        return 0 
    }
    return 1; 
}


// Создание нового чата
app.post('/api/chat/new', (req, res) => {
    try {
        const now = new Date();
        const day = now.getDate().toString().padStart(2, '0');
        const month = (now.getMonth() + 1).toString().padStart(2, '0');
        const year = now.getFullYear();
        const hours = now.getHours().toString().padStart(2, '0');
        const minutes = now.getMinutes().toString().padStart(2, '0');
  
        const formattedDate = `${day}.${month}.${year}_${hours}-${minutes}`;
        const chatName = `log_${formattedDate}`; 

        const chatId = crypto.randomUUID();
        const systemPrompt = req.body?.systemPrompt || null;
       
        createChatInDB(chatId, chatName, systemPrompt);
        
        dbLogger.info('New chat created', { 
          chatId, 
          chatName,
          hasSystemPrompt: !!systemPrompt 
        });
        
        res.json({ id: chatId, name: chatName });

    } catch (error) {
        // console.error('Ошибка создания чата в SQLite:', error); // 
        dbLogger.error('Failed to create chat', { 
          error: error.message, 
          stack: error.stack 
        });
        res.status(500).json({ error: 'Не удалось создать чат' });
    }
});

// Обновление системного промпта
app.put('/api/chat/:id/system-prompt', (req, res) => {
    try {
        const chatId = req.params.id;
        const { systemPrompt } = req.body;
        
        updateSystemPrompt(chatId, systemPrompt);
        
        dbLogger.info('System prompt updated', { 
          chatId,
          promptLength: systemPrompt ? systemPrompt.length : 0 
        });
        
        res.json({ success: true });
    } catch (error) {
        // console.error('Ошибка обновления системного промпта:', error);
        dbLogger.error('Failed to update system prompt', { 
          chatId: req.params.id,
          error: error.message, 
          stack: error.stack 
        });
        res.status(500).json({ error: 'Не удалось обновить системный промпт' });
    }
});

// обработчик для приёма логов с фронтенда
app.post('/api/log', (req, res) => {
    try {
        const { level, message, meta } = req.body;
        
        // принимаем только известные эти уровни
        const validLevels = ['error', 'warn', 'info', 'debug'];
        const logLevel = validLevels.includes(level) ? level : 'info';
        
        // записываем в winston через основной логер с меткой module: 'frontend'
        logger.log(logLevel, message, { 
            module: 'frontend',
            ...meta 
        });
        
        res.json({ success: true });
    } catch (error) {
        res.status(500).json({ success: false });
    }
});



// Обработка необработанных ошибок
process.on('uncaughtException', (error) => {
    logger.error('Uncaught Exception', { 
        error: error.message, 
        stack: error.stack 
    });
    // Не завершаем процесс
});

process.on('unhandledRejection', (reason, promise) => {
    logger.error('Unhandled Rejection', { 
        reason: reason?.message || reason,
        stack: reason?.stack 
    });
    // Не завершаем процесс
});

app.listen(PORT, () => {
    // console.log(`--- Бот запущен ---`); 
    // console.log(`Откройте страницу в браузере: http://localhost:${PORT}`); 
    
    logger.info('Server started', {
      port: PORT,
      nodeVersion: process.version,
      dbPath: dbPath,
      llamaApiUrl: url,
      env: process.env.NODE_ENV || 'development'
    });
    logger.info(`Open in browser: http://localhost:${PORT}`);
});