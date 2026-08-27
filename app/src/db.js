import Database from 'better-sqlite3';
import { fileURLToPath } from 'url';
import path from 'path';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const dbPath = path.resolve(__dirname, '..', 'db', 'chats_storage.db');
const db = new Database(dbPath);

db.pragma('foreign_keys = ON');

//табло 1 - Сессии (общая информация о чатах)
db.exec(`
  CREATE TABLE IF NOT EXISTS sessions (
    id TEXT PRIMARY KEY,               -- Уникальный ID чата (UUID)
    title TEXT NOT NULL,               -- Название чата
    system_prompt TEXT,                -- Системный промпт для Llama
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP  -- Дата и время создания
  )
`);

//табло 2 - Сообщения (все реплики с поддержкой ветвления)
db.exec(`
  CREATE TABLE IF NOT EXISTS messages (
    id INTEGER PRIMARY KEY AUTOINCREMENT, -- Уникальный числовой ID сообщения
    session_id TEXT NOT NULL,             -- К какой сессии относится сообщение
    parent_id INTEGER,                    -- ID сообщения, которое шло перед этим (NULL для первого)
    role TEXT NOT NULL,                   -- Роль: 'user', 'assistant' или 'system'
    clean_content TEXT NOT NULL,          -- Чистый текст сообщения для отображения
    rag_context TEXT,                     -- Скрытые куски документов из базы знаний (только для user)
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP, -- Точное время отправки
    
    FOREIGN KEY (session_id) REFERENCES sessions(id) ON DELETE CASCADE,
    FOREIGN KEY (parent_id) REFERENCES messages(id) ON DELETE SET NULL
  )
`);

//читает все сессии (названия и время создания)
export function getAllChats() {
  const stmt = db.prepare(`
    SELECT id, title, system_prompt, created_at 
    FROM sessions 
    ORDER BY created_at DESC
  `);
  return stmt.all();
}

//получает все сообщения сессии по id (отсортировано по времени)
export function getChatMessages(chatId) {
  const stmt = db.prepare(`
    SELECT id, parent_id, role, clean_content, rag_context, created_at 
    FROM messages 
    WHERE session_id = ? 
    ORDER BY created_at ASC
  `);
  return stmt.all(chatId);
}

//добавление сообщения в датабазу
export function addMessageToChat(chatId, role, content, parentId = null, ragContext = null) {
  const insertStmt = db.prepare(`
    INSERT INTO messages (session_id, parent_id, role, clean_content, rag_context)
    VALUES (?, ?, ?, ?, ?)
  `);
  insertStmt.run(chatId, parentId, role, content, ragContext);
}

// создание новой сессии
export function createChatInDB(chatId, chatName, systemPrompt = null) {
  const stmt = db.prepare(`
    INSERT INTO sessions (id, title, system_prompt) 
    VALUES (?, ?, ?)
  `);
  
  stmt.run(chatId, chatName, systemPrompt);
}

//удаления сессии
export function deleteChatFromDB(chatId) {
  const stmt = db.prepare('DELETE FROM sessions WHERE id = ?');
  const result = stmt.run(chatId);
  return result.changes > 0;
}

//обновление системного промпта
export function updateSystemPrompt(chatId, systemPrompt) {
  const stmt = db.prepare('UPDATE sessions SET system_prompt = ? WHERE id = ?');
  stmt.run(systemPrompt, chatId);
}

export default db;