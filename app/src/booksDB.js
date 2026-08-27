// booksDb.js — работа с базой данных книг (ESM)
import Database from 'better-sqlite3';
import path from 'path';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const dbPath =path.resolve(__dirname, '..', 'db',  'books.db');

// Инициализация базы данных
const db = new Database(dbPath);
db.pragma('journal_mode = DELETE'); // обычный режим журналирования
db.pragma('foreign_keys = ON'); // включаем внешние ключи

// Создание таблиц
db.exec(`
    CREATE TABLE IF NOT EXISTS books (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        title TEXT NOT NULL,
        filename TEXT,
        author TEXT,
        loaded_at DATETIME DEFAULT CURRENT_TIMESTAMP,
        UNIQUE(title)  -- не даём загрузить одну книгу дважды
    );
    
    CREATE TABLE IF NOT EXISTS chunks (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        book_id INTEGER NOT NULL REFERENCES books(id) ON DELETE CASCADE,
        chunk_index INTEGER NOT NULL,  -- порядковый номер куска в книге
        text TEXT NOT NULL,
        token_estimate INTEGER,  -- примерное количество токенов
        chapter TEXT,  -- название главы, если определили
        vector_position INTEGER,  -- позиция в векторном индексе
        created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
        UNIQUE(book_id, chunk_index)  -- один индекс на книгу
    );
    
    CREATE INDEX IF NOT EXISTS idx_chunks_book ON chunks(book_id);
    CREATE INDEX IF NOT EXISTS idx_chunks_lookup ON chunks(book_id, chunk_index);
    CREATE INDEX IF NOT EXISTS idx_books_title ON books(title);
`);


// добавляем книгу в базу (или игнорируем если уже есть)
function addBook(title, filename, author = null) {
    const stmt = db.prepare('INSERT OR IGNORE INTO books (title, filename, author) VALUES (?, ?, ?)');
    const result = stmt.run(title, filename, author);
    // если книга уже была — возвращаем её id
    return result.lastInsertRowid || getBookByTitle(title)?.id;
}

// ищем книгу по точному названию
function getBookByTitle(title) {
    return db.prepare('SELECT * FROM books WHERE title = ?').get(title);
}

// получаем книгу по id
function getBookById(id) {
    return db.prepare('SELECT * FROM books WHERE id = ?').get(id);
}

// список всех книг (свежие сверху)
function getAllBooks() {
    return db.prepare('SELECT * FROM books ORDER BY loaded_at DESC').all();
}

// удаляем книгу и все её чанки (каскадно)
function deleteBook(id) {
    db.prepare('DELETE FROM books WHERE id = ?').run(id);
}



// добавляем один чанк
function addChunk(bookId, chunkIndex, text, tokenEstimate = null, chapter = null) {
    const stmt = db.prepare(`
        INSERT OR REPLACE INTO chunks (book_id, chunk_index, text, token_estimate, chapter)
        VALUES (?, ?, ?, ?, ?)
    `);
    const result = stmt.run(bookId, chunkIndex, text, tokenEstimate, chapter);
    return result.lastInsertRowid;
}

// массовая вставка чанков 
function addChunksBatch(chunks) {
    const insert = db.prepare(`
        INSERT OR REPLACE INTO chunks (book_id, chunk_index, text, token_estimate, chapter)
        VALUES (?, ?, ?, ?, ?)
    `);
    
    const insertMany = db.transaction((items) => {
        for (const c of items) {
            insert.run(c.bookId, c.chunkIndex, c.text, c.tokenEstimate || null, c.chapter || null);
        }
    });
    
    insertMany(chunks);
}

// получаем чанк по его id
function getChunkById(id) {
    return db.prepare('SELECT * FROM chunks WHERE id = ?').get(id);
}

// получаем конкретный чанк книги по индексу
function getChunkByBookAndIndex(bookId, chunkIndex) {
    return db.prepare('SELECT * FROM chunks WHERE book_id = ? AND chunk_index = ?').get(bookId, chunkIndex);
}

// все чанки книги (по порядку)
function getChunksByBookId(bookId, orderBy = 'chunk_index ASC') {
    return db.prepare(`SELECT * FROM chunks WHERE book_id = ? ORDER BY ${orderBy}`).all(bookId);
}

// сколько чанков у книги
function getChunkCount(bookId) {
    const row = db.prepare('SELECT COUNT(*) as count FROM chunks WHERE book_id = ?').get(bookId);
    return row.count;
}

// помечаем позицию чанка в векторном хранилище
function updateVectorPosition(chunkId, position) {
    db.prepare('UPDATE chunks SET vector_position = ? WHERE id = ?').run(position, chunkId);
}

// получаем несколько чанков по списку id
function getChunksByIds(ids) {
    if (!ids || ids.length === 0) return [];
    const placeholders = ids.map(() => '?').join(',');
    return db.prepare(`SELECT * FROM chunks WHERE id IN (${placeholders})`).all(...ids);
}

//текстовый поиск по чанкам книги
function searchChunks(bookId, query, limit = 5) {
    return db.prepare(`
        SELECT * FROM chunks 
        WHERE book_id = ? AND text LIKE ? 
        ORDER BY chunk_index 
        LIMIT ?
    `).all(bookId, `%${query}%`, limit);
}

// экспортируем всё что нужно снаружи
export {
    addBook,
    getBookByTitle,
    getBookById,
    getAllBooks,
    deleteBook,
    addChunk,
    addChunksBatch,
    getChunkById,
    getChunkByBookAndIndex,
    getChunksByBookId,
    getChunkCount,
    updateVectorPosition,
    getChunksByIds,
    searchChunks,
    db
};