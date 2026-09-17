import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';
import { addBook, addChunksBatch, deleteBook, db } from './booksDB.js';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const BOOK_FILE = path.join(__dirname, 'newBook.txt'); // путь к файлу книги
const CPP_HOST = 'http://localhost:8081'; // адрес C++ сервера для векторизации
const CHUNK_TOKENS = 128; // размер чанка в токенах
const OVERLAP_TOKENS = 30; // перекрытие между чанками в токенах
const CHARS_PER_TOKEN = 4; // примерное соотношение символов к токенам
const BATCH_SIZE = 32; // размер батча для отправки



function estimateTokens(text) {
    return Math.ceil(text.length / CHARS_PER_TOKEN); // грубая оценка токенов по длине текста
}

function splitIntoChunks(text) {
    const chunks = [];
    const targetChars = CHUNK_TOKENS * CHARS_PER_TOKEN; // целевой размер чанка в символах
    const overlapChars = OVERLAP_TOKENS * CHARS_PER_TOKEN; // размер перекрытия в символах
    const safeOverlap = Math.min(overlapChars, targetChars - 1); // защита от отрицательного перекрытия
    
    let start = 0;
    let chunkIndex = 0;
    
    while (start < text.length && chunkIndex < 100000) { // ограничение на макс. число чанков
        let end = start + targetChars;
        
        if (end < text.length) {
            const searchEnd = Math.min(end + 100, text.length); // ищем ближайший конец предложения
            const slice = text.slice(end, searchEnd);
            const match = slice.match(/[.!?]\s/); // разбиваем по границам предложений
            if (match) end = end + match.index + 1;
        }
        
        if (end > text.length) end = text.length; // не выходим за границы текста
        
        const chunkText = text.slice(start, end).trim(); // вырезаем чанк
        if (chunkText.length > 0) {
            chunks.push({ chunkIndex, text: chunkText, tokenEstimate: estimateTokens(chunkText) });
            chunkIndex++;
        }
        
        const nextStart = end - safeOverlap; // следующий старт с учётом перекрытия
        if (nextStart <= start || nextStart >= text.length) break; // выход 
        start = nextStart;
    }
    
    return chunks;
}

async function sendBatch(batch, bookId) {
    const payload = {
        texts: batch.map(c => c.text), // тексты чанков
        book_id: String(bookId), // id книги
        chunk_indices: batch.map(c => c.chunkIndex) // индексы чанков
    };
    
    const response = await fetch(`${CPP_HOST}/vectorize_book`, { // отправка в c++
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload)
    });
    
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    return await response.json();
}

async function processBook() {
    console.log('обработка книги');
    
    if (!fs.existsSync(BOOK_FILE)) { // проверка наличия файла
        console.log(`❌ Файл не найден: ${BOOK_FILE}`);
        db.close();
        return;
    }
    
    const stat = fs.statSync(BOOK_FILE);
    console.log(` Файл: ${BOOK_FILE} (${(stat.size / 1024).toFixed(1)} КБ)\n`);
    
    const text = fs.readFileSync(BOOK_FILE, 'utf-8'); // читаем весь текст
    console.log(`   Символов: ${text.length.toLocaleString()}\n`);
    
    const chunks = splitIntoChunks(text); // режем на чанки
    console.log(`   Чанков: ${chunks.length}\n`);
    
    const bookTitle = path.basename(BOOK_FILE, '.txt'); // название книги из имени файла
    
    // удаляем старую книгу если есть
    const existing = db.prepare('SELECT id FROM books WHERE title = ?').get(bookTitle);
    if (existing) {
        deleteBook(existing.id); // удаление предыдущей версии
        console.log(` Старая книга "${bookTitle}" удалена\n`);
    }
    
    // сохраняем новую книгу в sqlite
    const bookId = addBook(bookTitle, BOOK_FILE);
    
    const dbChunks = chunks.map(c => ({ // готовим чанки для записи в бд
        bookId,
        chunkIndex: c.chunkIndex,
        text: c.text,
        tokenEstimate: c.tokenEstimate,
        chapter: null
    }));
    addChunksBatch(dbChunks); // вставка чанков
    console.log(` ${chunks.length} чанков сохранено в books.db\n`);
    
    // отправляем в c++ сервис для векторизации
    const totalBatches = Math.ceil(chunks.length / BATCH_SIZE); // всего чанков
    let success = 0;
    let fail = 0;
    
    console.log(` Отправдено (${totalBatches} батчей по ${BATCH_SIZE})\n`);
    
    for (let i = 0; i < chunks.length; i += BATCH_SIZE) { // отправляем чанки
        const batch = chunks.slice(i, i + BATCH_SIZE);
        const batchNum = Math.floor(i / BATCH_SIZE) + 1;
        
        try {
            const result = await sendBatch(batch, bookId); // отправка одного чанка
            success += batch.length;
            console.log(`    Батч ${batchNum}/${totalBatches} — ${batch.length} чанков (всего в БД C++: ${result.total_in_db})`);
        } catch (err) {
            fail += batch.length;
            console.log(`    Батч ${batchNum}/${totalBatches} — ${err.message}`);
        }
        
        if (i + BATCH_SIZE < chunks.length) {
            await new Promise(r => setTimeout(r, 50)); // небольшая задержка между отправками
        }
    }
    

    console.log(`  Книга ID: ${String(bookId).padEnd(34)}`);
    if (fail > 0) console.log(`║  Ошибок:  ${String(fail).padEnd(34)}║`);
    
    db.close(); // закрываем соединение с бд
}

processBook(); // запуск 