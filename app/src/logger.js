import winston from 'winston';
import DailyRotateFile from 'winston-daily-rotate-file';
import path from 'path';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

// уровни логов
const customLevels = {
  levels: {
    fatal: 0,
    error: 1,
    warn: 2,
    info: 3,
    debug: 4
  },
  colors: {
    fatal: 'red bold',
    error: 'red',
    warn: 'yellow',
    info: 'green',
    debug: 'blue'
  }
};

// кастомный формат для цветной консоли в dev-режиме
const devFormat = winston.format.printf(({ level, message, timestamp, module, ...meta }) => {
  let metaStr = '';
  const cleanMeta = { ...meta };
  // убираем служебные поля
  delete cleanMeta.service;
  delete cleanMeta.pid;
  
  if (Object.keys(cleanMeta).length > 0) {
    metaStr = ' ' + JSON.stringify(cleanMeta);
  }
  
  const moduleTag = module ? `[${module}]` : '';
  return `${timestamp} ${level} ${moduleTag}: ${message}${metaStr}`;
});

// основной логгер
export const logger = winston.createLogger({
  levels: customLevels.levels,  
  level: process.env.LOG_LEVEL || 'debug',
  // информация которая добавляется ко всем записям
  defaultMeta: {
    service: 'chat4-backend',
    pid: process.pid
  },
  transports: [
    // консоль с цветным выводом для разработки
    new winston.transports.Console({
      format: winston.format.combine(
        winston.format.colorize({ all: true }),
        winston.format.timestamp({ format: 'HH:mm:ss' }),
        devFormat
      )
    }),
    
    // файл со всеми логами
    new DailyRotateFile({
      filename: path.join(__dirname, 'logs', 'app-%DATE%.log'),
      datePattern: 'YYYY-MM-DD',
      zippedArchive: true,
      maxSize: '20m',
      maxFiles: '14d',
      format: winston.format.combine(
        winston.format.timestamp(),
        winston.format.json()
      )
    }),
    
    //файл только с ошибками 
    new DailyRotateFile({
      filename: path.join(__dirname, 'logs', 'errors-%DATE%.log'),
      datePattern: 'YYYY-MM-DD',
      level: 'error',
      zippedArchive: true,
      maxSize: '20m',
      maxFiles: '30d',
      format: winston.format.combine(
        winston.format.timestamp(),
        winston.format.json()
      )
    })
  ],
  
  // ловим непойманные исключения и отклонённые промисы
  exceptionHandlers: [
    new DailyRotateFile({
      filename: path.join(__dirname, 'logs', 'exceptions-%DATE%.log'),
      datePattern: 'YYYY-MM-DD',
      zippedArchive: true,
      maxSize: '20m',
      maxFiles: '30d',
      format: winston.format.combine(
        winston.format.timestamp(),
        winston.format.json()
      )
    })
  ],
  rejectionHandlers: [
    new DailyRotateFile({
      filename: path.join(__dirname, 'logs', 'rejections-%DATE%.log'),
      datePattern: 'YYYY-MM-DD',
      zippedArchive: true,
      maxSize: '20m',
      maxFiles: '30d',
      format: winston.format.combine(
        winston.format.timestamp(),
        winston.format.json()
      )
    })
  ]
});

// очерний логгер для операций с базой данных
export const dbLogger = logger.child({ module: 'database' });

//дочерний логгер для запросов к llama
export const llamaLogger = logger.child({ module: 'llama' });

//дочерний логгер для http запросов 
export const httpLogger = logger.child({ module: 'http' });

// экспортируем для удобства и функцию смены уровня
export function setLogLevel(level) {
  logger.level = level;
  logger.warn(`Log level changed to: ${level}`);
}
winston.addColors(customLevels.colors);