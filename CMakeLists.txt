cmake_minimum_required(VERSION 3.20)
project(VectorDB VERSION 1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# 1. Находим nlohmann_json и flatc
find_package(nlohmann_json REQUIRED)
find_program(FLATBUFFERS_COMPILER flatc REQUIRED)

# 2. ONNX Runtime - указываем путь вручную
set(ONNXRUNTIME_ROOT "/opt/onnxruntime")
set(ONNXRUNTIME_INCLUDE_DIRS "${ONNXRUNTIME_ROOT}/include")
set(ONNXRUNTIME_LIBRARIES "${ONNXRUNTIME_ROOT}/lib/libonnxruntime.so")

# 3. Генерация FlatBuffers
set(FBS_SCHEMA "${CMAKE_CURRENT_SOURCE_DIR}/schemas/schema.fbs")
set(GEN_HEADER_DIR "${CMAKE_CURRENT_BINARY_DIR}/generated")
set(GEN_HEADER_FILE "${GEN_HEADER_DIR}/schema_generated.h")

file(MAKE_DIRECTORY ${GEN_HEADER_DIR})

add_custom_command(
    OUTPUT ${GEN_HEADER_FILE}
    COMMAND ${FLATBUFFERS_COMPILER} --cpp -o ${GEN_HEADER_DIR} ${FBS_SCHEMA}
    DEPENDS ${FBS_SCHEMA}
    COMMENT "Generating FlatBuffers C++ header from ${FBS_SCHEMA}"
)

# 4. Сборка исполняемого файла
add_executable(vector_db 
    src/main.cpp 
    src/storage_manager.cpp 
    src/text_receiver.cpp
    src/vectorizer.cpp
    ${GEN_HEADER_FILE}
)

target_include_directories(vector_db PRIVATE 
    include
    src
    ${GEN_HEADER_DIR}
    ${ONNXRUNTIME_INCLUDE_DIRS}
)

# 5. Линковка библиотек
find_package(Threads REQUIRED)

target_link_libraries(vector_db PRIVATE 
    nlohmann_json::nlohmann_json 
    Threads::Threads
    cpp-httplib
    ${ONNXRUNTIME_LIBRARIES}
)

# Копируем .so файл в билд-директорию
add_custom_command(TARGET vector_db POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy
    ${ONNXRUNTIME_LIBRARIES}
    $<TARGET_FILE_DIR:vector_db>
)

# 6. Флаги компиляции
target_compile_options(vector_db PRIVATE -Wall -Wextra -Wpedantic)
target_compile_options(vector_db PRIVATE
    $<$<CONFIG:Release>:-O3 -march=native -ffast-math>
)