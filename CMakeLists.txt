cmake_minimum_required(VERSION 3.10)
project(TransformerFromScratch VERSION 1.1 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    add_compile_options(-Wall -Wextra)
elseif(MSVC)
    add_compile_options(/W4)
endif()

# Core library: tokenizer, embedding, positional encoding, attention, FFN, layer norm,
# encoder, decoder and the Transformer wrapper.
add_library(transformer STATIC
    src/Common/Embedding.cpp
    src/Common/FeedForwardNetwork.cpp
    src/Common/FinalLayer.cpp
    src/Common/LayerNorm.cpp
    src/Common/SelfAttention.cpp
    src/Common/Tokenizer.cpp
    src/Common/VectorMath.cpp
    src/Decoding/Decoder.cpp
    src/Decoding/DecoderLayer.cpp
    src/Encoding/Encoder.cpp
    src/Encoding/EncoderLayer.cpp
    src/Encoding/PositionalEncoding.cpp
    src/Transformer.cpp
)
target_include_directories(transformer PUBLIC include)

# Demo: one forward pass through the full (untrained) encoder-decoder.
add_executable(transformer_demo examples/demo.cpp)
target_link_libraries(transformer_demo PRIVATE transformer)

