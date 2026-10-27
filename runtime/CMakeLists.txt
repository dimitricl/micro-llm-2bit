cmake_minimum_required(VERSION 3.20)
project(micro_llm_runtime LANGUAGES C)

set(CMAKE_C_STANDARD 99)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_C_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
endif()

option(LLM2_FORCE_SCALAR "Utiliser le noyau scalaire même sur ARM/x86" OFF)

# Sélection du noyau : un seul fichier compilé (même symbole ternary_matvec).
if(LLM2_FORCE_SCALAR)
  set(LLM2_KERNEL src/kernels/scalar.c)
elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "^(arm64|aarch64|ARM64)")
  set(LLM2_KERNEL src/kernels/neon.c)
elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "^(x86_64|AMD64|amd64)")
  set(LLM2_KERNEL src/kernels/avx2.c)
else()
  set(LLM2_KERNEL src/kernels/scalar.c)
endif()

message(STATUS "LLM2 kernel: ${LLM2_KERNEL}")

add_executable(micro-llm
  src/main.c
  src/engine.c
  ${LLM2_KERNEL}
)

target_include_directories(micro-llm PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)

if(NOT MSVC)
  target_compile_options(micro-llm PRIVATE -O3 -Wall -Wextra)
  if(CMAKE_SYSTEM_PROCESSOR MATCHES "^(x86_64|AMD64|amd64)" AND NOT LLM2_FORCE_SCALAR)
    target_compile_options(micro-llm PRIVATE -mavx2)
  endif()
endif()

if(UNIX)
  target_link_libraries(micro-llm PRIVATE m)
endif()
