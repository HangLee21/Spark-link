cmake_minimum_required(VERSION 3.0)

project(splicer LANGUAGES C CXX)

set(LIBRARY_OUTPUT_PATH ${PROJECT_BINARY_DIR}/lib) # 指定动态库输出目录

add_subdirectory(splicer)