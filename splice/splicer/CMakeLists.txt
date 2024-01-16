cmake_minimum_required(VERSION 3.0)
project(splicer LANGUAGES C CXX)

# Use FFmpeg
find_package(PkgConfig REQUIRED)
pkg_check_modules(
    ffmpeg
    REQUIRED
    IMPORTED_TARGET
    libavfilter
    libavformat
    libavcodec
    libavutil
    libswscale
    libavdevice
    libswresample
)

aux_source_directory(. SOURCES)
add_library(${PROJECT_NAME} SHARED ${SOURCES})

target_link_libraries(${PROJECT_NAME}
    PRIVATE
    PkgConfig::ffmpeg
)
