cmake_minimum_required(VERSION 3.18)

project(XGBOOST-TEST CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
# add_compile_options(-Wall -Wextra)

include(FetchContent)
FetchContent_Declare(
    xgboost
    GIT_REPOSITORY "https://github.com/dmlc/xgboost.git"
    GIT_TAG v1.5.1
)
FetchContent_MakeAvailable(xgboost)
FetchContent_GetProperties(xgboost SOURCE_DIR XGBOOST_INCLUDE_DIR)
include_directories(${XGBOOST_INCLUDE_DIR})
include_directories("${XGBOOST_INCLUDE_DIR}/src") # in order to use internal headers for adapter

include(FetchContent)
FetchContent_Declare(
        csvparser
        GIT_REPOSITORY "https://github.com/vincentlaucsb/csv-parser.git"
        GIT_TAG 2.1.3
        PATCH_COMMAND ""
)
FetchContent_MakeAvailable(csvparser)
FetchContent_GetProperties(csvparser SOURCE_DIR CSVPARSER_INCLUDE_DIR)
add_library(csvparser INTERFACE)
target_include_directories(csvparser INTERFACE ${CSVPARSER_INCLUDE_DIR})
include_directories(${CSVPARSER_INCLUDE_DIR})

add_executable(urls urls.cc)
target_link_libraries(urls xgboost dmlc csvparser)
