cmake_minimum_required(VERSION 3.16)
project(cortex_hw CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

include(FetchContent)
FetchContent_Declare(
  googletest
  GIT_REPOSITORY https://github.com/google/googletest.git
  GIT_TAG release-1.12.1
)
FetchContent_MakeAvailable(googletest)

FetchContent_Declare(
  faiss
  GIT_REPOSITORY https://github.com/facebookresearch/faiss.git
  GIT_TAG v1.7.3
)
set(FAISS_ENABLE_GPU OFF CACHE BOOL "" FORCE)
set(FAISS_ENABLE_PYTHON OFF CACHE BOOL "" FORCE)
set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
set(BUILD_TESTING OFF CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(faiss)

add_library(cortex_hw_mock SHARED
  src/mock_smartssd.cc
  src/cortex_hw_mock_exports.cc
)
target_include_directories(cortex_hw_mock
  PUBLIC include
  PRIVATE src
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../include
)
target_link_libraries(cortex_hw_mock PRIVATE faiss)

install(TARGETS cortex_hw_mock LIBRARY DESTINATION lib)

enable_testing()
add_subdirectory(test)
