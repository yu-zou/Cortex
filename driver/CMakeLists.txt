cmake_minimum_required(VERSION 3.16)
project(cortex_driver CXX)

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

set(CORTEX_HW_LIB_DIR "" CACHE PATH "Path to hw/ library output directory")

add_library(cortex_driver SHARED
  src/smartssd_driver.cc
  src/driver_manager.cc
  src/cortex_driver_exports.cc
)

target_include_directories(cortex_driver
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../include
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../hw/include
)

target_link_libraries(cortex_driver PRIVATE dl)

set_target_properties(cortex_driver PROPERTIES
  INSTALL_RPATH "${CORTEX_HW_LIB_DIR}"
  BUILD_WITH_INSTALL_RPATH TRUE
)

install(TARGETS cortex_driver LIBRARY DESTINATION lib)

enable_testing()
add_subdirectory(test)
