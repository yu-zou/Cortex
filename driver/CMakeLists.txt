cmake_minimum_required(VERSION 3.16)
project(cortex_driver CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include(FetchContent)
FetchContent_Declare(
  googletest
  GIT_REPOSITORY https://github.com/google/googletest.git
  GIT_TAG release-1.12.1
)
FetchContent_MakeAvailable(googletest)

set(CORTEX_HW_LIB_DIR "" CACHE PATH "Path to hw/ library output directory")
if(CORTEX_HW_LIB_DIR)
  find_library(CORTEX_HW_MOCK cortex_hw_mock PATHS ${CORTEX_HW_LIB_DIR} NO_DEFAULT_PATH)
endif()

add_library(cortex_driver SHARED
  src/smartssd_driver.cc
  src/driver_manager.cc
  src/cortex_driver_exports.cc
)
target_include_directories(cortex_driver
  PUBLIC include
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../include
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../hw/include
)
if(CORTEX_HW_MOCK)
  target_link_libraries(cortex_driver PRIVATE ${CORTEX_HW_MOCK})
endif()

install(TARGETS cortex_driver LIBRARY DESTINATION lib)

enable_testing()
add_subdirectory(test)
