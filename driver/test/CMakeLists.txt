add_executable(test_smartssd_driver test_smartssd_driver.cc)
target_include_directories(test_smartssd_driver PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/../src
  ${CMAKE_CURRENT_SOURCE_DIR}/../../include
  ${CMAKE_CURRENT_SOURCE_DIR}/../../hw/include
)
target_compile_definitions(test_smartssd_driver PRIVATE
  CORTEX_FIXTURE_DIR="${CMAKE_CURRENT_SOURCE_DIR}/../../hw/test/fixtures"
  HW_LIB_PATH="${CORTEX_HW_LIB_DIR}/libcortex_hw_mock.so"
)
target_link_libraries(test_smartssd_driver cortex_driver GTest::gtest_main)
add_test(NAME test_smartssd_driver COMMAND test_smartssd_driver)

add_executable(test_driver_abi test_driver_abi.cc)
target_include_directories(test_driver_abi PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/../../include
)
target_compile_definitions(test_driver_abi PRIVATE
  CORTEX_FIXTURE_DIR="${CMAKE_CURRENT_SOURCE_DIR}/../../hw/test/fixtures"
  HW_LIB_PATH="${CORTEX_HW_LIB_DIR}/libcortex_hw_mock.so"
  DRIVER_SO_PATH="${CMAKE_BINARY_DIR}/lib/libcortex_driver.so"
)
target_link_libraries(test_driver_abi cortex_driver GTest::gtest_main dl)
add_test(NAME test_driver_abi COMMAND test_driver_abi)

add_executable(test_driver_hw_integration test_driver_hw_integration.cc)
target_include_directories(test_driver_hw_integration PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/../src
  ${CMAKE_CURRENT_SOURCE_DIR}/../../include
  ${CMAKE_CURRENT_SOURCE_DIR}/../../hw/include
)
target_compile_definitions(test_driver_hw_integration PRIVATE
  CORTEX_FIXTURE_DIR="${CMAKE_CURRENT_SOURCE_DIR}/../../hw/test/fixtures"
  HW_LIB_PATH="${CORTEX_HW_LIB_DIR}/libcortex_hw_mock.so"
)
target_link_libraries(test_driver_hw_integration cortex_driver GTest::gtest_main)
add_test(NAME test_driver_hw_integration COMMAND test_driver_hw_integration)

add_executable(test_driver_cache_stress test_driver_cache_stress.cc)
target_include_directories(test_driver_cache_stress PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/../src
  ${CMAKE_CURRENT_SOURCE_DIR}/../../include
  ${CMAKE_CURRENT_SOURCE_DIR}/../../hw/include
)
target_compile_definitions(test_driver_cache_stress PRIVATE
  CORTEX_FIXTURE_DIR="${CMAKE_CURRENT_SOURCE_DIR}/../../hw/test/fixtures"
  HW_LIB_PATH="${CORTEX_HW_LIB_DIR}/libcortex_hw_mock.so"
)
target_link_libraries(test_driver_cache_stress cortex_driver GTest::gtest_main)
add_test(NAME test_driver_cache_stress COMMAND test_driver_cache_stress)

add_executable(test_driver_errors test_driver_errors.cpp)
target_include_directories(test_driver_errors PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/../src
  ${CMAKE_CURRENT_SOURCE_DIR}/../../include
  ${CMAKE_CURRENT_SOURCE_DIR}/../../hw/include
)
target_compile_definitions(test_driver_errors PRIVATE
  CORTEX_FIXTURE_DIR="${CMAKE_CURRENT_SOURCE_DIR}/../../hw/test/fixtures"
  HW_LIB_PATH="${CORTEX_HW_LIB_DIR}/libcortex_hw_mock.so"
)
target_link_libraries(test_driver_errors cortex_driver GTest::gtest_main)
add_test(NAME test_driver_errors COMMAND test_driver_errors)

# test_abi_boundary disabled: requires Vitis HLS headers (ap_int.h).
# Struct layout verification must run inside HLS environment (csim_design).
# add_executable(test_abi_boundary test_abi_boundary.cc)
# target_include_directories(test_abi_boundary PRIVATE ...)
# target_link_libraries(test_abi_boundary cortex_driver GTest::gtest_main)
# add_test(NAME test_abi_boundary COMMAND test_abi_boundary)
