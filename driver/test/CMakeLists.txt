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
