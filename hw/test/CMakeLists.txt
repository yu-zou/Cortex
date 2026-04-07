add_executable(test_mock_device test_mock_device.cc)
target_include_directories(test_mock_device PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/../include
  ${CMAKE_CURRENT_SOURCE_DIR}/../src
  ${CMAKE_CURRENT_SOURCE_DIR}
)
target_compile_definitions(test_mock_device PRIVATE
  CORTEX_FIXTURE_DIR="/cortex/hw/test/fixtures/"
)
target_link_libraries(test_mock_device cortex_hw_mock GTest::gtest_main)
add_test(NAME test_mock_device COMMAND test_mock_device)

add_executable(test_device_interface test_device_interface.cc)
target_include_directories(test_device_interface PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/../include
)
target_link_libraries(test_device_interface cortex_hw_mock GTest::gtest_main)
add_test(NAME test_device_interface COMMAND test_device_interface)
