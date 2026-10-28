add_executable(test_mock_device test_mock_device.cc)
target_include_directories(test_mock_device PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/../include
  ${CMAKE_CURRENT_SOURCE_DIR}/../src
  ${CMAKE_CURRENT_SOURCE_DIR}
)
target_compile_definitions(test_mock_device PRIVATE
  CORTEX_FIXTURE_DIR="${CMAKE_CURRENT_SOURCE_DIR}/fixtures/"
)
target_link_libraries(test_mock_device cortex_hw_mock GTest::gtest_main)
add_test(NAME test_mock_device COMMAND test_mock_device)

add_executable(test_device_interface test_device_interface.cc)
target_include_directories(test_device_interface PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/../include
)
target_link_libraries(test_device_interface cortex_hw_mock GTest::gtest_main)
add_test(NAME test_device_interface COMMAND test_device_interface)

# test_dm_header — has its own main() calling InitGoogleTest
add_executable(test_dm_header test_dm_header.cpp)
target_include_directories(test_dm_header PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/../include
  ${CMAKE_CURRENT_SOURCE_DIR}/../src
)
target_link_libraries(test_dm_header GTest::gtest)
add_test(NAME test_dm_header COMMAND test_dm_header)

# test_result_packer — needs ../test/test_utils.h path
add_executable(test_result_packer test_result_packer.cpp)
target_include_directories(test_result_packer PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}
)
target_link_libraries(test_result_packer GTest::gtest_main)
add_test(NAME test_result_packer COMMAND test_result_packer)

# test_systolic_topk_sw — standalone algorithm
add_executable(test_systolic_topk_sw test_systolic_topk_sw.cpp)
target_link_libraries(test_systolic_topk_sw GTest::gtest_main)
add_test(NAME test_systolic_topk_sw COMMAND test_systolic_topk_sw)


# test_multi_cluster — needs mock, fixtures, and src includes
add_executable(test_multi_cluster test_multi_cluster.cpp)
target_include_directories(test_multi_cluster PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${CMAKE_CURRENT_SOURCE_DIR}/../include
  ${CMAKE_CURRENT_SOURCE_DIR}/../src
)
target_compile_definitions(test_multi_cluster PRIVATE
  CORTEX_FIXTURE_DIR="${CMAKE_CURRENT_SOURCE_DIR}/fixtures/"
)
target_link_libraries(test_multi_cluster cortex_hw_mock GTest::gtest_main)
add_test(NAME test_multi_cluster COMMAND test_multi_cluster)

# test_adc_distance_sw — needs fixtures and fixture_loader.h
add_executable(test_adc_distance_sw test_adc_distance_sw.cpp)
target_include_directories(test_adc_distance_sw PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${CMAKE_CURRENT_SOURCE_DIR}/../include
)
target_compile_definitions(test_adc_distance_sw PRIVATE
  CORTEX_FIXTURE_DIR="${CMAKE_CURRENT_SOURCE_DIR}/fixtures/"
)
target_link_libraries(test_adc_distance_sw GTest::gtest_main)
add_test(NAME test_adc_distance_sw COMMAND test_adc_distance_sw)
