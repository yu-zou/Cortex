add_executable(test_smartssd_driver test_smartssd_driver.cc)
target_include_directories(test_smartssd_driver PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${CMAKE_CURRENT_SOURCE_DIR}/../include
  ${CMAKE_CURRENT_SOURCE_DIR}/../../../include
)
target_link_libraries(test_smartssd_driver cortex_driver)
add_test(NAME test_smartssd_driver COMMAND test_smartssd_driver)

add_executable(test_driver_abi test_driver_abi.cc)
target_include_directories(test_driver_abi PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${CMAKE_CURRENT_SOURCE_DIR}/../include
  ${CMAKE_CURRENT_SOURCE_DIR}/../../../include
)
target_link_libraries(test_driver_abi cortex_driver)
add_test(NAME test_driver_abi COMMAND test_driver_abi)
