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
add_library(loader_contract_driver OBJECT loader_abi_contract.cc)
target_include_directories(loader_contract_driver PRIVATE ../../include)
target_compile_definitions(loader_contract_driver PRIVATE
  CORTEX_CONTRACT_HEADER="cortex_driver_api.h")
add_library(loader_contract_ceph OBJECT loader_abi_contract.cc)
target_include_directories(loader_contract_ceph PRIVATE ../../sys/src)
target_compile_definitions(loader_contract_ceph PRIVATE
  CORTEX_CONTRACT_HEADER="cortex/cortex_driver_api.h")

add_library(loader_abi_stub SHARED loader_abi_stub.cc)
target_include_directories(loader_abi_stub PRIVATE ../../include)
add_library(loader_old_abi_stub SHARED loader_abi_stub.cc)
target_include_directories(loader_old_abi_stub PRIVATE ../../include)
target_compile_definitions(loader_old_abi_stub PRIVATE CORTEX_TEST_OLD_VERSION)
add_library(loader_incomplete_abi_stub SHARED loader_abi_stub.cc)
target_include_directories(loader_incomplete_abi_stub PRIVATE ../../include)
target_compile_definitions(loader_incomplete_abi_stub PRIVATE CORTEX_TEST_MISSING_SYMBOL)
add_executable(test_loader_abi test_loader_abi.cc
  ../../sys/src/cortex/loader/cortex_driver_loader.cc)
target_include_directories(test_loader_abi PRIVATE ../../sys/src)
target_compile_definitions(test_loader_abi PRIVATE
  ABI_STUB="$<TARGET_FILE:loader_abi_stub>"
  OLD_ABI_STUB="$<TARGET_FILE:loader_old_abi_stub>"
  INCOMPLETE_ABI_STUB="$<TARGET_FILE:loader_incomplete_abi_stub>"
  DRIVER_SO_PATH="$<TARGET_FILE:cortex_driver>")
target_link_libraries(test_loader_abi GTest::gtest_main dl)
add_dependencies(test_loader_abi loader_abi_stub loader_old_abi_stub
  loader_incomplete_abi_stub loader_contract_driver loader_contract_ceph cortex_driver)
add_test(NAME test_loader_abi COMMAND test_loader_abi)
add_executable(test_loader_leases test_loader_leases.cc
  ../../sys/src/cortex/loader/cortex_driver_loader.cc)
target_include_directories(test_loader_leases PRIVATE ../../sys/src)
target_compile_definitions(test_loader_leases PRIVATE ABI_STUB="$<TARGET_FILE:loader_abi_stub>")
target_link_libraries(test_loader_leases GTest::gtest_main dl)
add_dependencies(test_loader_leases loader_abi_stub)
add_test(NAME test_loader_leases COMMAND test_loader_leases)
add_library(loader_lease_stub SHARED loader_lease_stub.cc)
target_include_directories(loader_lease_stub PRIVATE ../../include)
add_executable(test_semantic_result_guard test_semantic_result_guard.cc)
target_include_directories(test_semantic_result_guard PRIVATE ../../sys/src)
target_link_libraries(test_semantic_result_guard GTest::gtest_main)
add_test(NAME test_semantic_result_guard COMMAND test_semantic_result_guard)
find_program(CORTEX_TEST_PYTHON python3 REQUIRED)
add_test(NAME test_entrypoint_contract COMMAND ${CORTEX_TEST_PYTHON}
  ${CMAKE_CURRENT_SOURCE_DIR}/test_entrypoint_contract.py
  ${CMAKE_CURRENT_SOURCE_DIR}/../../test/integration/entrypoint.sh
  $<TARGET_FILE:cortex_driver> $<TARGET_FILE:loader_old_abi_stub>)

# Struct layout verification must run inside HLS environment (csim_design).
# add_executable(test_abi_boundary test_abi_boundary.cc)
# target_include_directories(test_abi_boundary PRIVATE ...)
# target_link_libraries(test_abi_boundary cortex_driver GTest::gtest_main)
# add_test(NAME test_abi_boundary COMMAND test_abi_boundary)
