include_directories(
  ${CMAKE_SOURCE_DIR}/include
)

set(dependencies
  toml11_vendor
)

ament_add_gtest(
  common_protocol_test common_protocol_test/common_protocol_test.cpp
  TIMEOUT 20
)
ament_target_dependencies(common_protocol_test ${dependencies})

target_compile_definitions(common_protocol_test PRIVATE
  "PASER_PATH=\"${CMAKE_SOURCE_DIR}/test/common_protocol_test/parser\""
  "COMMON_PROTOCOL_TEST=true"
)
