ament_add_gtest(
  cyberdog_lifecycle_test cyberdog_lifecycle_test.cpp
  TIMEOUT 120
)
ament_target_dependencies(cyberdog_lifecycle_test ${dependencies})
target_link_libraries(cyberdog_lifecycle_test ${library_name})
