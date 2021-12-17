find_package(rclcpp REQUIRED)
find_package(pluginlib REQUIRED)

include_directories(
  display_test/include
)

set(dependencies
  rclcpp
  pluginlib
)

add_library(light_plugins SHARED
  display_test/src/light_plugin.cpp
)

pluginlib_enable_plugin_testing(
  CMAKE_TARGET_VAR mock_light_target
  AMENT_PREFIX_PATH_VAR mock_light_path
  PACKAGE_NAME "light_pluginlib"
  PACKAGE_XML "display_test/light_package.xml"
  PLUGIN_CATEGORY "light_pluginlib"
  PLUGIN_DESCRIPTIONS "display_test/light_plugins.xml"
  PLUGIN_LIBRARIES light_plugins
)

ament_target_dependencies(light_plugins ${dependencies})

ament_add_gtest(
  display__light_plugins__test display_test/light_plugins_test.cpp
  APPEND_LIBRARY_DIRS $<TARGET_FILE_DIR:light_plugins>
  APPEND_ENV AMENT_PREFIX_PATH=${mock_light_path}
  TIMEOUT 20
)

ament_target_dependencies(display__light_plugins__test
  ${dependencies}
)
add_dependencies(display__light_plugins__test
  ${mock_light_target}
)
