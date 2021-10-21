// Copyright (c) 2021 Beijing Xiaomi Mobile Software Co., Ltd. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef CYBERDOG_UTILS__PARSER_HPP_
#define CYBERDOG_UTILS__PARSER_HPP_

#include <experimental/filesystem>  // NOLINT
#include <string>

#include "rclcpp/rclcpp.hpp"
#include "toml11/toml.hpp"

namespace cyberdog_utils
{
void message_info(
  rclcpp::node_interfaces::NodeLoggingInterface::SharedPtr log_interface,
  std::string_view log)
{
  RCLCPP_INFO_STREAM(log_interface->get_logger(), log);
}

void message_warn(
  rclcpp::node_interfaces::NodeLoggingInterface::SharedPtr log_interface,
  std::string_view log)
{
  RCLCPP_INFO_STREAM(log_interface->get_logger(), log);
}

void message_error(
  rclcpp::node_interfaces::NodeLoggingInterface::SharedPtr log_interface,
  std::string_view log)
{
  RCLCPP_INFO_STREAM(log_interface->get_logger(), log);
}

template<typename T>
bool get_raw_toml_data(
  const std::string file_path,
  const std::string data_tag,
  T & raw_data)
{
  bool rtn(false);

  if (std::experimental::filesystem::exists(file_path)) {
    raw_data = toml::find<T>(toml::parse(file_path), data_tag);
    rtn = true;
  }

  return rtn;
}

}  // namespace cyberdog_utils

#endif  // CYBERDOG_UTILS__PARSER_HPP_
