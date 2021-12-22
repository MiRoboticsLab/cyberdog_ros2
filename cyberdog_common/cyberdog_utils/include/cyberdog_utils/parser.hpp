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

#define C_END "\033[m"
#define C_RED "\033[0;32;31m"
#define C_YELLOW "\033[1;33m"

#include <experimental/filesystem>  // NOLINT
#include <string>
#include <string_view>  // NOLINT
#include <type_traits>

#include "rclcpp/rclcpp.hpp"
#include "toml11/toml.hpp"

namespace cyberdog
{
namespace util
{

void message_info(
  std::string_view log,
  const rclcpp::Logger log_head = rclcpp::get_logger("Common_Logger"))
{
  RCLCPP_INFO_STREAM(log_head, log);
}

void message_warn(
  std::string_view log,
  const rclcpp::Logger log_head = rclcpp::get_logger("Common_Logger"))
{
  RCLCPP_INFO_STREAM(log_head, log);
}

void message_error(
  std::string_view log,
  const rclcpp::Logger log_head = rclcpp::get_logger("Common_Logger"))
{
  RCLCPP_INFO_STREAM(log_head, log);
}

/**
 * @brief Get the raw toml data from filepath
 *
 * @param file_path Absolute path of TOML file
 * @param toml_value Value to return
 * @return true
 * @return false
 */
bool get_toml_from_path(
  const std::string file_path,
  toml::value & toml_value)
{
  bool _rtn(true);
  if (!std::experimental::filesystem::exists(file_path)) {
    message_error(std::string("TOML file not found"));
    _rtn = false;
  } else {
    try {
      toml_value = toml::parse(file_path);
    } catch (const std::runtime_error & e) {
      message_error(std::string("Value not found in TOML file"));
      _rtn = false;
    }
  }
  return _rtn;
}

/**
 * @brief Get the raw data from TOML value
 *
 * @param toml_value TOML value to read
 * @param data_tag Data tag of value in TOML file
 * @param raw_data Value to return
 * @return true
 * @return false
 */
template<typename T>
bool get_raw_from_toml(
  const toml::value toml_value,
  const std::string data_tag,
  T & raw_data)
{
  bool _rtn(true);
  try {
    raw_data = toml::find<T>(toml_value, data_tag);
  } catch (const std::out_of_range & out_err) {
    _rtn = false;
    message_error(std::string("Value not found in TOML value."));
  }
  return _rtn;
}

/**
 * @brief Get value from toml::table
 * @param m Table to search
 * @param key Key in table
 * @param default_value Default key value
 * @return Key value or default value
 */
template<typename Value>
Value toml_get_or(
  const toml::table & m,
  const std::string & key,
  const Value & default_value,
  bool & missing_flag = false)
{
  auto value_iter = m.find(key);
  auto value(default_value);
  if (value_iter == m.end()) {
    missing_flag = true;
    return value;
  }
  try {
    value = toml::get<Value>(value_iter->second);
  } catch (std::runtime_error & ex) {
    missing_flag = true;
    printf(C_RED "[STD][RUNTIME] %s\n" C_END, ex.what());
  } catch (toml::type_error & ex) {
    missing_flag = true;
    printf(C_RED "[TOML][ERROR] %s\n" C_END, ex.what());
  }
  return value;
}

/**
 * @brief Get value from map or unordered map
 * @param m map or unordered map to search
 * @param key key in map or unordered map
 * @param default_value default key value
 * @return key value or default value
 */
template<template<typename ...> class MapType, typename Key, typename Value>
const Value & map_get_or(
  const MapType<Key, Value> & m,
  const Key & key,
  const Value & default_value,
  bool & missing_flag = false)
{
  auto value_iter = m.find(key);
  if (value_iter == m.end()) {
    missing_flag = true;
    return default_value;
  }
  return value_iter->second;
}

}  // namespace util
}  // namespace cyberdog

#endif  // CYBERDOG_UTILS__PARSER_HPP_
