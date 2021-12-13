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

#ifndef COMMON_DEVICE__COMMON_HPP_
#define COMMON_DEVICE__COMMON_HPP_

#include <set>
#include <string>
#include <iostream>

#include "toml11/toml.hpp"

namespace bit_tool
{
union u16u8_ {
  uint16_t u16_;
  uint8_t u8_[2];
} u16u8;
union u32u8_ {
  uint32_t u32_;
  uint8_t u8_[4];
} u32u8;
union u64u8_ {
  uint64_t u64_;
  uint8_t u8_[8];
} u64u8;
}  // namespace bit_tool

namespace common_device
{
#define C_END "\033[m"
#define C_RED "\033[0;32;31m"
#define C_YELLOW "\033[1;33m"

std::set<std::string> common_type =
  std::set<std::string> {"float", "double", "i64", "i32", "i16", "i8", "u64", "u32", "u16",
  "u8", "bool"};

std::string get_var_name(const std::string & full_name)
{
  bool get = false;
  bool start = false;
  std::string var_name = "";
  for (auto & c : full_name) {
    if (start) {
      if (get) {var_name += c;}
      if (c == '>' || c == '.') {
        if (get == true) {
          printf(
            C_RED "[DEVICE_BASE][ERROR] Not support class or struct "
            "in TDataClass:\"%s\"\n" C_END,
            full_name.c_str());
        }
        var_name = "";
        get = true;
      }
    } else if (c == ')') {start = true;}
  }
  if (get == false) {
    printf(C_YELLOW "[DEVICE_BASE][WARN] LINK_VAR(x) may get error var\n" C_END);
    return full_name;
  }
  return var_name;
}

unsigned int HEXtoUINT(const std::string & str, bool * error_flag = nullptr)
{
  unsigned int id = 0x0;
  bool start = false;
  for (auto & ch : str) {
    if (start) {
      if (ch == ' ' || ch == '\'') {continue;}
      id *= 16;
      if ('0' <= ch && ch <= '9') {
        id += ch - '0';
      } else if ('a' <= ch && ch <= 'f') {
        id += ch - 'a' + 10;
      } else if ('A' <= ch && ch <= 'F') {
        id += ch - 'A' + 10;
      } else {
        if (error_flag != nullptr) {*error_flag = true;}
        printf(
          C_RED "[DEVICE_BASE][ERROR] HEX string:\"%s\" format error, illegal char:\"%c\"\n" C_END,
          str.c_str(), ch);
        return 0x0;
      }
    }
    if (ch == 'x' || ch == 'X') {start = true;}
  }
  if (start == false) {
    if (error_flag != nullptr) {*error_flag = true;}
    printf(
      C_RED "[DEVICE_BASE][ERROR] HEX string:\"%s\" format error, need start with \"0x\"\n" C_END,
      str.c_str());
  }
  return id;
}

template<typename T>
T toml_at(const toml::table & table, const std::string & key, bool & error_flag)
{
  if (table.find(key) == table.end()) {
    error_flag = true;
    printf(C_RED "[TOML][ERROR] Can't find key:\"%s\"\n" C_END, key.c_str());
    return T();
  }
  try {
    return toml::get<T>(table.at(key));
  } catch (toml::type_error & ex) {
    printf(C_RED "[TOML][ERROR] %s\n" C_END, ex.what());
  }
  error_flag = true;
  return T();
}

template<typename T>
T toml_at(const toml::table & table, const std::string & key, T default_value = T())
{
  if (table.find(key) == table.end()) {
    return default_value;
  }
  try {
    return toml::get<T>(table.at(key));
  } catch (toml::type_error & ex) {
    printf(C_RED "[TOML][ERROR] %s\n" C_END, ex.what());
  }
  return default_value;
}

bool toml_parse(toml::value & toml, const std::string & path)
{
  try {
    toml = toml::parse(path);
  } catch (toml::syntax_error & ex) {
    printf(C_RED "[DEVICE][ERROR] %s\n" C_END, ex.what());
    return false;
  } catch (std::runtime_error & ex) {
    printf(C_RED "[DEVICE][ERROR] %s\n" C_END, ex.what());
    return false;
  } catch (...) {
    printf(C_RED "[DEVICE][ERROR] Some unknow error\n" C_END);
    return false;
  }
  return true;
}

class device_data
{
public:
  device_data(uint8_t len, void * addr)
  {
    this->len = len;
    this->addr = addr;
    loaded = false;
    array_except = 0;
  }
  uint8_t len;
  void * addr;
  bool loaded;
  int array_except;
};
}  // namespace common_device

#endif  // COMMON_DEVICE__COMMON_HPP_
