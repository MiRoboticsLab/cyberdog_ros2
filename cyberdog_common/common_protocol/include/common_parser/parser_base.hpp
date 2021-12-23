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

#ifndef COMMON_PARSER__PARSER_BASE_HPP_
#define COMMON_PARSER__PARSER_BASE_HPP_

#include <vector>
#include <string>
#include <memory>

#include "toml11/toml.hpp"
#include "common_protocol/common.hpp"

namespace cyberdog
{
namespace common
{
class RuleVarBase
{
public:
  explicit RuleVarBase(
    CHILD_STATE_CLCT clct,
    const toml::table & table,
    const std::string & name,
    int can_len,
    bool extended)
  {
    error_clct = clct;
    warn_flag = false;
    var_name = toml_at<std::string>(table, "var_name", error_clct);
    var_type = toml_at<std::string>(table, "var_type", error_clct);
    can_id = HEXtoUINT(toml_at<std::string>(table, "can_id", error_clct), error_clct);
    CanidRangeCheck(can_id, extended, var_name, name, error_clct);
    if (var_name == "") {
      error_clct->LogState(ErrorCode::RULEVAR_ILLEGAL_VARNAME);
      printf(
        C_RED "[PARSER_BASE][ERROR][%s] var_name error, not support empty string\n" C_END,
        name.c_str());
    }
    if (common_type.find(var_type) == common_type.end()) {
      error_clct->LogState(ErrorCode::RULEVAR_ILLEGAL_VARTYPE);
      printf(
        C_RED "[PARSER_BASE][ERROR][%s][var:%s] var_type error, type:\"%s\" not support; "
        "only support:[", name.c_str(), var_name.c_str(), var_type.c_str());
      for (auto & t : common_type) {
        printf("%s, ", t.c_str());
      }
      printf("]\n" C_END);
    }
    if (table.find("var_zoom") == table.end()) {var_zoom = 1.0;} else {
      if (var_type != "float" && var_type != "double") {
        warn_flag = true;
        printf(
          C_YELLOW "[PARSER_BASE][WARN][%s][var:%s] Only double/float need var_zoom\n" C_END,
          name.c_str(), var_name.c_str());
      }
      var_zoom = toml_at<float>(table, "var_zoom", error_clct);
    }
    parser_type = toml_at<std::string>(table, "parser_type", "auto");
    auto tmp_parser_param = toml_at<std::vector<uint8_t>>(table, "parser_param", error_clct);
    size_t param_size = tmp_parser_param.size();
    if (parser_type == "auto") {
      if (param_size == 3) {
        parser_type = "bit";
      } else if (param_size == 2) {
        parser_type = "var";
      } else {
        error_clct->LogState(ErrorCode::RULEVAR_ILLEGAL_PARSERPARAM_SIZE);
        printf(
          C_RED "[PARSER_BASE][ERROR][%s][var:%s] Can't get parser_type via parser_param, "
          "only param_size == 2 or 3, but get param_size = %ld\n" C_END,
          name.c_str(), var_name.c_str(), param_size);
      }
    }
    if (parser_type == "bit") {
      if (param_size == 3) {
        parser_param[0] = tmp_parser_param[0];
        parser_param[1] = tmp_parser_param[1];
        parser_param[2] = tmp_parser_param[2];
        if (parser_param[0] >= can_len) {
          error_clct->LogState(ErrorCode::RULEVAR_ILLEGAL_PARSERPARAM_VALUE);
          printf(
            C_RED "[PARSER_BASE][ERROR][%s][var:%s] \"bit\" type parser_param error, "
            "parser_param[0] value need between 0-%d\n" C_END,
            name.c_str(), var_name.c_str(), can_len - 1);
        }
        if (parser_param[1] < parser_param[2]) {
          error_clct->LogState(ErrorCode::RULEVAR_ILLEGAL_PARSERPARAM_VALUE);
          printf(
            C_RED "[PARSER_BASE][ERROR][%s][var:%s] \"bit\" type parser_param error, "
            "parser_param[1] need >= parser_param[2]\n" C_END,
            name.c_str(), var_name.c_str());
        }
        if (parser_param[1] >= 8 || parser_param[2] >= 8) {
          error_clct->LogState(ErrorCode::RULEVAR_ILLEGAL_PARSERPARAM_VALUE);
          printf(
            C_RED "[PARSER_BASE][ERROR][%s][var:%s] \"bit\" type parser_param error, "
            "parser_param[1] and parser_param[2] value need between 0-7\n" C_END,
            name.c_str(), var_name.c_str());
        }
      } else {
        error_clct->LogState(ErrorCode::RULEVAR_ILLEGAL_PARSERPARAM_SIZE);
        printf(
          C_RED "[PARSER_BASE][ERROR][%s][var:%s] \"bit\" type parser error, "
          "parser[bit] need 3 parser_param, but get %d\n" C_END,
          name.c_str(), var_name.c_str(), static_cast<uint8_t>(param_size));
      }
    } else if (parser_type == "var") {
      if (param_size == 2) {
        parser_param[0] = tmp_parser_param[0];
        parser_param[1] = tmp_parser_param[1];
        if (parser_param[0] > parser_param[1]) {
          error_clct->LogState(ErrorCode::RULEVAR_ILLEGAL_PARSERPARAM_VALUE);
          printf(
            C_RED "[PARSER_BASE][ERROR][%s][var:%s] \"var\" type parser_param error, "
            "parser_param[0] need <= parser_param[1]\n" C_END,
            name.c_str(), var_name.c_str());
        }
        if (parser_param[0] >= can_len || parser_param[1] >= can_len) {
          error_clct->LogState(ErrorCode::RULEVAR_ILLEGAL_PARSERPARAM_VALUE);
          printf(
            C_RED "[PARSER_BASE][ERROR][%s][var:%s] \"var\" type parser_param error, "
            "parser_param[0] and parser_param[1] value need between 0-%d\n" C_END,
            name.c_str(), var_name.c_str(), can_len - 1);
        }
      } else {
        error_clct->LogState(ErrorCode::RULEVAR_ILLEGAL_PARSERPARAM_SIZE);
        printf(
          C_RED "[PARSER_BASE][ERROR][%s][var:%s] \"var\" type parser error, "
          "parser[var] need 2 parser_param, but get %d\n" C_END,
          name.c_str(), var_name.c_str(), static_cast<uint8_t>(param_size));
      }
    } else if (parser_type != "auto") {
      error_clct->LogState(ErrorCode::RULEVAR_ILLEGAL_PARSERTYPE);
      printf(
        C_RED "[PARSER_BASE][ERROR][%s][var:%s] var can parser error, "
        "only support \"bit/var\", but get %s\n" C_END,
        name.c_str(), var_name.c_str(), parser_type.c_str());
    }
  }
  CHILD_STATE_CLCT error_clct;
  bool warn_flag;
  canid_t can_id;
  std::string var_name;
  std::string var_type;
  float var_zoom;
  std::string parser_type;
  uint8_t parser_param[3];
};    // class RuleVarBase
}  // namespace common
}  // namespace cyberdog

#endif  // COMMON_PARSER__PARSER_BASE_HPP_
