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

#ifndef COMMON_PROTOCOL__COMMON_HPP_
#define COMMON_PROTOCOL__COMMON_HPP_

#include <set>
#include <map>
#include <string>
#include <memory>
#include <vector>
#include <utility>
#include <iostream>
#include <algorithm>

#include "toml11/toml.hpp"

namespace cyberdog
{
namespace common
{
#define C_END "\033[m"
#define C_RED "\033[0;32;31m"
#define C_YELLOW "\033[1;33m"

#define CAN_STD_MAX_ID 0x7FFU
#define CAN_EXT_MAX_ID 0x1FFF'FFFFU

#define STATE_CODE_TYPE uint8_t
#define STATE_CODE_TIMES uint8_t
#define MAX_STATE_TIMES static_cast<int>(STATE_CODE_TIMES(-1))
#define CHILD_STATE_CLCT std::shared_ptr<StateCollector>
#define PROTOCOL_DATA_MAP std::map<std::string, ProtocolData>

using StateMap = std::map<STATE_CODE_TYPE, STATE_CODE_TIMES>;

std::set<std::string> common_type =
  std::set<std::string> {"float", "double", "i64", "i32", "i16", "i8", "u64", "u32", "u16",
  "u8", "bool"};

enum ErrorCode
{
  INIT_ERROR,
  ILLEGAL_PROTOCOL,
  DATA_AREA_CONFLICT,

  CAN_MIXUSING_ERROR,
  CAN_FD_SEND_ERROR,
  CAN_STD_SEND_ERROR,
  CAN_ID_OUTOFRANGE,

  TOML_NOKEY_ERROR,
  TOML_TYPE_ERROR,
  TOML_OTHER_ERROR,

  HEXTOUINT_ILLEGAL_CHAR,
  HEXTOUINT_ILLEGAL_START,

  RULE_SAMENAME_ERROR,

  RULEVAR_ILLEGAL_PARSERTYPE,
  RULEVAR_ILLEGAL_PARSERPARAM_SIZE,
  RULEVAR_ILLEGAL_PARSERPARAM_VALUE,
  RULEVAR_ILLEGAL_VARTYPE,
  RULEVAR_ILLEGAL_VARNAME,

  RULEARRAY_ILLEGAL_ARRAYNAME,
  RULEARRAY_SAMECANID_ERROR,
  RULEARRAY_ILLEGAL_PARSERPARAM_VALUE,

  RULECMD_ILLEGAL_CMDNAME,
  RULECMD_CTRLDATA_ERROR,
  RULECMD_SAMECMD_ERROR,
  RULECMD_MISSING_ERROR,

  DOUBLE_SIMPLIFY_ERROR,
  FLOAT_SIMPLIFY_ERROR,

  RUNTIME_UNEXPECT_ORDERPACKAGE,
  RUNTIME_SIZEOVERFLOW,
  RUNTIME_SIZENOTMATCH,
  RUNTIME_OPERATE_ERROR,
  RUNTIME_NOLINK_ERROR,
  RUNTIME_SAMELINK_ERROR,
  RUNTIME_ILLEGAL_LINKVAR,
};

class ProtocolData
{
public:
  ProtocolData(uint8_t len, void * addr)
  {
    this->len = len;
    this->addr = addr;
    loaded = false;
    array_expect = 0;
  }
  uint8_t len;
  void * addr;
  bool loaded;
  int array_expect;
};  // class ProtocolData

class StateCollector
{
public:
  StateCollector() {}
  explicit StateCollector(STATE_CODE_TYPE state_code) {LogState(state_code);}
  void LogState(STATE_CODE_TYPE state_code) {LogState(state_code, 1);}
  StateMap GetAllStateMap()
  {
    auto all = StateMap();
    for (auto & pair : state_map_) {all.insert(pair);}
    for (auto & child : children_) {
      for (auto & pair : child->GetAllStateMap()) {
        if (all.find(pair.first) == all.end()) {all.insert(pair);} else {
          all.at(pair.first) = std::min(all.at(pair.first) + pair.second, MAX_STATE_TIMES);
        }
      }
    }
    return all;
  }
  const StateMap & GetSelfStateMap() {return state_map_;}
  void ClearAllState()
  {
    ClearSelfState();
    for (auto & child : children_) {child->ClearAllState();}
  }
  void ClearSelfState() {state_map_.clear();}
  unsigned int GetSelfStateTypeNum()
  {
    return static_cast<unsigned int>(state_map_.size());
  }
  unsigned int GetAllStateTypeNum()
  {
    unsigned int num = 0;
    num += GetSelfStateTypeNum();
    for (auto & child : children_) {num += child->GetAllStateTypeNum();}
    return num;
  }
  unsigned int GetSelfStateTimesNum()
  {
    unsigned int num = 0;
    for (auto & pair : state_map_) {num += pair.second;}
    return num;
  }
  unsigned int GetAllStateTimesNum()
  {
    unsigned int num = 0;
    num += GetSelfStateTimesNum();
    for (auto & child : children_) {num += child->GetAllStateTimesNum();}
    return num;
  }
  unsigned int GetSelfStateTimesNum(STATE_CODE_TYPE state_code)
  {
    return (state_map_.find(state_code) == state_map_.end()) ? 0 : state_map_.at(state_code);
  }
  unsigned int GetAllStateTimesNum(STATE_CODE_TYPE state_code)
  {
    unsigned int num = 0;
    num += GetSelfStateTimesNum(state_code);
    for (auto & child : children_) {num += child->GetAllStateTimesNum(state_code);}
    return num;
  }
  void PrintfSelfStateStr()
  {
    if (state_map_.size() == 0) {printf("[STATE_COLLECTOR] NoStateCode\n");} else {
      printf("[STATE_COLLECTOR] StateType:%ld\n", state_map_.size());
    }
    for (auto & a : state_map_) {
      printf("[STATE_COLLECTOR] StateCode[%3d]-times:%d\n", a.first, a.second);
    }
  }
  void PrintfAllStateStr()
  {
    auto all_map = GetAllStateMap();
    if (all_map.size() == 0) {printf("[STATE_COLLECTOR] NoStateCode\n");} else {
      printf("[STATE_COLLECTOR] StateType:%ld\n", all_map.size());
    }
    for (auto & a : all_map) {
      printf("[STATE_COLLECTOR] StateCode[%3d]-times:%d\n", a.first, a.second);
    }
  }

  CHILD_STATE_CLCT CreatChild()
  {
    children_.push_back(std::make_shared<StateCollector>());
    return children_[children_.size() - 1];
  }

private:
  StateMap state_map_ = StateMap();
  std::vector<CHILD_STATE_CLCT> children_ = std::vector<CHILD_STATE_CLCT>();
  void LogState(STATE_CODE_TYPE state_code, uint8_t state_times)
  {
    auto it = state_map_.find(state_code);
    if (it != state_map_.end()) {
      it->second = std::min(it->second + state_times, MAX_STATE_TIMES);
    } else {
      state_map_.insert(std::pair<STATE_CODE_TYPE, uint8_t>(state_code, 1));
    }
  }
};  // class StateCollector

std::string get_var_name(const std::string & full_name, StateCollector & clct)
{
  bool get = false;
  bool start = false;
  std::string var_name = "";
  for (auto & c : full_name) {
    if (start) {
      if (get) {var_name += c;}
      if (c == '>' || c == '.') {
        if (get == true) {
          clct.LogState(ErrorCode::RUNTIME_ILLEGAL_LINKVAR);
          printf(
            C_RED "[PROTOCOL_BASE][ERROR] Not support class or struct "
            "in TDataClass:\"%s\"\n" C_END,
            full_name.c_str());
        }
        var_name = "";
        get = true;
      }
    } else if (c == ')') {start = true;}
  }
  if (get == false) {
    clct.LogState(ErrorCode::RUNTIME_ILLEGAL_LINKVAR);
    printf(C_YELLOW "[PROTOCOL_BASE][WARN] LINK_VAR(x) may get error var\n" C_END);
    return full_name;
  }
  return var_name;
}

unsigned int HEXtoUINT(const std::string & str, CHILD_STATE_CLCT clct)
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
        if (clct != nullptr) {clct->LogState(ErrorCode::HEXTOUINT_ILLEGAL_CHAR);}
        printf(
          C_RED "[PROTOCOL_BASE][ERROR] HEX string:\"%s\" format error, "
          "illegal char:\"%c\"\n" C_END, str.c_str(), ch);
        return 0x0;
      }
    }
    if (ch == 'x' || ch == 'X') {start = true;}
  }
  if (start == false) {
    if (clct != nullptr) {clct->LogState(ErrorCode::HEXTOUINT_ILLEGAL_START);}
    printf(
      C_RED "[PROTOCOL_BASE][ERROR] HEX string:\"%s\" format error, need start with \"0x\"\n" C_END,
      str.c_str());
  }
  return id;
}

bool CanidRangeCheck(
  uint can_id, bool extended,
  const std::string & name,
  const std::string & sub_name,
  CHILD_STATE_CLCT clct)
{
  if (can_id > (extended ? CAN_EXT_MAX_ID : CAN_STD_MAX_ID)) {
    if (clct != nullptr) {clct->LogState(ErrorCode::CAN_ID_OUTOFRANGE);}
    printf(
      C_RED "[CAN_PARSER][ERROR][%s][%s] CAN_ID:0x%x out of range, %s\n" C_END,
      name.c_str(), sub_name.c_str(), can_id, std::string(
        extended ? "Extend ID:(0x0000'0000~0x1FFF'FFFF)" :
        "Stand ID:(0x000~0x7FF)").c_str());
    return false;
  }
  return true;
}

template<typename T>
T toml_at(const toml::table & table, const std::string & key, CHILD_STATE_CLCT clct)
{
  if (table.find(key) == table.end()) {
    if (clct != nullptr) {clct->LogState(ErrorCode::TOML_NOKEY_ERROR);}
    printf(C_RED "[TOML][ERROR] Can't find key:\"%s\"\n" C_END, key.c_str());
    return T();
  }
  try {
    return toml::get<T>(table.at(key));
  } catch (toml::type_error & ex) {
    if (clct != nullptr) {clct->LogState(ErrorCode::TOML_TYPE_ERROR);}
    printf(C_RED "[TOML][ERROR] %s\n" C_END, ex.what());
  } catch (std::exception & ex) {
    if (clct != nullptr) {clct->LogState(ErrorCode::TOML_OTHER_ERROR);}
    printf(C_RED "[TOML][ERROR] %s\n" C_END, ex.what());
  }
  return T();
}

template<typename T>
T toml_at(const toml::table & table, const std::string & key, T default_value = T())
{
  if (table.find(key) == table.end()) {return default_value;}
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
    return true;
  } catch (toml::syntax_error & ex) {
    printf(C_RED "[PROTOCOL][ERROR] %s\n" C_END, ex.what());
  } catch (std::runtime_error & ex) {
    printf(C_RED "[PROTOCOL][ERROR] %s\n" C_END, ex.what());
  } catch (...) {
    printf(C_RED "[PROTOCOL][ERROR] Some unknow error\n" C_END);
  }
  return false;
}
}  // namespace common
}  // namespace cyberdog

#endif  // COMMON_PROTOCOL__COMMON_HPP_
