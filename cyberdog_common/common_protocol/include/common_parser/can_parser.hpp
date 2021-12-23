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

#ifndef COMMON_PARSER__CAN_PARSER_HPP_
#define COMMON_PARSER__CAN_PARSER_HPP_

#include <set>
#include <map>
#include <string>
#include <memory>
#include <vector>
#include <utility>
#include <iostream>

#include "toml11/toml.hpp"

#include "protocol/can/can_utils.hpp"
#include "common_parser/parser_base.hpp"

namespace cyberdog
{
namespace common
{
class CanParser
{
  class RuleVar : public RuleVarBase
  {
public:
    explicit RuleVar(
      CHILD_STATE_CLCT clct,
      const toml::table & table,
      const std::string & name,
      int can_len,
      bool extended)
    : RuleVarBase(clct, table, name, can_len, extended)
    {}
  };

  class ArrayRule
  {
public:
    explicit ArrayRule(
      CHILD_STATE_CLCT clct,
      const toml::table & table,
      const std::string & name,
      bool extended)
    {
      error_clct = clct;
      warn_flag = false;
      can_package_num = toml_at<size_t>(table, "can_package_num", error_clct);
      array_name = toml_at<std::string>(table, "array_name", error_clct);
      if (array_name == "") {
        error_clct->LogState(ErrorCode::RULEARRAY_ILLEGAL_ARRAYNAME);
        printf(
          C_RED "[CAN_PARSER][ERROR][%s] array_name error, not support empty string\n" C_END,
          name.c_str());
      }
      auto tmp_can_id = toml_at<std::vector<std::string>>(table, "can_id", error_clct);
      auto canid_num = tmp_can_id.size();
      if (canid_num == can_package_num) {
        int index = 0;
        for (auto & single_id : tmp_can_id) {
          canid_t canid = HEXtoUINT(single_id, error_clct);
          CanidRangeCheck(canid, extended, name, array_name, error_clct);
          if (can_id.find(canid) == can_id.end()) {
            can_id.insert(std::pair<canid_t, int>(canid, index++));
          } else {
            error_clct->LogState(ErrorCode::RULEARRAY_SAMECANID_ERROR);
            printf(
              C_RED "[CAN_PARSER][ERROR][%s][array:%s] array error, "
              "get same can_id:\"0x%x\"\n" C_END,
              name.c_str(), array_name.c_str(), canid);
          }
        }
        // continuous check
        bool first = true;
        canid_t last_id;
        for (auto & id : can_id) {
          if (first == true) {
            first = false;
            last_id = id.first;
            continue;
          }
          if (id.first - last_id != 1) {
            warn_flag = true;
            printf(
              C_YELLOW "[CAN_PARSER][WARN][%s][array:%s] toml_array:\"can_id\" not "
              "continuous increase\n" C_END,
              name.c_str(), array_name.c_str());
            break;
          }
        }
      } else if (can_package_num > 2 && canid_num == 2) {
        canid_t start_id = HEXtoUINT(tmp_can_id[0], error_clct);
        canid_t end_id = HEXtoUINT(tmp_can_id[1], error_clct);
        if (end_id - start_id + 1 == can_package_num) {
          int index = 0;
          for (canid_t a = start_id; a <= end_id; a++) {
            can_id.insert(std::pair<canid_t, int>(a, index++));
          }
        } else {
          error_clct->LogState(ErrorCode::RULEARRAY_ILLEGAL_PARSERPARAM_VALUE);
          printf(
            C_RED "[CAN_PARSER][ERROR][%s][array:%s] simple array method must follow: "
            "(canid low to high) and (high - low + 1 == can_package_num)\n" C_END,
            name.c_str(), array_name.c_str());
        }
      } else {
        error_clct->LogState(ErrorCode::RULEARRAY_ILLEGAL_PARSERPARAM_VALUE);
        printf(
          C_RED "[CAN_PARSER][ERROR][%s][array:%s] toml_array:\"can_id\" length not match "
          "can_package_num, toml_array:\"can_id\" length=%ld but can_package_num=%ld\n" C_END,
          name.c_str(), array_name.c_str(), canid_num, can_package_num);
      }
    }
    CHILD_STATE_CLCT error_clct;
    bool warn_flag;
    size_t can_package_num;
    std::map<canid_t, int> can_id;
    std::string array_name;

    inline int get_offset(canid_t canid)
    {
      if (can_id.find(canid) != can_id.end()) {
        return can_id.at(canid);
      }
      return -1;
    }
  };  // class ArrayRule

  class CmdRule
  {
public:
    explicit CmdRule(
      CHILD_STATE_CLCT clct,
      const toml::table & table,
      const std::string & name,
      bool extended)
    {
      error_clct = clct;
      warn_flag = false;
      cmd_name = toml_at<std::string>(table, "cmd_name", error_clct);
      if (cmd_name == "") {
        error_clct->LogState(ErrorCode::RULECMD_ILLEGAL_CMDNAME);
        printf(
          C_RED "[CAN_PARSER][ERROR][%s] cmd_name error, not support empty string\n" C_END,
          name.c_str());
      }
      can_id = HEXtoUINT(toml_at<std::string>(table, "can_id", error_clct), error_clct);
      CanidRangeCheck(can_id, extended, name, cmd_name, error_clct);
      ctrl_len = toml_at<uint8_t>(table, "ctrl_len", 0);
      auto tmp_ctrl_data = toml_at<std::vector<std::string>>(table, "ctrl_data");
      ctrl_data = std::vector<uint8_t>();
      for (auto & str : tmp_ctrl_data) {
        auto uint_hex = HEXtoUINT(str, error_clct);
        if (uint_hex != (uint_hex & 0xFF)) {
          error_clct->LogState(ErrorCode::RULECMD_CTRLDATA_ERROR);
          printf(
            C_RED "[CAN_PARSER][ERROR][%s][cmd:%s] ctrl_data HEX to uint8 overflow, "
            "HEX_string:\"%s\"\n" C_END,
            name.c_str(), cmd_name.c_str(), str.c_str());
        }
        ctrl_data.push_back(static_cast<uint8_t>(uint_hex & 0xFF));
      }
      int size = ctrl_data.size();
      if (ctrl_len < size) {
        error_clct->LogState(ErrorCode::RULECMD_CTRLDATA_ERROR);
        printf(
          C_RED "[CAN_PARSER][ERROR][%s][cmd:%s] ctrl_data overflow, "
          "ctrl_len:%d < ctrl_data.size:%d\n" C_END,
          name.c_str(), cmd_name.c_str(), ctrl_len, size);
      }
    }
    CHILD_STATE_CLCT error_clct;
    bool warn_flag;
    std::string cmd_name;
    canid_t can_id;
    uint8_t ctrl_len;
    std::vector<uint8_t> ctrl_data;
  };  // class CmdRule

public:
  CanParser(
    CHILD_STATE_CLCT error_clct,
    const toml::value & toml_config,
    const std::string & name)
  {
    name_ = name;
    canfd_ = toml::find_or<bool>(toml_config, "canfd_enable", false);
    extended_ = toml::find_or<bool>(toml_config, "extended_frame", false);
    error_clct_ = (error_clct == nullptr) ? std::make_shared<StateCollector>() : error_clct;

    auto var_list = toml::find_or<std::vector<toml::table>>(
      toml_config, "var",
      std::vector<toml::table>());
    auto array_list = toml::find_or<std::vector<toml::table>>(
      toml_config, "array",
      std::vector<toml::table>());
    auto cmd_list = toml::find_or<std::vector<toml::table>>(
      toml_config, "cmd",
      std::vector<toml::table>());

    std::set<std::string> var_name_check = std::set<std::string>();
    std::map<canid_t, std::vector<uint8_t>> data_check = std::map<canid_t, std::vector<uint8_t>>();
    // get var rule
    for (auto & var : var_list) {
      auto single_var = RuleVar(error_clct_->CreatChild(), var, name_, CAN_LEN(), extended_);
      if (single_var.warn_flag) {warn_num_++;}
      if (single_var.error_clct->GetSelfStateTimesNum() == 0) {
        canid_t canid = single_var.can_id;
        if (parser_var_map_.find(canid) == parser_var_map_.end()) {
          parser_var_map_.insert(
            std::pair<canid_t, std::vector<RuleVar>>(canid, std::vector<RuleVar>()));
        }
        // check error and warning
        if (same_var_error(single_var.var_name, var_name_check)) {continue;}
        check_data_area_error(single_var, data_check);
        parser_var_map_.at(canid).push_back(single_var);
      }
    }
    // get array rule
    for (auto & array : array_list) {
      auto single_array = ArrayRule(error_clct_->CreatChild(), array, name_, extended_);
      if (single_array.warn_flag) {warn_num_++;}
      if (single_array.error_clct->GetSelfStateTimesNum() == 0) {
        // check error and warning
        if (same_var_error(single_array.array_name, var_name_check)) {continue;}
        check_data_area_error(single_array, data_check);
        parser_array_.push_back(single_array);
      }
    }
    // get cmd rule
    for (auto & cmd : cmd_list) {
      auto single_cmd = CmdRule(error_clct_->CreatChild(), cmd, name_, extended_);
      if (single_cmd.warn_flag) {warn_num_++;}
      if (single_cmd.error_clct->GetSelfStateTimesNum() == 0) {
        if (single_cmd.ctrl_len > CAN_LEN()) {
          error_clct_->LogState(ErrorCode::RULECMD_CTRLDATA_ERROR);
          printf(
            C_RED "[CAN_PARSER][ERROR][%s] cmd_name:\"%s\", ctrl_len:%d > MAX_CAN_DATA:%d\n" C_END,
            name_.c_str(), single_cmd.cmd_name.c_str(), single_cmd.ctrl_len, CAN_LEN());
          continue;
        }
        std::string cmd_name = single_cmd.cmd_name;
        if (parser_cmd_map_.find(cmd_name) == parser_cmd_map_.end()) {
          parser_cmd_map_.insert(std::pair<std::string, CmdRule>(cmd_name, single_cmd));
        } else {
          error_clct_->LogState(ErrorCode::RULECMD_SAMECMD_ERROR);
          printf(
            C_RED "[CAN_PARSER][ERROR][%s] get same cmd_name:\"%s\"\n" C_END,
            name_.c_str(), cmd_name.c_str());
        }
      }
    }
  }

  int GetInitErrorNum() {return error_clct_->GetAllStateTimesNum();}
  int GetInitWarnNum() {return warn_num_;}
  uint8_t CAN_LEN() {return canfd_ ? CANFD_MAX_DLEN : CAN_MAX_DLEN;}
  bool IsCanfd() {return canfd_;}

  std::vector<canid_t> GetRecvList()
  {
    auto recv_list = std::vector<canid_t>();
    for (auto & a : parser_var_map_) {recv_list.push_back(a.first);}
    for (auto & a : parser_array_) {for (auto & b : a.can_id) {recv_list.push_back(b.first);}}
    return recv_list;
  }

  // return true when finish all package
  bool Decode(
    PROTOCOL_DATA_MAP & protocol_data_map,
    std::shared_ptr<canfd_frame> rx_frame,
    bool & error_flag)
  {
    return Decode(protocol_data_map, rx_frame->can_id, rx_frame->data, error_flag);
  }
  // return true when finish all package
  bool Decode(
    PROTOCOL_DATA_MAP & protocol_data_map,
    std::shared_ptr<can_frame> rx_frame,
    bool & error_flag)
  {
    return Decode(protocol_data_map, rx_frame->can_id, rx_frame->data, error_flag);
  }

  bool Encode(can_frame & tx_frame, const std::string & CMD, const std::vector<uint8_t> & data)
  {
    if (canfd_ == true) {
      error_clct_->LogState(ErrorCode::CAN_MIXUSING_ERROR);
      printf(
        C_RED "[CAN_PARSER][ERROR][%s][cmd:%s] Can't encode std_can via fd_can params\n" C_END,
        name_.c_str(), CMD.c_str());
      return false;
    }
    tx_frame.can_dlc = CAN_LEN();
    return Encode(CMD, tx_frame.can_id, tx_frame.data, data);
  }

  bool Encode(canfd_frame & tx_frame, const std::string & CMD, const std::vector<uint8_t> & data)
  {
    if (canfd_ == false) {
      error_clct_->LogState(ErrorCode::CAN_MIXUSING_ERROR);
      printf(
        C_RED "[CAN_PARSER][ERROR][%s][cmd:%s] Can't encode fd_can via std_can params\n" C_END,
        name_.c_str(), CMD.c_str());
      return false;
    }
    tx_frame.len = CAN_LEN();
    return Encode(CMD, tx_frame.can_id, tx_frame.data, data);
  }

  bool Encode(
    const PROTOCOL_DATA_MAP & protocol_data_map,
    std::shared_ptr<CanDev> can_op)
  {
    bool no_error = true;
    canid_t * can_id;
    uint8_t * data;

    can_frame * std_frame = nullptr;
    canfd_frame * fd_frame = nullptr;
    if (canfd_) {
      fd_frame = new canfd_frame;
      fd_frame->len = 64;
      can_id = &fd_frame->can_id;
      data = &fd_frame->data[0];
    } else {
      std_frame = new can_frame;
      std_frame->can_dlc = 8;
      can_id = &std_frame->can_id;
      data = &std_frame->data[0];
    }

    // var encode
    for (auto & parser_var : parser_var_map_) {
      *can_id = parser_var.first;
      for (auto & rule : parser_var.second) {
        std::string var_name = rule.var_name;
        if (protocol_data_map.find(var_name) == protocol_data_map.end()) {
          no_error = false;
          error_clct_->LogState(ErrorCode::RUNTIME_NOLINK_ERROR);
          printf(
            C_RED "[CAN_PARSER][ERROR][%s] Can't find var_name:\"%s\" in protocol_data_map\n"
            "\tYou may need use LINK_VAR() to link data class/struct in protocol_data_map\n" C_END,
            name_.c_str(), var_name.c_str());
          continue;
        }
        const ProtocolData * const var = &protocol_data_map.at(var_name);
        if (rule.var_type == "double") {
          uint8_t u8_num = rule.parser_param[1] - rule.parser_param[0] + 1;
          if (u8_num == 2) {
            put_var<int16_t, double>(var, data, rule, name_, no_error);
          } else if (u8_num == 4) {
            put_var<int32_t, double>(var, data, rule, name_, no_error);
          } else if (u8_num == 8) {
            put_var<double>(var, data, rule, name_, no_error);
          } else {
            no_error = false;
            error_clct_->LogState(ErrorCode::DOUBLE_SIMPLIFY_ERROR);
            printf(
              C_RED "[CAN_PARSER][ERROR][%s] size %d can't send double\n" C_END,
              name_.c_str(), u8_num);
          }
        } else if (rule.var_type == "float") {
          uint8_t u8_num = rule.parser_param[1] - rule.parser_param[0] + 1;
          if (u8_num == 2) {
            put_var<int16_t, float>(var, data, rule, name_, no_error);
          } else if (u8_num == 4) {
            put_var<float>(var, data, rule, name_, no_error);
          } else {
            no_error = false;
            error_clct_->LogState(ErrorCode::FLOAT_SIMPLIFY_ERROR);
            printf(
              C_RED "[CAN_PARSER][ERROR][%s] size %d can't send float\n" C_END,
              name_.c_str(), u8_num);
          }
        } else if (rule.var_type == "bool") {
          put_var<bool>(var, data, rule, name_, no_error);
        } else if (rule.var_type == "u64") {
          put_var<uint64_t>(var, data, rule, name_, no_error);
        } else if (rule.var_type == "u32") {
          put_var<uint32_t>(var, data, rule, name_, no_error);
        } else if (rule.var_type == "u16") {
          put_var<uint16_t>(var, data, rule, name_, no_error);
        } else if (rule.var_type == "u8") {
          put_var<uint8_t>(var, data, rule, name_, no_error);
        } else if (rule.var_type == "i64") {
          put_var<int64_t>(var, data, rule, name_, no_error);
        } else if (rule.var_type == "i32") {
          put_var<int32_t>(var, data, rule, name_, no_error);
        } else if (rule.var_type == "i16") {
          put_var<int16_t>(var, data, rule, name_, no_error);
        } else if (rule.var_type == "i8") {
          put_var<int8_t>(var, data, rule, name_, no_error);
        }
      }
      // send out
      if (canfd_) {
        if (fd_frame == nullptr || can_op->send_can_message(*fd_frame) == false) {
          no_error = false;
          error_clct_->LogState(ErrorCode::CAN_FD_SEND_ERROR);
          printf(C_RED "[CAN_PARSER][ERROR][%s] Send fd_frame error\n" C_END, name_.c_str());
        }
      } else {
        if (std_frame == nullptr || can_op->send_can_message(*std_frame) == false) {
          no_error = false;
          error_clct_->LogState(ErrorCode::CAN_STD_SEND_ERROR);
          printf(C_RED "[CAN_PARSER][ERROR][%s] Send std_frame error\n" C_END, name_.c_str());
        }
      }
      // clear data buff
      std::memset(data, 0, CAN_LEN());
    }
    // array encode
    int frame_index = 0;
    for (auto & rule : parser_array_) {
      std::string array_name = rule.array_name;
      if (protocol_data_map.find(array_name) == protocol_data_map.end()) {
        no_error = false;
        error_clct_->LogState(ErrorCode::RUNTIME_NOLINK_ERROR);
        printf(
          C_RED "[CAN_PARSER][ERROR][%s] Can't find array_name:\"%s\" in protocol_data_map\n"
          "\tYou may need use LINK_VAR() to link data class/struct in protocol_data_map\n" C_END,
          name_.c_str(), array_name.c_str());
        continue;
      }
      const ProtocolData * const var = &protocol_data_map.at(array_name);
      int frame_num = static_cast<int>(rule.can_id.size());
      if (frame_num * CAN_LEN() != var->len) {
        no_error = false;
        error_clct_->LogState(ErrorCode::RULEARRAY_ILLEGAL_PARSERPARAM_VALUE);
        printf(
          C_RED "[CAN_PARSER][ERROR][%s] array_name:\"%s\" size not match, "
          "can't write to can frame data for send\n" C_END,
          name_.c_str(), array_name.c_str());
      }
      auto ids = std::vector<canid_t>(frame_num);
      for (auto & a : rule.can_id) {ids[a.second] = a.first;}
      uint8_t * protocol_array = static_cast<uint8_t *>(var->addr);
      for (int index = frame_index; index < frame_num; index++) {
        *can_id = ids[index];
        for (int a = 0; a < CAN_LEN(); a++) {
          data[a] = *protocol_array;
          protocol_array++;
        }
        // send out
        if (canfd_) {
          if (fd_frame == nullptr || can_op->send_can_message(*fd_frame) == false) {
            no_error = false;
            error_clct_->LogState(ErrorCode::CAN_FD_SEND_ERROR);
            printf(C_RED "[CAN_PARSER][ERROR][%s] Send fd_frame error\n" C_END, name_.c_str());
          }
        } else {
          if (std_frame == nullptr || can_op->send_can_message(*std_frame) == false) {
            no_error = false;
            error_clct_->LogState(ErrorCode::CAN_STD_SEND_ERROR);
            printf(C_RED "[CAN_PARSER][ERROR][%s] Send std_frame error\n" C_END, name_.c_str());
          }
        }
      }
    }

    if (canfd_) {delete fd_frame;} else {delete std_frame;}
    return no_error;
  }

private:
  bool canfd_;
  bool extended_;
  int warn_num_ = 0;

  std::string name_;
  CHILD_STATE_CLCT error_clct_;
  std::map<canid_t, std::vector<RuleVar>> parser_var_map_ =
    std::map<canid_t, std::vector<RuleVar>>();
  std::vector<ArrayRule> parser_array_ = std::vector<ArrayRule>();
  std::map<std::string, CmdRule> parser_cmd_map_ =
    std::map<std::string, CmdRule>();

  // return true when finish all package
  bool Decode(
    PROTOCOL_DATA_MAP & protocol_data_map,
    canid_t can_id,
    uint8_t * data,
    bool & error_flag)
  {
    // var decode
    if (parser_var_map_.find(can_id) != parser_var_map_.end()) {
      for (auto & rule : parser_var_map_.at(can_id)) {
        if (protocol_data_map.find(rule.var_name) != protocol_data_map.end()) {
          ProtocolData * var = &protocol_data_map.at(rule.var_name);
          // main decode begin
          if (rule.var_type == "double") {
            uint8_t u8_num = rule.parser_param[1] - rule.parser_param[0] + 1;
            if (u8_num == 2) {
              get_var<double, int16_t>(var, data, rule, name_, error_flag);
            } else if (u8_num == 4) {
              get_var<double, int32_t>(var, data, rule, name_, error_flag);
            } else if (u8_num == 8) {
              get_var<double>(var, data, rule, name_, error_flag);
            } else {
              error_flag = true;
              error_clct_->LogState(ErrorCode::DOUBLE_SIMPLIFY_ERROR);
              printf(
                C_RED "[CAN_PARSER][ERROR][%s] size %d can't get double\n" C_END,
                name_.c_str(), u8_num);
            }
            zoom_var<double>(var, rule.var_zoom);
          } else if (rule.var_type == "float") {
            uint8_t u8_num = rule.parser_param[1] - rule.parser_param[0] + 1;
            if (u8_num == 2) {
              get_var<float, int16_t>(var, data, rule, name_, error_flag);
            } else if (u8_num == 4) {
              get_var<float>(var, data, rule, name_, error_flag);
            } else {
              error_flag = true;
              error_clct_->LogState(ErrorCode::FLOAT_SIMPLIFY_ERROR);
              printf(
                C_RED "[CAN_PARSER][ERROR][%s] size %d can't get float\n" C_END,
                name_.c_str(), u8_num);
            }
            zoom_var<float>(var, rule.var_zoom);
          } else if (rule.var_type == "bool") {
            get_var<bool>(var, data, rule, name_, error_flag);
          } else if (rule.var_type == "u64") {
            get_var<uint64_t>(var, data, rule, name_, error_flag);
          } else if (rule.var_type == "u32") {
            get_var<uint32_t>(var, data, rule, name_, error_flag);
          } else if (rule.var_type == "u16") {
            get_var<uint16_t>(var, data, rule, name_, error_flag);
          } else if (rule.var_type == "u8") {
            get_var<uint8_t>(var, data, rule, name_, error_flag);
          } else if (rule.var_type == "i64") {
            get_var<int64_t>(var, data, rule, name_, error_flag);
          } else if (rule.var_type == "i32") {
            get_var<int32_t>(var, data, rule, name_, error_flag);
          } else if (rule.var_type == "i16") {
            get_var<int16_t>(var, data, rule, name_, error_flag);
          } else if (rule.var_type == "i8") {
            get_var<int8_t>(var, data, rule, name_, error_flag);
          }
        } else {
          error_flag = true;
          error_clct_->LogState(ErrorCode::RUNTIME_NOLINK_ERROR);
          printf(
            C_RED "[CAN_PARSER][ERROR][%s] Can't find var_name:\"%s\" in protocol_data_map\n"
            "\tYou may need use LINK_VAR() to link data class/struct in protocol_data_map\n" C_END,
            name_.c_str(), rule.var_name.c_str());
        }
      }
    }
    // array decode
    for (auto & rule : parser_array_) {
      auto offset = rule.get_offset(can_id);
      if (offset == -1) {continue;}
      if (protocol_data_map.find(rule.array_name) != protocol_data_map.end()) {
        ProtocolData * var = &protocol_data_map.at(rule.array_name);
        if (var->len < rule.can_package_num * CAN_LEN()) {
          error_flag = true;
          error_clct_->LogState(ErrorCode::RULEARRAY_ILLEGAL_PARSERPARAM_VALUE);
          printf(
            C_RED "[CAN_PARSER][ERROR][%s] array_name:\"%s\" length overflow\n" C_END,
            name_.c_str(), rule.array_name.c_str());
          continue;
        }
        if (offset == var->array_expect) {
          // main decode begin
          uint8_t * data_area = reinterpret_cast<uint8_t *>(var->addr);
          data_area += offset * CAN_LEN();
          for (int a = 0; a < static_cast<int>(CAN_LEN()); a++) {
            data_area[a] = data[a];
          }
          var->array_expect++;
          if (offset == static_cast<int>(rule.can_package_num) - 1) {
            var->loaded = true;
            var->array_expect = 0;
          }
        } else {
          canid_t expect_id = 0x0;
          for (auto & id : rule.can_id) {
            if (id.second == var->array_expect) {
              expect_id = id.first;
              break;
            }
          }
          var->array_expect = 0;
          error_flag = true;
          error_clct_->LogState(ErrorCode::RUNTIME_UNEXPECT_ORDERPACKAGE);
          printf(
            C_RED "[CAN_PARSER][ERROR][%s] array_name:\"%s\", expect can frame 0x%x, "
            "but get 0x%x, reset expect can_id and you need send array in order\n" C_END,
            name_.c_str(), rule.array_name.c_str(), expect_id, can_id);
        }
      } else {
        error_flag = true;
        error_clct_->LogState(ErrorCode::RUNTIME_NOLINK_ERROR);
        printf(
          C_RED "[CAN_PARSER][ERROR][%s] Can't find array_name:\"%s\" in protocol_data_map\n"
          "\tYou may need use LINK_VAR() to link data class/struct in protocol_data_map\n" C_END,
          name_.c_str(), rule.array_name.c_str());
      }
    }

    // check all frame received
    for (auto & single_var : protocol_data_map) {
      if (single_var.second.loaded == false) {return false;}
    }
    for (auto & single_var : protocol_data_map) {single_var.second.loaded = false;}
    return true;
  }

  bool Encode(
    const std::string & CMD,
    canid_t & can_id,
    uint8_t * can_data,
    const std::vector<uint8_t> & data)
  {
    if (parser_cmd_map_.find(CMD) != parser_cmd_map_.end()) {
      CmdRule * cmd = &parser_cmd_map_.at(CMD);
      can_id = cmd->can_id;
      uint8_t ctrl_len = cmd->ctrl_len;
      bool no_warn = true;
      if (ctrl_len + data.size() > CAN_LEN()) {
        no_warn = false;
        error_clct_->LogState(ErrorCode::RULEARRAY_ILLEGAL_PARSERPARAM_VALUE);
        printf(
          C_RED "[CAN_PARSER][ERROR][%s][cmd:%s] CMD data overflow, "
          "ctrl_len:%d + data_len:%ld > max_can_len:%d\n" C_END,
          name_.c_str(), CMD.c_str(), ctrl_len, data.size(), CAN_LEN());
      }
      for (int a = 0; a < CAN_LEN() && a < static_cast<int>(cmd->ctrl_data.size()); a++) {
        can_data[a] = cmd->ctrl_data[a];
      }
      for (int a = ctrl_len; a < CAN_LEN() && a - ctrl_len < static_cast<int>(data.size()); a++) {
        can_data[a] = data[a - ctrl_len];
      }
      return no_warn;
    } else {
      error_clct_->LogState(ErrorCode::RULECMD_MISSING_ERROR);
      printf(
        C_RED "[CAN_PARSER][ERROR][%s] can't find cmd:\"%s\"\n" C_END,
        name_.c_str(), CMD.c_str());
      return false;
    }
  }

  template<typename Target, typename Source>
  void get_var(
    ProtocolData * var,
    const uint8_t * const can_data,
    const RuleVar & rule,
    const std::string & parser_name,
    bool & error_flag)
  {
    if (sizeof(Target) > var->len) {
      error_flag = true;
      error_clct_->LogState(ErrorCode::RUNTIME_SIZEOVERFLOW);
      printf(
        C_RED "[CAN_PARSER][ERROR][%s] var_name:\"%s\" size overflow, "
        "can't write to protocol TDataClass\n" C_END,
        parser_name.c_str(), rule.var_name.c_str());
      return;
    }
    uint64_t result = 0;
    if (rule.parser_type == "var") {
      // var
      for (int a = rule.parser_param[0]; a <= rule.parser_param[1]; a++) {
        if (a != rule.parser_param[0]) {result <<= 8;}
        result |= can_data[a];
      }
    } else if (rule.parser_type == "bit") {
      // bit
      result =
        (can_data[rule.parser_param[0]] &
        creat_mask(rule.parser_param[1], rule.parser_param[2])) >> rule.parser_param[2];
    }

    *static_cast<Target *>(var->addr) = *(reinterpret_cast<Source *>(&result));
    var->loaded = true;
  }

  template<typename Target>
  inline void get_var(
    ProtocolData * var,
    const uint8_t * const can_data,
    const RuleVar & rule,
    const std::string & parser_name,
    bool & error_flag)
  {
    get_var<Target, Target>(var, can_data, rule, parser_name, error_flag);
  }

  template<typename Target, typename Source>
  void put_var(
    const ProtocolData * const var,
    uint8_t * can_data,
    const RuleVar & rule,
    const std::string & parser_name,
    bool & no_error_flag)
  {
    bool no_error = true;
    if (rule.parser_type == "bit") {
      uint8_t h_pos = rule.parser_param[1];
      uint8_t l_pos = rule.parser_param[2];
      can_data[rule.parser_param[0]] |=
        (*static_cast<uint8_t *>(var->addr) << l_pos) & creat_mask(h_pos, l_pos);
    } else if (rule.parser_type == "var") {
      uint8_t u8_num = rule.parser_param[1] - rule.parser_param[0] + 1;
      if (sizeof(Target) != u8_num) {
        no_error = false;
        error_clct_->LogState(ErrorCode::RUNTIME_SIZENOTMATCH);
        printf(
          C_RED "[CAN_PARSER][ERROR][%s] var_name:\"%s\" size not match, Target need:%ld - get:%d"
          ", can't write to can frame data for send\n" C_END,
          parser_name.c_str(), rule.var_name.c_str(), sizeof(Target), u8_num);
      }
      Target target;
      if (rule.var_type == "float" || rule.var_type == "double") {
        target = static_cast<Target>(*static_cast<Source *>(var->addr) / rule.var_zoom);
      } else {target = static_cast<Target>(*static_cast<Source *>(var->addr));}
      uint64_t * hex = reinterpret_cast<uint64_t *>(&target);
      uint8_t min = rule.parser_param[0];
      uint8_t max = rule.parser_param[1];
      for (int a = min; a <= max; a++) {
        can_data[a] = (*hex >> (max - a) * 8) & 0xFF;
      }
    }
    if (no_error == false) {no_error_flag = false;}
  }

  template<typename Target>
  inline void put_var(
    const ProtocolData * const var,
    uint8_t * can_data,
    const RuleVar & rule,
    const std::string & parser_name,
    bool & no_error_flag)
  {
    put_var<Target, Target>(var, can_data, rule, parser_name, no_error_flag);
  }

  template<typename T>
  inline void zoom_var(ProtocolData * var, float kp)
  {
    *static_cast<T *>(var->addr) *= kp;
  }

  uint8_t creat_mask(uint8_t h_bit, uint8_t l_bit)
  {
    uint8_t tmp = 0x1;
    uint8_t mask = 0x0;
    for (int a = 0; a < 8; a++) {
      if (l_bit <= a && a <= h_bit) {mask |= tmp;}
      tmp <<= 1;
    }
    return mask;
  }

  std::string show_conflict(uint8_t mask)
  {
    uint8_t tmp = 0b10000000;
    std::string result = "";
    for (int a = 0; a < 8; a++) {
      result += ((mask & tmp) != 0) ? ("*") : ("-");
      tmp >>= 1;
    }
    return result;
  }

  bool same_var_error(std::string & name, std::set<std::string> & checker)
  {
    if (checker.find(name) != checker.end()) {
      error_clct_->LogState(ErrorCode::RULE_SAMENAME_ERROR);
      printf(
        C_RED "[CAN_PARSER][ERROR][%s] get same var_name:\"%s\"\n" C_END,
        name_.c_str(), name.c_str());
      return true;
    } else {checker.insert(name);}
    return false;
  }

  void var_area_error(
    canid_t can_id,
    int data_l,
    int data_h,
    std::map<canid_t, std::vector<uint8_t>> & checker)
  {
    uint8_t mask = 0xFF;
    uint8_t conflict = 0x00;
    bool first_index = true;
    for (int index = data_l; index <= data_h; index++) {
      conflict = checker.at(can_id)[index] & mask;
      if (conflict != 0x0) {
        if (first_index) {
          first_index = false;
          error_clct_->LogState(ErrorCode::DATA_AREA_CONFLICT);
          printf(
            C_RED "[CAN_PARSER][ERROR][%s] data area decode/encode many times at pos'*':\n"
            "\tcan_id[0x%08x],DATA[%d]%s\n" C_END,
            name_.c_str(),
            can_id, index, show_conflict(conflict).c_str());
        } else {
          printf(
            C_RED "\t                   DATA[%d]%s\n" C_END,
            index, show_conflict(mask).c_str());
        }
      }
      checker.at(can_id)[index] |= mask;
    }
  }

  void check_data_area_error(RuleVar & rule, std::map<canid_t, std::vector<uint8_t>> & checker)
  {
    if (checker.find(rule.can_id) == checker.end()) {
      checker.insert(
        std::pair<canid_t, std::vector<uint8_t>>(
          rule.can_id, std::vector<uint8_t>(CAN_LEN())));
    }

    if (rule.parser_type == "bit") {
      uint8_t data_index = rule.parser_param[0];
      uint8_t mask = creat_mask(rule.parser_param[1], rule.parser_param[2]);
      uint8_t conflict = checker.at(rule.can_id)[data_index] & mask;
      if (conflict != 0x0) {
        error_clct_->LogState(ErrorCode::DATA_AREA_CONFLICT);
        printf(
          C_RED "[CAN_PARSER][ERROR][%s] data area decode/encode many times at pos'*':\n"
          "\tcan_id[0x%08x],DATA[%d]%s\n" C_END,
          name_.c_str(), rule.can_id, data_index, show_conflict(conflict).c_str());
      }
      checker.at(rule.can_id)[data_index] |= mask;
    } else if (rule.parser_type == "var") {
      var_area_error(rule.can_id, rule.parser_param[0], rule.parser_param[1], checker);
    }
  }

  void check_data_area_error(ArrayRule & rule, std::map<canid_t, std::vector<uint8_t>> & checker)
  {
    for (auto & can_id : rule.can_id) {
      if (checker.find(can_id.first) == checker.end()) {
        checker.insert(
          std::pair<canid_t, std::vector<uint8_t>>(
            can_id.first, std::vector<uint8_t>(CAN_LEN())));
      }
      var_area_error(can_id.first, 0, CAN_LEN() - 1, checker);
    }
  }
};  // class CanParser
}  // namespace common
}  // namespace cyberdog


#endif  // COMMON_PARSER__CAN_PARSER_HPP_
