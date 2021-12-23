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

#ifndef COMMON_PROTOCOL__PROTOCOL_BASE_HPP_
#define COMMON_PROTOCOL__PROTOCOL_BASE_HPP_

#include <map>
#include <memory>
#include <string>
#include <vector>
#include <utility>
#include <iostream>
#include <functional>

#include "toml11/toml.hpp"
#include "common_protocol/common.hpp"

namespace cyberdog
{
namespace common
{
#define MIN_TIME_OUT_US     1'000L  // 1ms
#define MAX_TIME_OUT_US 3'000'000L  // 3s

template<typename TDataClass>
class ProtocolBase
{
public:
  std::shared_ptr<TDataClass> GetData() {return protocol_data_;}

  void SetDataCallback(std::function<void(std::shared_ptr<TDataClass>)> callback)
  {
    if (for_send_) {
      printf(
        C_YELLOW "[PROTOCOL][WARN][%s] for_send protocol not need callback function, "
        "please check the code\n" C_END, name_.c_str());
    }
    if (callback != nullptr) {protocol_data_callback_ = callback;}
  }

  void LinkVar(const std::string & name, const ProtocolData & var)
  {
    if (protocol_data_map_.find(name) != protocol_data_map_.end()) {
      error_clct_->LogState(ErrorCode::RUNTIME_SAMELINK_ERROR);
      printf(
        C_RED "[PROTOCOL][ERROR][%s] LINK_VAR error, get same name:\"%s\"\n" C_END,
        name_.c_str(), name.c_str());
      return;
    }
    protocol_data_map_.insert(std::pair<std::string, ProtocolData>(name, var));
  }

  virtual bool Operate(
    const std::string & CMD,
    const std::vector<uint8_t> & data = std::vector<uint8_t>()) = 0;
  virtual bool SendSelfData() = 0;

  virtual int GetInitErrorNum() = 0;
  virtual int GetInitWarnNum() = 0;

  virtual bool IsRxTimeout() = 0;
  virtual bool IsTxTimeout() = 0;
  bool IsRxError() {return rx_error_;}

protected:
  ProtocolBase()
  {
    protocol_data_ = std::make_shared<TDataClass>();
    protocol_data_map_ = PROTOCOL_DATA_MAP();
  }
  ~ProtocolBase() {}

  bool for_send_;
  bool rx_error_;
  std::string name_;
  CHILD_STATE_CLCT error_clct_;
  PROTOCOL_DATA_MAP protocol_data_map_;
  std::shared_ptr<TDataClass> protocol_data_;
  std::function<void(std::shared_ptr<TDataClass>)> protocol_data_callback_;
};  // class ProtocolBase
}  // namespace common
}  // namespace cyberdog

#endif  // COMMON_PROTOCOL__PROTOCOL_BASE_HPP_
