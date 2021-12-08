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

#ifndef COMMON_DEVICE__DEVICE_BASE_HPP_
#define COMMON_DEVICE__DEVICE_BASE_HPP_

#include <map>
#include <memory>
#include <string>
#include <vector>
#include <utility>
#include <iostream>
#include <functional>

#include "toml11/toml.hpp"
#include "common_device/common.hpp"

namespace common_device
{
#define MIN_TIME_OUT_US     1'000L  // 1ms
#define MAX_TIME_OUT_US 3'000'000L  // 3s

template<typename TDataClass>
class device_base
{
public:
  std::shared_ptr<TDataClass> GetData() {return device_data_;}

  void SetDataCallback(std::function<void(std::shared_ptr<TDataClass>)> callback)
  {
    if (for_send_) {
      printf(
        C_YELLOW "[DEVICE][WARN][%s] for_send device not need callback function, "
        "please check the code\n" C_END, name_.c_str());
    }
    if (callback != nullptr) {devicedata_callback_ = callback;}
  }

  void LinkVar(const std::string & name, const device_data & var)
  {
    device_data_map_.insert(std::pair<std::string, device_data>(name, var));
  }

  virtual bool Operate(
    const std::string & CMD,
    const std::vector<uint8_t> & data = std::vector<uint8_t>()) = 0;
  virtual bool SendSelfData() = 0;

  virtual int GetErrorNum() = 0;
  virtual int GetWarnNum() = 0;

protected:
  device_base()
  {
    device_data_ = std::make_shared<TDataClass>();
    device_data_map_ = std::map<std::string, device_data>();
  }
  ~device_base() {}

  bool for_send_;
  std::string name_;
  std::shared_ptr<TDataClass> device_data_;
  std::map<std::string, device_data> device_data_map_;
  std::function<void(std::shared_ptr<TDataClass>)> devicedata_callback_;
};  // class device_base
}  // namespace common_device

#endif  // COMMON_DEVICE__DEVICE_BASE_HPP_
