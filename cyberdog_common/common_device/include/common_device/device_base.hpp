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
#include <iostream>
#include <functional>

#include "toml11/toml.hpp"
#include "common_device/common.hpp"

namespace common_device
{
template<typename T>
class device_base
{
public:
  // !! DO NOT use this except with LINK_VAR !!
  std::shared_ptr<T> DATA() {return device_data_;}
  // !! DO NOT use this except with LINK_VAR !!
  std::map<std::string, device_data_var> & VAR_MAP() {return device_var_map_;}

  void SetDataCallback(std::function<void(std::shared_ptr<T>)> callback)
  {
    if (callback != nullptr) {devicedata_callback_ = callback;}
  }
  virtual bool Operate(
    const std::string & CMD,
    const std::vector<uint8_t> & data = std::vector<uint8_t>()) = 0;

  virtual int GetErrorNum() = 0;
  virtual int GetWarnNum() = 0;

protected:
  device_base()
  {
    device_data_ = std::make_shared<T>();
    device_var_map_ = std::map<std::string, device_data_var>();
  }
  ~device_base() {}

  std::shared_ptr<T> device_data_;
  std::map<std::string, device_data_var> device_var_map_;
  std::function<void(std::shared_ptr<T>)> devicedata_callback_;
};  // class device_base
}  // namespace common_device

#endif  // COMMON_DEVICE__DEVICE_BASE_HPP_
