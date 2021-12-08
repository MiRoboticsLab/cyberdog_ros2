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

#ifndef COMMON_DEVICE__COMMON_DEVICE_HPP_
#define COMMON_DEVICE__COMMON_DEVICE_HPP_

#include <string>
#include <utility>
#include <memory>
#include <vector>

#include "common_device/device_base.hpp"
#include "common_device/can_device.hpp"

#define XNAME(x) (#x)
#define LINK_VAR(var) LinkVar( \
    common_device::get_var_name(XNAME(var)), \
    common_device::device_data(sizeof((var)), static_cast<void *>(&(var))))


namespace common_device
{
template<typename TDataClass>
class device
{
public:
  device(const device &) = delete;
  explicit device(const std::string & device_toml_path)
  {
    toml::value toml_config;
    if (toml_parse(toml_config, device_toml_path) == false) {
      printf(
        C_RED "[DEVICE][ERROR] toml file:\"%s\" error, load prebuilt file\n" C_END,
        device_toml_path.c_str());
      // TBD toml_config = ;
    }

    Init(toml_config, device_toml_path);
    if (base == nullptr || base->GetErrorNum() != 0) {
      printf(
        C_RED "[DEVICE][ERROR] toml file:\"%s\" init error, load prebuilt file\n" C_END,
        device_toml_path.c_str());
      // TBD Init();
    }
  }

  std::shared_ptr<TDataClass> GetData()
  {
    if (base != nullptr) {return base->GetData();}
    if (tmp_data == nullptr) {tmp_data = std::make_shared<TDataClass>();}
    return tmp_data;
  }

  // please use "#define LINK_VAR(var)" instead
  void LinkVar(const std::string name, const device_data & var)
  {
    if (base != nullptr) {base->LinkVar(name, var);}
  }

  bool Operate(const std::string & CMD, const std::vector<uint8_t> & data = std::vector<uint8_t>())
  {
    if (base != nullptr) {return base->Operate(CMD, data);}
    return false;
  }

  bool SendSelfData()
  {
    if (base != nullptr) {return base->SendSelfData();}
    return false;
  }

  void SetDataCallback(std::function<void(std::shared_ptr<TDataClass>)> callback)
  {
    if (base != nullptr) {base->SetDataCallback(callback);}
  }

private:
  std::shared_ptr<device_base<TDataClass>> base;
  std::shared_ptr<TDataClass> tmp_data;

  void Init(const toml::value & toml_config, const std::string & device_toml_path = "")
  {
    auto protocol = toml::find_or<std::string>(toml_config, "protocol", "#unknow");
    auto name = toml::find_or<std::string>(toml_config, "name", "#unknow");
    if (device_toml_path != "") {
      printf(
        "[DEVICE][INFO] Creat common device[%s], protocol:\"%s\", path:\"%s\"\n",
        name.c_str(), protocol.c_str(), device_toml_path.c_str());
    }

    if (protocol == "can") {
      auto can_interface = toml::find_or<std::string>(toml_config, "can_interface", "can0");
      base = std::make_shared<can_device<TDataClass>>(can_interface, name, toml_config);
    } else if (protocol == "spi") {
      // todo when need
      printf(C_RED "[DEVICE][ERROR] protocol:\"%s\" not support yet\n" C_END, protocol.c_str());
    } else if (protocol == "iic" || protocol == "i2c") {
      // todo when need
      printf(C_RED "[DEVICE][ERROR] protocol:\"%s\" not support yet\n" C_END, protocol.c_str());
    } else {
      printf(
        C_RED "[DEVICE][ERROR][%s] protocol:\"%s\" not support, parser path=\"%s\"\n" C_END,
        name.c_str(), protocol.c_str(), device_toml_path.c_str());
    }
  }
};  // class device
}  // namespace common_device

#endif  // COMMON_DEVICE__COMMON_DEVICE_HPP_
