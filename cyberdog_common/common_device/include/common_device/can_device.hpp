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

#ifndef COMMON_DEVICE__CAN_DEVICE_HPP_
#define COMMON_DEVICE__CAN_DEVICE_HPP_

#include <string>
#include <memory>
#include <vector>
#include <algorithm>

#include "common_device/common.hpp"
#include "common_device/device_base.hpp"
#include "common_parser/can_parser.hpp"

namespace common_device
{
template<typename TDataClass>
class can_device : public device_base<TDataClass>
{
public:
  can_device(
    const std::string & interface,
    const std::string & name,
    const toml::value & toml_config)
  {
    this->name_ = name;
    this->for_send_ = toml::find_or<bool>(toml_config, "for_send", false);

    auto extended_frame = toml::find_or<bool>(toml_config, "extended_frame", false);
    auto canfd_enable = toml::find_or<bool>(toml_config, "canfd_enable", false);
    auto timeout_us = toml::find_or<int64_t>(toml_config, "timeout_us", MAX_TIME_OUT_US);
    timeout_us = std::clamp(timeout_us, MIN_TIME_OUT_US, MAX_TIME_OUT_US);

    can_parser_ = std::make_shared<can_parser>(toml_config, this->name_, canfd_enable);
    printf(
      "[CAN_DEVICE][INFO] Creat can device[%s]: %d error, %d warning\n",
      this->name_.c_str(), can_parser_->GetInitErrorNum(), can_parser_->GetInitWarnNum());
    auto recv_list = can_parser_->GetRecvList();
    int recv_num = recv_list.size();
    bool send_only = (recv_num == 0) ? true : this->for_send_;

    if (send_only) {
      printf("[CAN_DEVICE][INFO][%s] No recv canid, enable send-only mode\n", this->name_.c_str());
      can_op_ = std::make_shared<cyberdog_utils::can_dev>(
        interface,
        this->name_,
        extended_frame,
        canfd_enable,
        timeout_us * 1000);
    } else {
      can_op_ = std::make_shared<cyberdog_utils::can_dev>(
        interface,
        this->name_,
        extended_frame,
        std::bind(&can_device::recv_callback, this, std::placeholders::_1),
        timeout_us * 1000);
    }

    // set can_filter
    if (can_op_ != nullptr && send_only == false) {
      auto filter = new struct can_filter[recv_num];
      for (int a = 0; a < recv_num; a++) {
        filter[a].can_id = recv_list[a];
        filter[a].can_mask = CAN_EFF_MASK;
      }
      can_op_->set_filter(filter, recv_num * sizeof(struct can_filter));
      delete[] filter;
    }
  }
  ~can_device() {}

  bool Operate(
    const std::string & CMD,
    const std::vector<uint8_t> & data = std::vector<uint8_t>()) override
  {
    if (can_parser_->IsCanfd() == false) {
      can_frame tx_frame;
      if (can_parser_->Encode(tx_frame, CMD, data) &&
        can_op_ != nullptr && can_op_->send_can_message(tx_frame))
      {
        return true;
      } else {
        printf(
          C_RED "[CAN_DEVICE][ERROR][%s] Operate CMD:\"%s\" sending data error\n" C_END,
          this->name_.c_str(), CMD.c_str());
      }
    } else {
      canfd_frame tx_frame;
      if (can_parser_->Encode(tx_frame, CMD, data) &&
        can_op_ != nullptr && can_op_->send_can_message(tx_frame))
      {
        return true;
      } else {
        printf(
          C_RED "[CAN_DEVICE][ERROR][%s] Operate CMD:\"%s\" sending data error\n" C_END,
          this->name_.c_str(), CMD.c_str());
      }
    }
    return false;
  }

  bool SendSelfData() override
  {
    if (this->for_send_ == false) {
      printf(
        C_YELLOW "[CAN_DEVICE][WARN][%s] Device not in sending mode, "
        "should not send data from data class\n" C_END,
        this->name_.c_str());
    }
    return can_parser_->Encode(this->device_data_map_, can_op_);
  }

  int GetInitErrorNum() override {return can_parser_->GetInitErrorNum();}
  int GetInitWarnNum() override {return can_parser_->GetInitWarnNum();}

  virtual bool IsRxTimeout() override {return can_op_->is_rx_timeout();}
  virtual bool IsTxTimeout() override {return can_op_->is_tx_timeout();}

private:
  std::shared_ptr<can_parser> can_parser_;
  std::shared_ptr<cyberdog_utils::can_dev> can_op_;
  void recv_callback(std::shared_ptr<can_frame> recv_frame)
  {
    if (can_parser_->Decode(this->device_data_map_, recv_frame, this->rx_error_) &&
      this->devicedata_callback_ != nullptr)
    {
      this->rx_error_ = false;
      this->devicedata_callback_(this->device_data_);
    }
  }
};  // class can_device
}  // namespace common_device

#endif  // COMMON_DEVICE__CAN_DEVICE_HPP_
