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


#ifndef BMS_COMMON_HPP_
#define BMS_COMMON_HPP_
#include "std_msgs/msg/string.hpp"
#include "ception_msgs/msg/bms.hpp"
#include "ception_msgs/srv/bms_info.hpp"
#include "lcm_translate_msgs/bms_response_lcmt.hpp"

#define MIN_FULL_SOC                100
#define MIN_FULL_VOLTAGE_MV         24900
#define MIN_FULL_CURRENT_MA         600

#define CHARGING_STATUS_DISCHARGE   0
#define CHARGING_STATUS_CHARGING    1
#define CHARGING_STATUS_FULL        2
#define CHARGING_STATUS_UNKNOWN     3
#define SOFT_SHUTDOWN_BIT 0
#define IS_BIT_SET(number, n) ((number >> n) & (0x1))
typedef enum
{
  USB_INSERT_BIT  = 0,
  BATTERY_LOW_BIT = 1,
  CHARING_BIT = 2,
  CHARGED_BIT = 3,
  SHORT_CUT_BIT = 4,
  OVER_TEMP_BIT = 5,
  BMS_ERROR_BIT = 6
} BMS_STATUS_BIT_T;

int log_file_status_check(void);
int bms_log_store(const bms_response_lcmt * msg);
int convert_status(const bms_response_lcmt * lcm_data);
#endif  // BMS_COMMON_HPP_
