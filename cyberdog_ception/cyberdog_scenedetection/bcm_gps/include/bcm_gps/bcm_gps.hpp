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

#ifndef BCM_GPS__BCM_GPS_HPP_
#define BCM_GPS__BCM_GPS_HPP_

#include <map>
#include <memory>
#include <thread>
#include <string>
#include <functional>

#include "bream_vendor/bream.h"

namespace bcm_gps
{
// struct LD2BRM_PvtPvtPolledPayload can find in bream_vendor/bream.h
// and bream_vendor URL can find in CMakeLists.txt
typedef LD2BRM_PvtPvtPolledPayload GPS_Payload;

using NMEA_callback = std::function<void (uint8_t * str, uint32_t len)>;
using PAYLOAD_callback = std::function<void (std::shared_ptr<GPS_Payload> payload)>;

class GPS
{
public:
  explicit GPS(PAYLOAD_callback PAYLOAD_cb = nullptr, NMEA_callback NMEA_cb = nullptr);
  ~GPS();
  bool Open();
  void Start();
  void Stop();
  void Close();
  bool Ready();
  void SetCallback(NMEA_callback NMEA_cb);
  void SetCallback(PAYLOAD_callback PAYLOAD_cb);

  void SetL5Bias(uint32_t biasCm);
  void SetLteFilterEn(bool enable = true);

private:
  inline static int all_num_;
  inline static int init_num_;
  inline static int start_num_;
  inline static bool ready_;
  inline static bool main_running_;
  inline static std::thread main_T_;

  int id_;
  bool opened_ = false;
  bool start_ = false;
  bool Init();
  void MainThread();
};  // class GPS
}  // namespace bcm_gps

#endif  // BCM_GPS__BCM_GPS_HPP_
