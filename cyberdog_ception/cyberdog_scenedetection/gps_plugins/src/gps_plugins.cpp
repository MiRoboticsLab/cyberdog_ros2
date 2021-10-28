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

#include <memory>

#include "gps_plugins/gps_plugins.hpp"
#include "pluginlib/class_list_macros.hpp"

void gps_plugins::Cyberdog_BCMGPS::Open()
{
  bcmgps_ = std::make_shared<bcm_gps::GPS>();
  bcmgps_->SetCallback(
    std::bind(
      &Cyberdog_BCMGPS::BCMGPS_Payload_callback, this,
      std::placeholders::_1));
}

void gps_plugins::Cyberdog_BCMGPS::Start()
{
  if (bcmgps_ != nullptr) {bcmgps_->Start();}
}

void gps_plugins::Cyberdog_BCMGPS::Stop()
{
  if (bcmgps_ != nullptr) {bcmgps_->Stop();}
}

void gps_plugins::Cyberdog_BCMGPS::Close()
{
  if (bcmgps_ != nullptr) {bcmgps_->Close();}
  bcmgps_ = nullptr;
}

void gps_plugins::Cyberdog_BCMGPS::BCMGPS_Payload_callback(
  std::shared_ptr<bcm_gps::GPS_Payload> payload)
{
  /*
  printf(
    "NAV-PVT : Tow=%u, DateTime=%.4u-%.2u-%.2u %.2u:%.2u:%.2u, FixType=%u, NumSv=%u, \
    LLA=(%f, %f, %f), \
    DoP=%f, leapS:%d, Speed=%f, Heading=%f, Valid=%u \n",
    payload->iTOW, payload->year, payload->month, payload->day, payload->hour, payload->min,
    payload->sec, payload->fixType, payload->numSV,
    payload->lat * 1e-7, payload->lon * 1e-7, payload->hMSL * 1e-3,
    payload->pDOP * 1e-2, payload->leapS, payload->gSpeed * 1e-3, payload->headMot * 1e-5,
    payload->valid
  );
  */
  auto cyberdog_payload = std::make_shared<gps_base::Cyberdog_GPS_payload>();
  cyberdog_payload->iTOW = payload->iTOW;
  cyberdog_payload->lat = payload->lat * 1e-7;
  cyberdog_payload->lon = payload->lon * 1e-7;
  cyberdog_payload->fixType = payload->fixType;
  cyberdog_payload->numSV = payload->numSV;
  if (payload_callback_ != nullptr) {payload_callback_(cyberdog_payload);}
}

PLUGINLIB_EXPORT_CLASS(gps_plugins::Cyberdog_BCMGPS, gps_base::Cyberdog_GPS)
