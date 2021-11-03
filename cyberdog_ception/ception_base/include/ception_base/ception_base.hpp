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

#ifndef CEPTION_BASE__CEPTION_BASE_HPP_
#define CEPTION_BASE__CEPTION_BASE_HPP_

#include <memory>
#include <functional>

namespace ception_base
{
class Cyberdog_GPS_payload
{
public:
  uint32_t iTOW;
  uint8_t fixType;
  uint8_t numSV;
  int32_t lon;
  int32_t lat;
};  // class Cyberdog_GPS_payload

class Cyberdog_GPS
{
public:
  virtual void Open() = 0;
  virtual void Start() = 0;
  virtual void Stop() = 0;
  virtual void Close() = 0;
  virtual ~Cyberdog_GPS() {}
  void SetPayloadCallback(std::function<void(std::shared_ptr<Cyberdog_GPS_payload> payload)> cb)
  {
    payload_callback_ = cb;
  }

protected:
  std::function<void(std::shared_ptr<Cyberdog_GPS_payload> payload)> payload_callback_;
  Cyberdog_GPS() {}
};  // class Cyberdog_GPS
}  // namespace ception_base

#endif  // CEPTION_BASE__CEPTION_BASE_HPP_
