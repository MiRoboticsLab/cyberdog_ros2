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

#ifndef LED_BASE__LED_BASE_HPP_
#define LED_BASE__LED_BASE_HPP_

namespace led_base
{
class Cyberdog_LED
{
  virtual bool Init() = 0;
  virtual bool Set() = 0;
  virtual bool Play() = 0;

protected:
  Cyberdog_LED() {}
};  // class Cyberdog_LED
}  // led_base

#endif // LED_BASE__LED_BASE_HPP_
