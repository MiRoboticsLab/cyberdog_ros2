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

#ifndef FAKE_DATA_HPP_
#define FAKE_DATA_HPP_

#include <map>
#include <vector>

#include "device_display/strip_light.hpp"

namespace cyberdog
{
namespace device
{
enum EffectType
{
  HOLDON = 0,
  BLINK = 1,
  BREATH = 2
};

// data for testing
const std::map<uint32_t, LightTargetT> modules_test_real = {
  {1, {1, 1, "mono light"}}, {2, {2, 3, "three mono lights"}}, {3, {3, 3, "three rgb lights"}}};
const std::map<uint32_t, LightTargetT> modules_test_fake = {};
const std::map<uint32_t, LightTargetT> modules_test_single = {{5, {5, 2, "two mono lights"}}};
const LightTargetT test_module = {4, 2, "two rgb lights"};
const LightEffect effect_mono_1 = 10;
const LightEffect effect_mono_2 = 127;
const LightEffect effect_mono_3 = 255;
const LightEffect effect_rgb_1 = 255 << RED_SHIFT | 106 << GREEN_SHIFT | 106 << BLUE_SHIFT;
const LightEffect effect_rgb_2 = 255 << RED_SHIFT | 185 << GREEN_SHIFT | 15 << BLUE_SHIFT;
const LightEffect effect_rgb_3 = 0 << RED_SHIFT | 206 << GREEN_SHIFT | 209 << BLUE_SHIFT;
const LightArgV effect_frames_single =
{{{effect_mono_2}, EffectType::HOLDON, ""}};
const LightArgV effect_frames_multiple =
{{{effect_mono_1, effect_mono_2, effect_mono_3}, EffectType::BLINK, "300000"}};
const LightArgV effect_frames_rgb_a =
{{{effect_rgb_1, effect_rgb_2, effect_rgb_3}, EffectType::BREATH, "125000"}};
const LightArgV effect_frames_rgb_b =
{{{effect_rgb_3, effect_rgb_3, effect_rgb_3}, EffectType::BREATH, "125000"}};
const LightEffectMapT test_effects_three = {
  {1, effect_frames_single}, {2, effect_frames_multiple}, {4, effect_frames_rgb_a}};
const LightEffectMapT test_effects_full = {
  {1, effect_frames_single}, {2, effect_frames_multiple}, {3, effect_frames_rgb_a},
  {4, effect_frames_rgb_b}};

}  // namespace device
}  // namespace cyberdog

#endif  // FAKE_DATA_HPP_
