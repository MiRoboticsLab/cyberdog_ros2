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

#include "device_display/strip_light.hpp"
#include "fake_data.hpp"
#include "gtest/gtest.h"
#include "pluginlib/class_loader.hpp"

TEST(StripLightTest, initTestSucceed) {
  pluginlib::ClassLoader<cyberdog::device::StripLight> light_loader(
    "light_pluginlib", "cyberdog::device::StripLight");
  ASSERT_TRUE(
    light_loader.createUniqueInstance("light_pluginlib/StripLightInstance")->init(
      cyberdog::device::modules_test_real, cyberdog::device::test_effects_full));
}

TEST(StripLightTest, initTestFailed) {
  pluginlib::ClassLoader<cyberdog::device::StripLight> light_loader(
    "light_pluginlib", "cyberdog::device::StripLight");
  auto test_instance = light_loader.createUniqueInstance("light_pluginlib/StripLightInstance");
  ASSERT_FALSE(
    test_instance->init(
      cyberdog::device::modules_test_fake, cyberdog::device::test_effects_full));
  ASSERT_FALSE(
    test_instance->init(
      cyberdog::device::modules_test_real, cyberdog::device::test_effects_three));
}

TEST(StripLightTest, notInitTestFailed) {
  pluginlib::ClassLoader<cyberdog::device::StripLight> light_loader(
    "light_pluginlib", "cyberdog::device::StripLight");
  auto test_instance = light_loader.createUniqueInstance("light_pluginlib/StripLightInstance");
  ASSERT_FALSE(test_instance->pause());
  ASSERT_FALSE(
    test_instance->set_mode(
      cyberdog::device::modules_test_real.at(1), cyberdog::device::LightMode::SET));
  ASSERT_FALSE(
    test_instance->set_arg(
      cyberdog::device::modules_test_real.at(1), 1, cyberdog::device::effect_frames_single));
  ASSERT_FALSE(
    test_instance->send_with_value(
      cyberdog::device::modules_test_real.at(1), cyberdog::device::effect_frames_single));
  ASSERT_FALSE(
    test_instance->send_with_key(
      cyberdog::device::modules_test_real.at(1), 1, 1000));
  uint32_t status(cyberdog::device::LightMode::DEFAULT);
  ASSERT_FALSE(test_instance->get_status(cyberdog::device::modules_test_real.at(1), status));
  ASSERT_EQ(status, static_cast<uint32_t>(cyberdog::device::LightMode::DEFAULT));
}

TEST(StripLightTest, setModes) {
  pluginlib::ClassLoader<cyberdog::device::StripLight> light_loader(
    "light_pluginlib", "cyberdog::device::StripLight");
  auto test_instance = light_loader.createUniqueInstance("light_pluginlib/StripLightInstance");
  ASSERT_TRUE(
    test_instance->init(
      cyberdog::device::modules_test_real, cyberdog::device::test_effects_full));
  ASSERT_TRUE(
    test_instance->set_mode(
      cyberdog::device::modules_test_real.at(1), cyberdog::device::LightMode::RUNNING));
  ASSERT_FALSE(
    test_instance->set_mode(
      cyberdog::device::modules_test_single.at(5), cyberdog::device::LightMode::RUNNING));
  uint32_t status(cyberdog::device::LightMode::DEFAULT);
  ASSERT_TRUE(test_instance->get_status(cyberdog::device::modules_test_real.at(1), status));
  ASSERT_EQ(status, static_cast<uint32_t>(cyberdog::device::LightMode::RUNNING));
}

TEST(StripLightTest, setIDs) {
  pluginlib::ClassLoader<cyberdog::device::StripLight> light_loader(
    "light_pluginlib", "cyberdog::device::StripLight");
  auto test_instance = light_loader.createUniqueInstance("light_pluginlib/StripLightInstance");
  ASSERT_TRUE(
    test_instance->init(
      cyberdog::device::modules_test_real, cyberdog::device::test_effects_full));
  ASSERT_TRUE(
    test_instance->set_mode(
      cyberdog::device::modules_test_real.at(1), cyberdog::device::LightMode::SET));
  auto new_effect = cyberdog::device::effect_frames_single;
  new_effect.at(0).effect_type = cyberdog::device::EffectType::BLINK;
  ASSERT_TRUE(
    test_instance->set_arg(
      cyberdog::device::modules_test_real.at(1), 5, new_effect));
  ASSERT_TRUE(
    test_instance->set_arg(
      cyberdog::device::modules_test_real.at(1), 1, new_effect));
  ASSERT_TRUE(
    test_instance->set_mode(
      cyberdog::device::modules_test_real.at(1), cyberdog::device::LightMode::RUNNING));
  ASSERT_TRUE(
    test_instance->send_with_key(
      cyberdog::device::modules_test_real.at(1), 5, 1000));
}

TEST(StripLightTest, runIDs) {
  pluginlib::ClassLoader<cyberdog::device::StripLight> light_loader(
    "light_pluginlib", "cyberdog::device::StripLight");
  auto test_instance = light_loader.createUniqueInstance("light_pluginlib/StripLightInstance");
  ASSERT_TRUE(
    test_instance->init(
      cyberdog::device::modules_test_real, cyberdog::device::test_effects_full));
  ASSERT_TRUE(
    test_instance->set_mode(
      cyberdog::device::modules_test_real.at(1), cyberdog::device::LightMode::RUNNING));
  ASSERT_FALSE(
    test_instance->send_with_key(
      cyberdog::device::modules_test_real.at(1), 2, 1000));
  ASSERT_FALSE(
    test_instance->send_with_key(
      cyberdog::device::modules_test_single.at(5), 1, 1000));
  ASSERT_TRUE(
    test_instance->send_with_key(
      cyberdog::device::modules_test_real.at(1), 1, 1000));
}

TEST(StripLightTest, getStatus) {
  pluginlib::ClassLoader<cyberdog::device::StripLight> light_loader(
    "light_pluginlib", "cyberdog::device::StripLight");
  auto test_instance = light_loader.createUniqueInstance("light_pluginlib/StripLightInstance");
  ASSERT_TRUE(
    test_instance->init(
      cyberdog::device::modules_test_real, cyberdog::device::test_effects_full));
  ASSERT_TRUE(
    test_instance->set_mode(
      cyberdog::device::modules_test_real.at(1), cyberdog::device::LightMode::TEST));
  uint32_t status(cyberdog::device::LightMode::DEFAULT);
  ASSERT_TRUE(test_instance->get_status(cyberdog::device::modules_test_real.at(1), status));
  ASSERT_EQ(status, static_cast<uint32_t>(cyberdog::device::LightMode::TEST));
  ASSERT_TRUE(test_instance->pause());
  ASSERT_TRUE(test_instance->get_status(cyberdog::device::modules_test_real.at(2), status));
  ASSERT_EQ(status, static_cast<uint32_t>(cyberdog::device::LightMode::PAUSE));
  ASSERT_FALSE(test_instance->get_status(cyberdog::device::modules_test_single.at(5), status));
  ASSERT_EQ(status, static_cast<uint32_t>(cyberdog::device::LightMode::PAUSE));
}

TEST(StripLightTest, testFrames) {
  pluginlib::ClassLoader<cyberdog::device::StripLight> light_loader(
    "light_pluginlib", "cyberdog::device::StripLight");
  auto test_instance = light_loader.createUniqueInstance("light_pluginlib/StripLightInstance");
  ASSERT_TRUE(
    test_instance->init(
      cyberdog::device::modules_test_real, cyberdog::device::test_effects_full));
  ASSERT_TRUE(
    test_instance->set_mode(
      cyberdog::device::modules_test_real.at(1), cyberdog::device::LightMode::TEST));
  uint32_t status(cyberdog::device::LightMode::DEFAULT);
  ASSERT_TRUE(test_instance->get_status(cyberdog::device::modules_test_real.at(1), status));
  ASSERT_EQ(status, static_cast<uint32_t>(cyberdog::device::LightMode::TEST));
  ASSERT_TRUE(
    test_instance->send_with_value(
      cyberdog::device::modules_test_real.at(1), cyberdog::device::effect_frames_single));
  ASSERT_FALSE(
    test_instance->send_with_value(
      cyberdog::device::test_module, cyberdog::device::effect_frames_single));
  ASSERT_TRUE(
    test_instance->set_mode(
      cyberdog::device::modules_test_real.at(1), cyberdog::device::LightMode::RUNNING));
  ASSERT_FALSE(
    test_instance->send_with_value(
      cyberdog::device::modules_test_real.at(1), cyberdog::device::effect_frames_single));
}

int main(int argc, char ** argv)
{
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
