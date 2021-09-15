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
#include <thread>
#include <iostream>
#include <cstdlib>

#include "sys/stat.h"
#include "sys/fcntl.h"
#include "sys/types.h"
#include "sys/unistd.h"

#ifdef ENABLE_ASSISTANT
#include "audio_assistant/audio_assitant.hpp"
#endif

#include "audio_interaction/voice_cmd.hpp"

#ifdef ENABLE_BACKTRACE
#include "audio_backtrace/back_trace.h"
#endif

int main(int argc, char ** argv)
{
  #ifdef ENABLE_BACKTRACE
  signal(SIGSEGV, signal_handler);
  signal(SIGABRT, signal_handler);
  #endif

  rclcpp::init(argc, argv);
  rclcpp::executors::MultiThreadedExecutor exec_;

  auto node = std::make_shared<cyberdog_audio::VoiceCmd>();
  exec_.add_node(node->get_node_base_interface());

  #ifdef ENABLE_ASSISTANT
  auto node_ai = std::make_shared<cyberdog_audio::AudioAssistant>();
  exec_.add_node(node_ai->get_node_base_interface());
  #endif

  exec_.spin();
  rclcpp::shutdown();

  return 0;
}
