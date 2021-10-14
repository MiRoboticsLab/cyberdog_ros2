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

#ifndef AUDIO_ASSISTANT__COMBINER_HPP_
#define AUDIO_ASSISTANT__COMBINER_HPP_

#include <stdlib.h>
#include <cstdlib>
#include <memory>

#include "sys/stat.h"
#include "sys/types.h"

#include "audio_base/audio_player.hpp"
#include "audio_base/audio_queue.hpp"

#ifdef __cplusplus
extern "C"
{
#endif  // __cplusplus
#include <alsa/asoundlib.h>
#ifdef __cplusplus
}
#endif  // __cplusplus

#define DELAY_TIMES 10

extern bool player_end;

void recorder_work_loop(void);
// void *ai_rawdata_recorder(void *);

#endif  // AUDIO_ASSISTANT__COMBINER_HPP_
