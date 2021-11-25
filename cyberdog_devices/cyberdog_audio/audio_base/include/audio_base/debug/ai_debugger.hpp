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

#ifndef AUDIO_BASE__DEBUG__AI_DEBUGGER_HPP_
#define AUDIO_BASE__DEBUG__AI_DEBUGGER_HPP_

#include <iostream>
#include "sys/fcntl.h"
#include "sys/unistd.h"

#define DUMP_FILE_NAME_LEN 128

typedef enum
{
  AUDIO_DEBUG_CONFIG_AS_FORCE_ENABLE,
  AUDIO_DEBUG_CONFIG_DUMPRAW,
  AUDIO_DEBUG_CONFIG_DUMPASR,
  AUDIO_DEBUG_CONFIG_SILENCE,
  AUDIO_DEBUG_CONFIG_TESTTOKEN,
  AUDIO_DEBUG_CONFIG_MAX
} audio_debug_config_t;

namespace std
{
class audioDebugger
{
public:
  static bool mAsForceEnable;
  static bool mAudioDumpRaw;
  static bool mAudioDumpAsr;
  static bool mAudioDebugSilence;
  static bool mAudioDebugTesttoken;

  audioDebugger();
  ~audioDebugger();
  bool getConfig(audio_debug_config_t config);
  void setConfig(audio_debug_config_t config, bool enable);
  int audioDumpFileOpen(audio_debug_config_t dumpConfig, int * fd_p);
  int audioDumpFileWrite(int * fd_p, unsigned char * buf, unsigned int len);
  int audioDumpFileClose(audio_debug_config_t dumpConfig, int * fd_p);

protected:
private:
  char rawDumpFile[DUMP_FILE_NAME_LEN] = {'\0'};
};      // class audioDebugger
}  // namespace std

#endif  // AUDIO_BASE__DEBUG__AI_DEBUGGER_HPP_
