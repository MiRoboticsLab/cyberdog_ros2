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

#include "audio_base/debug/ai_debugger.hpp"

namespace std
{
bool audioDebugger::mAsForceEnable = false;
bool audioDebugger::mAudioDumpRaw = false;
bool audioDebugger::mAudioDumpAsr = false;
bool audioDebugger::mAudioDebugTesttoken = false;
bool audioDebugger::mAudioDebugSilence = false;

audioDebugger::audioDebugger()
{
  std::cout << "Creating audioDebugger." << std::endl;
}

audioDebugger::~audioDebugger()
{
  std::cout << "Destroying audioDebugger" << std::endl;
}

bool audioDebugger::getConfig(audio_debug_config_t config)
{
  bool enalbed = false;

  switch (config) {
    case AUDIO_DEBUG_CONFIG_AS_FORCE_ENABLE:
      enalbed = mAsForceEnable;
      break;

    case AUDIO_DEBUG_CONFIG_DUMPRAW:
      enalbed = mAudioDumpRaw;
      break;

    case AUDIO_DEBUG_CONFIG_DUMPASR:
      enalbed = mAudioDumpAsr;
      break;

    case AUDIO_DEBUG_CONFIG_SILENCE:
      enalbed = mAudioDebugSilence;
      break;

    case AUDIO_DEBUG_CONFIG_TESTTOKEN:
      enalbed = mAudioDebugTesttoken;
      break;

    default:
      std::cout << "audioDebugger::getConfig() no item matched !!!" << std::endl;
      break;
  }
  std::cout << "audioDebugger::getConfig() " << enalbed << std::endl;
  return enalbed;
}

void audioDebugger::setConfig(audio_debug_config_t config, bool enable)
{
  switch (config) {
    case AUDIO_DEBUG_CONFIG_AS_FORCE_ENABLE:
      mAsForceEnable = enable;
      break;

    case AUDIO_DEBUG_CONFIG_DUMPRAW:
      mAudioDumpRaw = enable;
      break;

    case AUDIO_DEBUG_CONFIG_DUMPASR:
      mAudioDumpAsr = enable;
      break;

    case AUDIO_DEBUG_CONFIG_SILENCE:
      mAudioDebugSilence = enable;
      break;

    case AUDIO_DEBUG_CONFIG_TESTTOKEN:
      mAudioDebugTesttoken = enable;
      break;

    default:
      std::cout << "audioDebugger::setConfig() no item matched !" << std::endl;
      break;
  }
  std::cout << "audioDebugger::setConfig() " << enable << std::endl;
}

int audioDebugger::audioDumpFileOpen(audio_debug_config_t dumpConfig, int * fd_p)
{
  time_t timep;
  time(&timep);
  if (dumpConfig == AUDIO_DEBUG_CONFIG_DUMPRAW) {
    strftime(
      rawDumpFile, sizeof(rawDumpFile),
      "/home/mi/Downloads/audio_dump_raw_7.16.16k-%Y_%m_%d_%H_%M_%S.pcm",
      localtime(&timep));
  } else if (dumpConfig == AUDIO_DEBUG_CONFIG_DUMPASR) {
    strftime(
      rawDumpFile, sizeof(rawDumpFile),
      "/home/mi/Downloads/audio_dump_asr_1.16.16k-%Y_%m_%d_%H_%M_%S.pcm",
      localtime(&timep));
  } else {
    return -1;
  }
  std::cout << "dump data name " << rawDumpFile << std::endl;
  *fd_p = open(rawDumpFile, O_CREAT | O_RDWR, 0777);
  return (*fd_p > 0) ? 0 : -1;
}

int audioDebugger::audioDumpFileWrite(int * fd_p, unsigned char * buf, unsigned int len)
{
  if (*fd_p < 0) {
    std::cout << "Write dump invalid paramters " << std::endl;
    return -1;
  }
  write(*fd_p, buf, len);
  return 0;
}

int audioDebugger::audioDumpFileClose(audio_debug_config_t dumpConfig, int * fd_p)
{
  if ((*fd_p < 0) ||
    ((dumpConfig != AUDIO_DEBUG_CONFIG_DUMPRAW) &&
    (dumpConfig != AUDIO_DEBUG_CONFIG_DUMPASR)))
  {
    std::cout << "Write dump invalid paramters " << std::endl;
    return -1;
  }
  close(*fd_p);
  return 0;
}
}  // namespace std
