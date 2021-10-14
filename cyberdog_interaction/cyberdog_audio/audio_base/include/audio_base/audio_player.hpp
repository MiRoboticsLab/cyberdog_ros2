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

#ifndef AUDIO_BASE__AUDIO_PLAYER_HPP_
#define AUDIO_BASE__AUDIO_PLAYER_HPP_

// C++ headers
#include <map>
#include <queue>
#include <thread>
#include <vector>
#include <memory>
#include <iostream>
#include <functional>
// SDL2 headers
#include "SDL2/SDL.h"
#include "SDL2/SDL_mixer.h"

namespace cyberdog_audio
{
#define ERROR_CHANNEL -2
#define DELAY_CHECK_TIME 1000
#define DEFAULT_PLAY_CHANNEL_NUM 4
#define DEFAULT_VOLUME MIX_MAX_VOLUME

#define AUDIO_FREQUENCY 16000
#define AUDIO_FORMAT MIX_DEFAULT_FORMAT
#define AUDIO_CHANNELS 1
#define AUDIO_CHUCKSIZE 2048

#define INDE_VOLUME_GROUP -1

#define MAX_QUEUE_BUFF_NUM 100

using chuck_ptr = std::shared_ptr<Mix_Chunk>;
using callback = std::function<void (void)>;

class AudioPlayer
{
public:
  explicit AudioPlayer(
    int channel,
    callback finish_callback = nullptr,
    int volume_group = INDE_VOLUME_GROUP,
    int volume = DEFAULT_VOLUME);
  ~AudioPlayer();
  static int GetFreeChannel();
  static bool InitSuccess();

  static bool OpenReference(int buffsize);
  static void CloseReference();
  static bool GetReferenceData(char * buff);
  static bool HaveReferenceData();
  static void ClearReferenceData();

  void SetFinishCallBack(callback finish_callback);
  void SetChuckVolume(int volume);
  int SetVolume(int volume);
  void SetVolumeGroup(int volume_gp, int default_volume);
  int GetVolume();

  void AddPlay(Uint8 * buff, int len);
  void AddPlay(const char * file);
  void StopPlay();
  bool IsPlaying();
  bool InitReady();

private:
  inline static bool init_success_;
  inline static int channelNum_;
  inline static int activeNum_;
  inline static std::vector<int> thread_num_;
  inline static std::vector<int> chuck_volume_;
  inline static std::vector<int> volume_group_;
  inline static std::map<int, std::queue<chuck_ptr>> chucks_;
  inline static std::map<int, std::queue<Uint8 *>> databuff_;
  inline static std::map<int, callback> finish_callbacks_;

  inline static std::queue<Uint8 *> reference_data_;
  inline static SDL_AudioDeviceID reference_id_;
  inline static SDL_AudioSpec obtained_spec_;
  inline static int ref_buffsize_;

  int self_channel_;
  int init_ready_;

  bool Init();
  void Close();
  int SetSingleVolume(int channel, int volume);
  int GetGroupVolume(int volume_group, int default_volume);
  static bool PopEmpty(int channel);
  static void PlayThreadFunc(int channel, int thread_num);
  static void chuckFinish_callback(int channel);
  static void audioRecording_callback(void *, Uint8 * stream, int len);
};  // class AudioPlayer
}  // namespace cyberdog_audio

#endif  // AUDIO_BASE__AUDIO_PLAYER_HPP_
