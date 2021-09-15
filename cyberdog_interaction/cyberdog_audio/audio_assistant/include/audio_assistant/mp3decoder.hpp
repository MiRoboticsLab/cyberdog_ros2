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

#ifndef  AUDIO_ASSISTANT__MP3DECODER_HPP_
#define  AUDIO_ASSISTANT__MP3DECODER_HPP_

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <sys/stat.h>
#include <sys/mman.h>
#include <sys/soundcard.h>
#include <sys/ioctl.h>
#include <sys/fcntl.h>
#include <sys/types.h>
#include <mutex>
#include <memory>

#include "audio_base/audio_player.hpp"
#include "combiner.hpp"

struct buffer
{
  unsigned char const * start;
  uint64_t length;
};
#define PCM_BUF_SIZE_MAX  (1024 * 1024) * 3
#define PCM_SDL_CH  1 /*tts is 1, normal playback 2*/
#define AUDIO_GROUP 1

namespace cyberdog_audio
{
class mp3decoder
{
public:
  mp3decoder();
  ~mp3decoder();

  static std::mutex mutex;
  static std::shared_ptr<AudioPlayer> mTtsPlayer;
  static std::shared_ptr<mp3decoder> decoder_;

  static std::shared_ptr<mp3decoder> GetInstance();
  static int tts_player_init(void);
  static int tts_player_accumulate(const unsigned char * data, unsigned int length);
  static int tts_player_decode_paly(void);
  static int tts_player_stop_now(void);
  static bool tts_player_is_playing(void);
  static int pcm_playback(const unsigned char * src, const unsigned int len);
  int tts_player_callback();

protected:
private:
};  // class mp3decoder
}  // namespace cyberdog_audio

#endif  // AUDIO_ASSISTANT__MP3DECODER_HPP_
