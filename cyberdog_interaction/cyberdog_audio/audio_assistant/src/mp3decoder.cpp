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

/*BEGAIN OF THIS FILE*/
#include <mpg123.h>
#include <stdlib.h>

#include <mutex>
#include <memory>

#include "sys/stat.h"
#include "sys/mman.h"
#include "sys/soundcard.h"
#include "sys/ioctl.h"
#include "sys/fcntl.h"
#include "sys/types.h"
#include "audio_base/audio_player.hpp"
#include "audio_assistant/mp3decoder.hpp"

int msg_playback_complete_broadcaster(int msg);

/*decoder begin*/
unsigned int decoderIndex = 0;
unsigned int pcm_length = 0;
unsigned char pcmBuffer[PCM_BUF_SIZE_MAX];

unsigned int mp3_current_length = 0;
unsigned char mp3Buffer[PCM_BUF_SIZE_MAX];

bool audiodata_silence_mode = true;
int  set_audiodata_silence_mode(bool enable)
{
  audiodata_silence_mode = enable;
  printf("vc:tts: set SSSSS  mode = %d\n", enable);
  return 0;
}

int tts_decode_mpg123(unsigned char const * start, uint64_t length)
{
  int err_ret;
  size_t size;

  printf("vc:tts: enter %s\n", __func__);
  mpg123_init();
  auto mpg_h = mpg123_new(NULL, &err_ret);
  if (err_ret == MPG123_OK) {
    mpg123_param(mpg_h, MPG123_VERBOSE, 2, 0);
    mpg123_open_feed(mpg_h);

    printf("[MPG123]Info, MP3 decode get data %ld\n", length);
    err_ret = mpg123_decode(mpg_h, start, length, pcmBuffer, PCM_BUF_SIZE_MAX, &size);
    if (err_ret == MPG123_NEW_FORMAT) {
      int64_t rate;
      int channels;
      int encoding;
      mpg123_getformat(mpg_h, &rate, &channels, &encoding);
      printf("[MPG123]MP3 decode rate:%ld; ch:%d; encoding:%d\n", rate, channels, encoding);
      err_ret = mpg123_decode(mpg_h, nullptr, 0, pcmBuffer, PCM_BUF_SIZE_MAX, &size);
    }
    pcm_length = static_cast<int>(size);

    if (err_ret == MPG123_OK || err_ret == MPG123_NEED_MORE) {
      printf("[MPG123]Success, MP3 decode size %ld\n", size);
    } else {
      printf("[MPG123]Error, MP3 decode failed %d\n", err_ret);
    }
  } else {
    printf("[MPG123]Error, MP3 decoder Init failed %d\n", err_ret);
  }
  mpg123_delete(mpg_h);
  mpg123_exit();
  printf("vc:tts: exit %s()\n", __func__);
  return err_ret;
}
/*decoder end*/

namespace cyberdog_audio
{
std::mutex mp3decoder::mutex;
std::shared_ptr<AudioPlayer> mp3decoder::mTtsPlayer = nullptr;
std::shared_ptr<mp3decoder> mp3decoder::decoder_ = nullptr;


mp3decoder::mp3decoder()
{
  player_end = false;
  mTtsPlayer =
    std::make_shared<AudioPlayer>(
    PCM_SDL_CH, std::bind(
      &mp3decoder::tts_player_callback,
      this), AUDIO_GROUP);
}

mp3decoder::~mp3decoder()
{
  printf("mp3decoder will destruct now!\n");
}


std::shared_ptr<mp3decoder> mp3decoder::GetInstance()
{
  if (!decoder_) {
    decoder_ = std::make_shared<mp3decoder>();
  }
  return decoder_;
}

int mp3decoder::pcm_playback(const unsigned char * src, const unsigned int len)
{
  unsigned char * audiobuffer = NULL;

  printf("vc:tts: enter %s(), len(bytes) = %d\n", __func__, len);
  if (audiodata_silence_mode == true) {
    printf("vc:tts: enter %s(), SSSSS  mode = %d\n", __func__, audiodata_silence_mode);
    return 0;
  }
  printf("vc:tts: enter %s(), SSSSS  mode = %d\n", __func__, audiodata_silence_mode);
  audiobuffer = (unsigned char *)malloc(len);  // alloc mem
  if (audiobuffer != NULL) {
    memcpy(audiobuffer, src, len);
  } else {
    printf("vc:tts: %s mem alloc error !!\n", __func__);
    return -1;
  }

  if (mTtsPlayer) {
    mTtsPlayer->AddPlay(audiobuffer, len);
  } else {
    printf("vc:tts: %s() mTtsPlayer is NULL !!\n", __func__);
  }

  free(audiobuffer);  // free mem
  printf("vc:tts: exit %s()\n", __func__);
  return 0;
}

int mp3decoder::tts_player_callback()
{
  printf("vc:tts: play end %s()\n", __func__);
  player_end = true;
  return msg_playback_complete_broadcaster(1);
}

int mp3decoder::tts_player_init(void)
{
  int result = -1;
  printf("vc:tts: enter %s()\n", __func__);
  // do nothing
  printf("vc:tts: exit %s(), result %d\n", __func__, result);
  return result;
}

int mp3decoder::tts_player_accumulate(const unsigned char * data, unsigned int length)
{
  int result = 0;
  printf("vc:tts: enter %s()\n", __func__);

  if (data == NULL || length == 0) {
    printf("vc:tts: tts_player_accumulate() invalid paramter \n");
    return -1;
  }

  memcpy((unsigned char *)(mp3Buffer + mp3_current_length), data, length);
  mp3_current_length += length;

  printf(
    "vc:tts: exit %s(), mp3_current_length[%d], result %d\n",
    __func__, mp3_current_length, result);
  return result;
}

int mp3decoder::tts_player_decode_paly(void)
{
  int result = -1;
  printf("vc:tts: enter %s()\n", __func__);

  mutex.lock();
  decoderIndex = 0;
  pcm_length = 0;

  /*decode*/
  if (mp3_current_length > PCM_BUF_SIZE_MAX) {
    mp3_current_length = PCM_BUF_SIZE_MAX;
  }
  result = tts_decode_mpg123(mp3Buffer, mp3_current_length);
  memset(mp3Buffer, 0x0, PCM_BUF_SIZE_MAX);
  mp3_current_length = 0;

  /*playback*/
  if (pcm_length > PCM_BUF_SIZE_MAX) {
    pcm_length = PCM_BUF_SIZE_MAX;
  }
  player_end = false;
  result = pcm_playback(pcmBuffer, pcm_length);
  memset(pcmBuffer, 0x0, PCM_BUF_SIZE_MAX);
  mutex.unlock();

  printf("vc:tts: exit %s(), result %d\n", __func__, result);
  return result;
}

int mp3decoder::tts_player_stop_now(void)
{
  printf("vc:tts: enter %s()\n", __func__);
  mTtsPlayer->StopPlay();
  printf("vc:tts: exit %s()\n", __func__);
  return 0;
}

bool mp3decoder::tts_player_is_playing(void)
{
  bool result = false;
  printf("vc:tts: enter %s()\n", __func__);
  result = mTtsPlayer->IsPlaying();
  printf("vc:tts: exit %s(), result %d\n", __func__, result);
  return result;
}

}  // namespace cyberdog_audio
/*END OF THIS FILE*/
