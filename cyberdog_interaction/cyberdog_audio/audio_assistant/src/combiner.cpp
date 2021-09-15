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
#include <mutex>
#include <condition_variable>

#include "sys/timeb.h"
#include "sys/time.h"
#include "sys/signal.h"
#include "audio_assistant/combiner.hpp"
#include "audio_base/capture.hpp"
#include "audio_base/debug/ai_debugger.hpp"

static bool mCaptureRunStatus = true;
bool player_end;

// vpm
#define AI_RECORD_DEBUG
void ai_rawdata_handle(char * buf, int buf_size);
bool getMainCaptureRunStatus(void);

// for debug
static int raw_dump_fd = -1;
static bool ai_need_main_dump = false;

typedef struct
{
  unsigned char * start;
  unsigned int length;
  std::mutex mutex;
} vpm_input_buf_t;

vpm_input_buf_t vpm_input_buffer;
bool vpm_buf_ready = false;
bool record_on_status = true;

static unsigned char buffer[2 * 7 * 320 + 14 * 2];

std::condition_variable cond_sound;
std::condition_variable cond_ref;
std::mutex mutex_sound;
std::mutex mutex_ref;
bool sound_ready = false;
bool ref_ready = false;
bool gNeedRunning = false;
extern bool audiodata_silence_mode;

typedef struct
{
  snd_pcm_t * handle;
  char * buf_1;
  char * buf_2;
  snd_pcm_uframes_t frames;
  int size;
} pcm_params_t;

void * pcm_read_sound(pcm_params_t params)
{
  pcm_params_t pcm_params = params;
  snd_pcm_t * hdl = pcm_params.handle;
  char * buffer_1 = pcm_params.buf_1;
  char * buffer_2 = pcm_params.buf_2;
  int frames = pcm_params.frames;
  int size = pcm_params.size;
  printf("vc:combiner: %s enter\n", __func__);
  while (gNeedRunning) {
    // if (getMainCaptureRunStatus() == false) {
    //     usleep(1000 * 50);
    //     continue;
    // }
    snd_pcm_readi(hdl, buffer_1, frames);
    memcpy(buffer_2, buffer_1, size);
    sound_ready = true;
    cond_sound.notify_one();
  }
  printf("vc:combiner: %s exit\n", __func__);
  return nullptr;
}

void * pcm_read_ref(pcm_params_t params)
{
  pcm_params_t pcm_params = params;
  snd_pcm_t * hdl = pcm_params.handle;
  char * buffer_1 = pcm_params.buf_1;
  char * buffer_2 = pcm_params.buf_2;
  int frames = pcm_params.frames;
  int size = pcm_params.size;
  printf("vc:combiner: %s enter\n", __func__);
  while (gNeedRunning) {
    // if (getMainCaptureRunStatus() == false) {
    //     usleep(1000 * 50);
    //     continue;
    // }
    snd_pcm_readi(hdl, buffer_1, frames);
    memcpy(buffer_2, buffer_1, size);

    ref_ready = true;
    cond_ref.notify_one();
  }
  printf("vc:combiner: %s exit\n", __func__);
  return nullptr;
}

void ai_rawdata_handle(unsigned char * buf, int buf_size)
{
  if (vpm_input_buffer.mutex.try_lock()) {
    memcpy(vpm_input_buffer.start, buf, buf_size);
    vpm_input_buffer.mutex.unlock();
  }
}

void setup_vpm_input_buffer(void)
{
  memset(buffer, 0x0, 2 * 7 * 320);
  vpm_input_buffer.start = buffer;
  vpm_input_buffer.length = 2 * 7 * 320;

  vpm_buf_ready = true;
}

bool getMainCaptureRunStatus(void)
{
  return mCaptureRunStatus;
}

int setMainCaptureRunStatus(bool on)
{
  mCaptureRunStatus = on;
  return 0;
}

void recorder_work_loop()
{
  player_end = true;
  unsigned char * buffer_mix;
  unsigned char * buffer_output;
  int output_size;
  int output_pos = 0;
  int output_left;
  snd_pcm_t * handle_ref;
  snd_pcm_t * handle_sound;
  char dev_ref[] = "hw:1,0";
  char dev_snd[] = "hw:1,1";
  int dir_ref, dir_sound;
  unsigned int val_ref, val_sound;
  snd_pcm_uframes_t frames_ref = 320;
  snd_pcm_uframes_t frames_sound = 320;
  snd_pcm_hw_params_t * params_ref;
  snd_pcm_hw_params_t * params_sound;
  int chk_index = 0;
  int rc;
  auto audioDebugger = std::make_shared<std::audioDebugger>();

  printf("vc:combiner: enter %s()\n", __func__);
  ai_need_main_dump = audioDebugger->getConfig(AUDIO_DEBUG_CONFIG_DUMPRAW);
  printf("vc:combiner: audioconfig need_main_dump[%d]\n", ai_need_main_dump);

  if (ai_need_main_dump) {
    audioDebugger->audioDumpFileOpen(AUDIO_DEBUG_CONFIG_DUMPRAW, &raw_dump_fd);
  }

  // init vpm input buffer
  printf("vc:combiner: vpm buffer setup \n");
  setup_vpm_input_buffer();

  printf("vc:combiner: open refernce data device \n");

  while ((rc = snd_pcm_open(&handle_ref, dev_ref, SND_PCM_STREAM_CAPTURE, 0)) < 0) {
    printf("vc:combiner: unable to open pcm device: %s/n", snd_strerror(rc));
    sleep(1);
  }

  printf("vc:combiner: open pcm device: %s successful/n", snd_strerror(rc));
  snd_pcm_hw_params_alloca(&params_ref);
  snd_pcm_hw_params_any(handle_ref, params_ref);
  snd_pcm_hw_params_set_access(
    handle_ref, params_ref,
    SND_PCM_ACCESS_RW_INTERLEAVED);
  snd_pcm_hw_params_set_format(
    handle_ref, params_ref,
    SND_PCM_FORMAT_S16_LE);
  snd_pcm_hw_params_set_channels(handle_ref, params_ref, 1);
  val_ref = RATE;
  snd_pcm_hw_params_set_rate_near(handle_ref, params_ref, &val_ref, &dir_ref);
  snd_pcm_hw_params_set_period_size_near(handle_ref, params_ref, &frames_ref, &dir_ref);
  rc = snd_pcm_hw_params(handle_ref, params_ref);
  snd_pcm_hw_params_get_period_time(params_ref, &val_ref, &dir_ref);
  snd_pcm_hw_params_get_period_size(params_ref, &frames_ref, &dir_ref);
  if (rc < 0) {
    printf("vc:combiner: unable to set hw parameters: %s/n", snd_strerror(rc));
    exit(1);
  }

  printf("vc:combiner: open mic data device \n");
  rc = snd_pcm_open(
    &handle_sound, dev_snd,
    SND_PCM_STREAM_CAPTURE, 0);
  if (rc < 0) {
    printf("vc:combiner: unable to open pcm device: %s/n", snd_strerror(rc));
    exit(1);
  }
  snd_pcm_hw_params_alloca(&params_sound);
  snd_pcm_hw_params_any(handle_sound, params_sound);
  snd_pcm_hw_params_set_access(
    handle_sound, params_sound,
    SND_PCM_ACCESS_RW_INTERLEAVED);
  snd_pcm_hw_params_set_format(
    handle_sound, params_sound,
    SND_PCM_FORMAT_S16_LE);
  snd_pcm_hw_params_set_channels(handle_sound, params_sound, 6);
  val_sound = RATE;
  snd_pcm_hw_params_set_rate_near(handle_sound, params_sound, &val_sound, &dir_sound);
  snd_pcm_hw_params_set_period_size_near(handle_sound, params_sound, &frames_sound, &dir_sound);
  rc = snd_pcm_hw_params(handle_sound, params_sound);
  snd_pcm_hw_params_get_period_time(params_sound, &val_sound, &dir_sound);
  snd_pcm_hw_params_get_period_size(params_sound, &frames_sound, &dir_sound);
  if (rc < 0) {
    printf(
      "vc:combiner: unable to set hw parameters: %s/n",
      snd_strerror(rc));
    exit(1);
  }

  printf("vc:combiner: mix buffer setup \n");
  int buffer_size_sound = 2 * 6 * frames_sound;
  int buffer_size_ref = 2 * 1 * frames_sound;
  int buffer_size_mix = 2 * 7 * frames_sound;
  output_size = buffer_size_mix * 5;
  output_left = output_size;

  buffer_mix = (unsigned char *)malloc(buffer_size_mix);
  buffer_output = (unsigned char *)malloc(output_size);
  char * buffer_sound_1 = reinterpret_cast<char *>(malloc(buffer_size_sound));
  char * buffer_sound_2 = reinterpret_cast<char *>(malloc(buffer_size_sound));
  char * buffer_ref_1 = reinterpret_cast<char *>(malloc(buffer_size_ref));
  char * buffer_ref_2 = reinterpret_cast<char *>(malloc(buffer_size_ref));

  pcm_params_t pcm_params_sound =
  {handle_sound, buffer_sound_1, buffer_sound_2, frames_sound, buffer_size_sound};
  pcm_params_t pcm_params_ref =
  {handle_ref, buffer_ref_1, buffer_ref_2, frames_ref, buffer_size_ref};

  printf("vc:combiner: sound/ref thread create \n");
  gNeedRunning = true;
  std::thread record_ref(pcm_read_ref, pcm_params_ref);
  usleep(5000);
  std::thread record_sound(pcm_read_sound, pcm_params_sound);

  printf("vc:combiner: before mix sound/ref data \n");
  for (;; ) {
    std::unique_lock<std::mutex> lock_sound(mutex_sound);

    if (audiodata_silence_mode || !player_end) {
      usleep(5000 * 100);
      continue;
    }

    cond_sound.wait(lock_sound, [] {return sound_ready;});
    sound_ready = false;

    std::unique_lock<std::mutex> lock_ref(mutex_ref);
    cond_ref.wait(lock_ref, [] {return ref_ready;});
    ref_ready = false;

    for (uint32_t i = 0; i < frames_sound; i++) {
      memcpy(buffer_mix + 14 * i, buffer_sound_2 + 12 * i, 12);
      memcpy(buffer_mix + 14 * i + 12, buffer_ref_2 + 2 * i, 2);
    }

    if (chk_index % 100 == 0) {
      printf("vc:combiner: recorder monitor ping...\n\n");
    }
    chk_index++;
    if (chk_index >= 1000) {chk_index = 0;}

    if (ai_need_main_dump) {
      audioDebugger->audioDumpFileWrite(
        &raw_dump_fd,
        buffer_mix,
        buffer_size_mix);
    }

    /*pass the data to ai keyword processing*/
    ai_rawdata_handle(buffer_mix, buffer_size_mix);

    if (output_left > buffer_size_mix) {
      memcpy(buffer_output + output_pos, buffer_mix, buffer_size_mix);
      output_pos += buffer_size_mix;
    } else if (output_left == buffer_size_mix) {
      memcpy(buffer_output + output_pos, buffer_mix, buffer_size_mix);
      output_pos = 0;
    } else {
      memcpy(buffer_output + output_pos, buffer_mix, output_left);
      memcpy(buffer_output, buffer_mix + output_left, buffer_size_mix - output_left);
      output_pos = buffer_size_mix - output_left;
    }
    output_left = output_size - output_pos;
  }

  gNeedRunning = false;

  snd_pcm_drain(handle_ref);
  snd_pcm_drain(handle_sound);
  snd_pcm_close(handle_ref);
  snd_pcm_close(handle_sound);
  free(buffer_mix);
  free(buffer_output);
  free(buffer_sound_1);
  free(buffer_sound_2);
  free(buffer_ref_1);
  free(buffer_ref_2);
}
