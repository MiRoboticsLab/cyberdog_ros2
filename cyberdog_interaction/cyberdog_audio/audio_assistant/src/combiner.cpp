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
#include <queue>
#include <mutex>
#include <condition_variable>

#include "sys/timeb.h"
#include "sys/time.h"
#include "sys/signal.h"
#include "audio_assistant/combiner.hpp"
#include "audio_base/capture.hpp"
#include "audio_base/debug/ai_debugger.hpp"
#include "audio_base/mempool.hpp"

// vpm
#define AI_RECORD_DEBUG
static bool mCaptureRunStatus = true;
bool vpm_buf_ready = false;
bool record_on_status = true;
bool player_end;
extern bool audiodata_silence_mode;
bool sound_ready = false;
bool ref_ready = false;
bool gNeedRunning = false;

// for debug
static int raw_dump_fd = -1;
static bool ai_need_main_dump = false;

typedef struct
{
  unsigned char * start;
  unsigned int length;
  void free_mem() {}
} vpm_input_buf_t;

vpm_input_buf_t vpm_input_buffer;

athena_audio::MsgQueue<vpm_input_buf_t> vpm_msg_queue;
std::shared_ptr<audio_base::MemPool> vpm_mp =
  std::make_shared<audio_base::MemPool>(128, 2 * 7 * 320);

bool getMainCaptureRunStatus(void)
{
  return mCaptureRunStatus;
}

int setMainCaptureRunStatus(bool on)
{
  mCaptureRunStatus = on;
  return 0;
}

void ai_rawdata_handle(unsigned char * buf, int buf_size)
{
  if (audiodata_silence_mode) {
    if (!vpm_msg_queue.Empty()) {
      vpm_msg_queue.Clear();
    }
  } else {
    vpm_input_buf_t vpm_data;
    vpm_data.length = buf_size;
    vpm_mp->GetMemory(&vpm_data.start);
    memcpy(vpm_data.start, buf, vpm_data.length);
    vpm_msg_queue.EnQueueOne(vpm_data);
  }
}

bool setup_vpm_input_buffer(void)
{
  if (!vpm_mp->Create(true)) {
    fprintf(stderr, "vc: create vpm buffer failed!\n");
    return false;
  }

  vpm_buf_ready = true;
  return true;
}

void recorder_work_loop()
{
  player_end = true;
  unsigned char * buffer_mix;
  snd_pcm_t * handle_sound;
  char dev_snd[] = "hw:1,1";
  int dir_sound;
  unsigned int val_sound;
  snd_pcm_uframes_t frames_sound = 320;
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
  auto recorder_ = std::make_shared<cyberdog_audio::AudioPlayer>(3);

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

  std::shared_ptr<audio_base::MemPool> mp = std::make_shared<audio_base::MemPool>(
    DELAY_TIMES * 10,
    buffer_size_sound);
  if (!mp->Create()) {
    fprintf(stderr, "vc: combiner cannot allocate memory pool!\n");
    return;
  }

  char * buffer_sound;
  bool reference_start = false;
  bool combiner_start = false;
  char * buffer_ref = reinterpret_cast<char *>(malloc(buffer_size_ref));
  buffer_mix = (unsigned char *)malloc(buffer_size_mix);
  std::queue<char *> sound_queue = std::queue<char *>();

  fprintf(stdout, "vc:combiner: before mix sound/ref data \n");
  for (;; ) {
    // if (audiodata_silence_mode) {
    //   usleep(5000 * 100);
    //   continue;
    // }
    if (reference_start == false) {
      reference_start = true;
      recorder_->OpenReference(buffer_size_ref);
    }
    while (true) {
      if (recorder_->HaveReferenceData() == false) {
        usleep(1000);
        continue;
      }
      recorder_->GetReferenceData(buffer_ref);
      break;
    }

    // sound_queue.push(reinterpret_cast<char *>(malloc(buffer_size_sound)));
    char * src;
    int mp_flag = mp->GetMemory(&src);
    sound_queue.push(src);
    while (true) {
      rc = snd_pcm_readi(handle_sound, sound_queue.back(), frames_sound);
      if (rc == -EAGAIN) {
        usleep(1000);
        continue;
      } else if (rc == -EPIPE) {
        std::cout << "vc:combiner:[sound]Overrun occured\n";
        snd_pcm_prepare(handle_sound);
        continue;
      } else if (rc < 0) {
        std::cout << "vc:combiner:[sound]Read Error: " << snd_strerror(rc) << "\n";
      }
      break;
    }
    if (combiner_start == false) {
      if (static_cast<int>(sound_queue.size()) == DELAY_TIMES) {
        combiner_start = true;
      }
      continue;
    } else {
      buffer_sound = sound_queue.front();
      sound_queue.pop();
    }

    for (uint32_t i = 0; i < frames_sound; i++) {
      memcpy(buffer_mix + 14 * i, buffer_sound + 12 * i, 12);
      memcpy(buffer_mix + 14 * i + 12, buffer_ref + 2 * i, 2);
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
    mp->Release(mp_flag);
  }
  recorder_->CloseReference();
  // while (sound_queue.empty() == false) {
  //   free(sound_queue.front());
  //   sound_queue.pop();
  // }
  mp->Clear();
  snd_pcm_drain(handle_sound);
  snd_pcm_close(handle_sound);
  free(buffer_mix);
  // free(buffer_output);
  free(buffer_ref);
}
