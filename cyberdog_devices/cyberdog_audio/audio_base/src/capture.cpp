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

#include "audio_base/capture.hpp"

void capture_test()
{
  WAVE_HEAD wavehead;
  // parameters for wave file head
  memcpy(wavehead.riff_head, "RIFF", 4);
  wavehead.riffdata_len = LENGTH * RATE * CHANNELS * SIZE / 8 + 36;
  memcpy(wavehead.wave_head, "WAVE", 4);
  memcpy(wavehead.fmt_head, "fmt ", 4);
  wavehead.fmtdata_len = 16;
  wavehead.format_tag = 1;
  wavehead.channels = CHANNELS;
  wavehead.samples_persec = RATE;
  wavehead.bytes_persec = RATE * CHANNELS * SIZE / 8;
  wavehead.block_align = CHANNELS * SIZE / 8;
  wavehead.bits_persec = SIZE;
  memcpy(wavehead.data_head, "data", 4);
  wavehead.data_len = LENGTH * RATE * CHANNELS * SIZE / 8;

  int rc, i = 0;
  int size;
  snd_pcm_t * handle;
  snd_pcm_hw_params_t * params;
  unsigned int val, val2;
  int dir;
  snd_pcm_uframes_t frames;
  char * buffer;
  // clear wav file already exist
  system("rm -f sound.wav");

  int fd_f;
  if (( fd_f = open("./sound.wav", O_CREAT | O_RDWR, 0777)) == -1) {
    perror("cannot creat the sound file");
  }
  // write to wave file head
  if (write(fd_f, &wavehead, sizeof(wavehead)) == -1) {
    perror("write to sound'head wrong!!");
  }

  /* Open PCM device for recording (capture). */
  rc = snd_pcm_open(
    &handle, "default",
    SND_PCM_STREAM_CAPTURE, 0);
  if (rc < 0) {
    fprintf(stderr, "unable to open pcm device: %s/n", snd_strerror(rc));
    exit(1);
  }
  /* Allocate a hardware parameters object. */
  snd_pcm_hw_params_alloca(&params);
  /* Fill it in with default values. */
  snd_pcm_hw_params_any(handle, params);
  /* Set the desired hardware parameters. */
  /* Interleaved mode */
  snd_pcm_hw_params_set_access(
    handle, params,
    SND_PCM_ACCESS_RW_INTERLEAVED);
  /* Signed 16-bit little-endian format */
  snd_pcm_hw_params_set_format(
    handle, params,
    SND_PCM_FORMAT_S16_LE);
  /* Two channels (stereo) */
  snd_pcm_hw_params_set_channels(handle, params, 1);
  /* 44100 bits/second sampling rate (CD quality) */
  val = RATE;
  snd_pcm_hw_params_set_rate_near(handle, params, &val, &dir);
  /* Set period size to 32 frames. */
  frames = RSIZE / (SIZE / 8);
  snd_pcm_hw_params_set_period_size_near(handle, params, &frames, &dir);
  /* Write the parameters to the driver */
  rc = snd_pcm_hw_params(handle, params);
  if (rc < 0) {
    fprintf(
      stderr, "unable to set hw parameters: %s/n",
      snd_strerror(rc));
    exit(1);
  }
  /* Use a buffer large enough to hold one period */
  snd_pcm_hw_params_get_period_size(params, &frames, &dir);
  size = frames * SIZE / 8;  /* 2 bytes/sample, 2 channels */
  printf("size = %d\n", size);
  buffer = reinterpret_cast<char *>(malloc(size));
  /* We want to loop for 5 seconds */
  snd_pcm_hw_params_get_period_time(params, &val, &dir);
  for (i = 0; i < (LENGTH * RATE * SIZE * CHANNELS / 8) / RSIZE; i++) {
    rc = snd_pcm_readi(handle, buffer, frames);
    if (rc == -EPIPE) {
      /* EPIPE means overrun */
      fprintf(stderr, "overrun occurred/n");
      snd_pcm_prepare(handle);
    } else if (rc < 0) {
      fprintf(
        stderr,
        "error from read: %s/n",
        snd_strerror(rc));
    } else if (rc != static_cast<int>(frames)) {
      fprintf(stderr, "short read, read %d frames/n", rc);
    }

    if (write(fd_f, buffer, size) == -1) {
      perror("write to sound wrong!!");
    }
    // else printf("fwrite buffer success\n");
  }
  /******************print parameters*********************/
  snd_pcm_hw_params_get_channels(params, &val);
  printf("channels = %d\n", val);
  snd_pcm_hw_params_get_rate(params, &val, &dir);
  printf("rate = %d bps\n", val);
  snd_pcm_hw_params_get_period_time(
    params,
    &val, &dir);
  printf("period time = %d us\n", val);
  snd_pcm_hw_params_get_period_size(
    params,
    &frames, &dir);
  printf("period size = %d frames\n", static_cast<int>(frames));
  snd_pcm_hw_params_get_buffer_time(
    params,
    &val, &dir);
  printf("buffer time = %d us\n", val);
  snd_pcm_hw_params_get_buffer_size(
    params,
    reinterpret_cast<snd_pcm_uframes_t *>(&val));
  printf("buffer size = %d frames\n", val);
  snd_pcm_hw_params_get_periods(params, &val, &dir);
  printf("periods per buffer = %d frames\n", val);
  snd_pcm_hw_params_get_rate_numden(
    params,
    &val, &val2);
  printf("exact rate = %d/%d bps\n", val, val2);
  val = snd_pcm_hw_params_get_sbits(params);
  printf("significant bits = %d\n", val);
  // snd_pcm_hw_params_get_tick_time(params,  &val, &dir);
  printf("tick time = %d us\n", val);
  val = snd_pcm_hw_params_is_batch(params);
  printf("is batch = %d\n", val);
  val = snd_pcm_hw_params_is_block_transfer(params);
  printf("is block transfer = %d\n", val);
  val = snd_pcm_hw_params_is_double(params);
  printf("is double = %d\n", val);
  val = snd_pcm_hw_params_is_half_duplex(params);
  printf("is half duplex = %d\n", val);
  val = snd_pcm_hw_params_is_joint_duplex(params);
  printf("is joint duplex = %d\n", val);
  val = snd_pcm_hw_params_can_overrange(params);
  printf("can overrange = %d\n", val);
  val = snd_pcm_hw_params_can_mmap_sample_resolution(params);
  printf("can mmap = %d\n", val);
  val = snd_pcm_hw_params_can_pause(params);
  printf("can pause = %d\n", val);
  val = snd_pcm_hw_params_can_resume(params);
  printf("can resume = %d\n", val);
  val = snd_pcm_hw_params_can_sync_start(params);
  printf("can sync start = %d\n", val);
  /*******************************************************************/
  snd_pcm_drain(handle);
  snd_pcm_close(handle);
  close(fd_f);
  free(buffer);
}
