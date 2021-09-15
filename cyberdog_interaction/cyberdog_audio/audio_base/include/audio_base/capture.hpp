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

#ifndef AUDIO_BASE__CAPTURE_HPP_
#define AUDIO_BASE__CAPTURE_HPP_

#include <stdio.h>
#ifdef __cplusplus
extern "C"
{
#endif  // __cplusplus
#include <alsa/asoundlib.h>
#ifdef __cplusplus
}
#endif  // __cplusplus

// time
#define LENGTH       10
#define RATE         16000
#define SIZE         16
#define CHANNELS     6
// buf size
#define RSIZE        64

typedef struct WAVEHEAD
{
  /****RIFF WAVE CHUNK*/
  unsigned char riff_head[4];
  int riffdata_len;
  unsigned char wave_head[4];
  /****RIFF WAVE CHUNK*/
  /****Format CHUNK*/
  unsigned char fmt_head[4];
  int fmtdata_len;
  int16_t format_tag;
  int16_t channels;
  int samples_persec;
  int bytes_persec;
  int16_t block_align;
  int16_t bits_persec;
  /****Format CHUNK*/
  /***Data Chunk**/
  unsigned char data_head[4];
  int data_len;
} WAVE_HEAD;

void capture_test(char * filename);
void capture_test_ref(char * filename);
#endif  // AUDIO_BASE__CAPTURE_HPP_
