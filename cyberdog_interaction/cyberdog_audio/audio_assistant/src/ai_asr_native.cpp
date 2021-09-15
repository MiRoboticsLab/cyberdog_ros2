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

#include <string.h>

#include <string>
#include <vector>
#include <mutex>
#include <deque>
#include <memory>

#include "sys/unistd.h"
#include "xiaoai_sdk/paddy/paddy-engine.h"

#define DEBUG

#define PACKAGE_SIZE (320 * 1)
bool b_confidence = true;
extern int ai_nlp_native_setup_check(const char *);
#define    ASR_MODEL_BIN  "/opt/ros2/cyberdog/ai_conf/model.bin"
#define    ASR_DOMAIN_MODEL_DIR   "/opt/ros2/cyberdog/ai_conf"
#define    PROCESS_CNT   6

PaddyAsr::PaddyEngine decoder;
void * decoder_handle;
FILE * model_fp;
char * model_data;
char package_buf[PACKAGE_SIZE];

typedef enum
{
  MSM_ASR_IDLE,
  MSM_ASR_START,
  MSM_ASR_PROCESS,
  MSM_ASR_STOP,
  MSM_ASR_MAX
} asr_msg_type_t;

typedef struct
{
  unsigned char * start;
  unsigned int length;
  std::mutex mutex;
} paddy_input_buf_t;

#define RETRY_TIMES 5
static unsigned char buffer_1920[1920 + 64];
std::deque<asr_msg_type_t> asr_queue;
static paddy_input_buf_t paddy_input_buffer;
static bool native_asr_enabled = false;

static bool mNativeAsrRunStatus = true;
bool getNativeAsrRunStatus(void)
{
  return mNativeAsrRunStatus;
}

int setNativeAsrRunStatus(bool on)
{
  mNativeAsrRunStatus = on;
  return 0;
}

int ai_native_asr_init()
{
  PaddyAsr::ReturnValue res = PaddyAsr::SUCCESS;
  decoder.SetLogLevel(0);

  // init paddy input buffer
  memset(buffer_1920, 0x0, 1920);
  paddy_input_buffer.start = buffer_1920;
  paddy_input_buffer.length = 1920;

  model_fp = fopen(ASR_MODEL_BIN, "rb");
  if (model_fp == NULL) {
    printf("vc: Can not open %s.\n", ASR_MODEL_BIN);
    return -1;
  }

  fseek(model_fp, 0, SEEK_END);
  int model_size = ftell(model_fp);

  fseek(model_fp, 0, SEEK_SET);
  model_data = reinterpret_cast<char *>(malloc(sizeof(char) * (model_size + 1)));
  fread(model_data, sizeof(char), model_size, model_fp);

  PaddyAsr::ReturnValue ret = decoder.Init(ASR_MODEL_BIN, model_data, model_size, 1, 1);
  if (!ret) {
    printf("vc: Init Success!\n");
  } else {
    printf("vc: Init Failed!\n");
    return -1;
  }

  std::string name = "k91";
  std::string domain_model_dir = ASR_DOMAIN_MODEL_DIR;
  float bonus = -0.2;
  /*no need handler*/
  res = decoder.EnableDomainGrammars(NULL, name.c_str(), bonus, domain_model_dir.c_str());
  if (res != PaddyAsr::SUCCESS) {
    printf("vc: Enable %s grammar faild, err code is %d\n", name.c_str(), res);
  } else {
    printf("vc: Enable  Success!\n");
  }

  return 0;
}

int ai_asr_create()
{
  printf("[AUDIO][ASR_N] vc: enter ai_asr_create \n");
  if (!native_asr_enabled) {
    printf("[AUDIO][ASR_N] vc: native_asr not enabled \n");
    return -1;
  }

  decoder_handle = decoder.CreateTask();
  return 0;
}

int ai_asr_process()
{
  int i = 0;

  if (!native_asr_enabled) {
    printf("[AUDIO][ASR_N] vc: native_asr not enabled \n");
    return -1;
  }

  for (i = 0; i < PROCESS_CNT; i++) {
    memcpy(package_buf, (unsigned char *)&buffer_1920[0] + i * PACKAGE_SIZE, PACKAGE_SIZE);
    PaddyAsr::DecoderResult result = decoder.ProcessTask(decoder_handle, package_buf, PACKAGE_SIZE);
    if (result.retval != PaddyAsr::SUCCESS) {
      printf("vc: ai_asr_process error !\n");
      break;
    }
  }
  return 0;
}

int ai_asr_result()
{
  printf("[AUDIO][ASR_N] Enter %s, handle is %p \n", __func__, decoder_handle);

  if (!native_asr_enabled) {
    printf("[AUDIO][ASR_N] vc: loop native_asr not enabled \n");
    return -1;
  }

  PaddyAsr::DecoderResult result = decoder.StopTask(decoder_handle);

  if (b_confidence) {
    float confidence = 0.f;
    if (result.grammar_nbest.size() > 0 && result.text == result.grammar_nbest[0].text) {
      printf("vc: Domain True!\n");
      confidence = result.grammar_nbest[0].confidence;
    } else {
      printf("vc: Domain False!\n");
      confidence = result.nbest[0].confidence;
    }
    printf("vc: result is %s, confidence is %f\n", result.text.c_str(), confidence);
    /*pass to nlp to check*/
    ai_nlp_native_setup_check(result.text.c_str());
  } else {
    printf("vc: result is %s\n", result.text.c_str());
  }

  std::vector<PaddyAsr::Result> & res_vec = result.nbest;
  while (!res_vec.empty()) {
    res_vec.pop_back();
  }
  std::vector<PaddyAsr::Result>(res_vec).swap(res_vec);
  res_vec.clear();

  return 0;
}

int ai_asr_destroy()
{
  if (model_data) {free(model_data);}
  decoder.Destroy();
  fclose(model_fp);
  return 0;
}

void ai_nativeasr_data_handler(asr_msg_type_t msg, unsigned char * buf, int buf_size)
{
  if (!native_asr_enabled) {
    printf("[AUDIO][ASR_PROX] Native asr engine is not ready !!!\n");
    return;
  }
  switch (msg) {
    case MSM_ASR_START:
      asr_queue.push_front(MSM_ASR_START);
      printf("[AUDIO][ASR_PROX] Push MSM_ASR_START to native asr engine.\n");
      break;

    case MSM_ASR_STOP:
      asr_queue.push_front(MSM_ASR_STOP);
      printf("[AUDIO][ASR_PROX] Push MSM_ASR_STOP to native asr engine.\n");
      break;

    case MSM_ASR_PROCESS:
      if (paddy_input_buffer.mutex.try_lock()) {
        memcpy(paddy_input_buffer.start, buf, buf_size);
        // printf("[AUDIO][ASR_PROX] Copy data to paddy_input_buffer.\n");
        paddy_input_buffer.mutex.unlock();

        asr_queue.push_front(MSM_ASR_PROCESS);
        printf("[AUDIO][ASR_PROX] Push MSM_ASR_PROCESS to native asr engine.\n");
      } else {
        printf("[AUDIO][ASR_PROX] Do not get lock, skip this frame !!!\n");
      }
      break;

    default:
      printf("[AUDIO][ASR_PROX] Can not handle this msg[%d] !!!\n", msg);
      break;
  }
}

void ai_native_asr_work_loop()
{
  asr_msg_type_t msg;
  for (;; ) {
    if (getNativeAsrRunStatus() == false) {
      printf("[AUDIO][ASR_N] Native asr engine switch off\n");
      usleep(1000 * 1000 * 1);
      continue;
    }

    if (!native_asr_enabled) {
      printf("[AUDIO][ASR_N] Native asr engine is not ready !!!\n");
      usleep(1000 * 1000 * 1);
      continue;
    }

    if (!asr_queue.empty()) {
      msg = asr_queue.back();
      asr_queue.pop_back();
    } else {
      usleep(1000 * 40);
      // printf("[AUDIO][ASR_N] Message queue is empty\n");
      continue;
    }

    if (msg == MSM_ASR_START) {
      printf("[AUDIO][ASR_N] Handle MSM_ASR_START\n");
      ai_asr_create();
    } else if (msg == MSM_ASR_STOP) {
      printf("[AUDIO][ASR_N] Handle MSM_ASR_STOP\n");
      ai_asr_result();
    } else if (msg == MSM_ASR_PROCESS) {
      ai_asr_process();
    } else {
      printf("[AUDIO][ASR_N] Can not handle this msg[%d] !!!\n", msg);
    }
  }
}

void ai_native_asr_task(void)
{
  int result = -1;

  for (int i = 0; i < RETRY_TIMES; i++) {
    result = ai_native_asr_init();
    if (result == 0) {
      printf("vc: ai_native_asr_init() init Ok\n");
      native_asr_enabled = true;
      break;
    } else {
      sleep(5);
      printf("vc: ai_native_asr_init() init failed, retry it\n");
    }
  }

  if (result == 0) {
    ai_native_asr_work_loop();
  } else {
    printf("vc: ai_native_asr_init() init failed, exit");
  }
}
