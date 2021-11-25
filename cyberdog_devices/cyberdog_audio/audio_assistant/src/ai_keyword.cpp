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
#include <deque>
#include <thread>

#include "rclcpp/rclcpp.hpp"
#include "xiaoai_sdk/vpm/vpm_sdk.h"
#include "audio_base/debug/ai_debugger.hpp"
#include "audio_base/audio_queue.hpp"
#include "audio_assistant/mp3decoder.hpp"

#define VPM_CONFIG_FILE_PATH "/opt/ros2/cyberdog/ai_conf/xaudio_engine.conf"

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
} vpm_input_buf_t;

extern athena_audio::MsgQueue<vpm_input_buf_t> vpm_msg_queue;
extern bool vpm_buf_ready;
extern void ai_nativeasr_data_handler(asr_msg_type_t msg, unsigned char * buf, int buf_size);
void ai_vpm_work_loop();
// Keyword data from wakeup sdk
extern int aivs_recognize_start(void);
extern int aivs_recognize_postData(uint8_t * data, uint32_t length);
extern int aivs_recognize_stop(void);
extern bool aivsFinishEventNow;
extern bool audiodata_silence_mode;

std::shared_ptr<std::audioDebugger> audioDebuggerPtr;
/*for playback complete sync*/
static std::mutex mMutexPlayComplete;
static std::condition_variable mCondVarPlayComplete;
static std::deque<int> mMsgQueuePlayComplete;

int tts_stream_flush(void);
int aivs_set_interrupt(void);
int tts_player_flush_and_close(void);

bool get_aivs_engine_status(void);
int ai_push_msg(int msg);
int set_vpm_status_unwakeup();
int set_vpm_status_wakeup();

static bool mAiWakeupdRunStatus = true;

// for debug
static int asrDumpFileFd = -1;
static bool isNeedDump = false;
static bool isNeedSilence = false;

/******************************/
bool getAiWakeupRunStatus(void)
{
  return mAiWakeupdRunStatus;
}

int setAiWakeupRunStatus(bool on)
{
  mAiWakeupdRunStatus = on;
  return 0;
}

static void vpm_log_cb(vpm_log_lvl_t, const char *, const char * format, ...)
{
  char buffer[2048] = {0};
  va_list args;
  va_start(args, format);
  vsnprintf((char * const)buffer, sizeof(buffer), format, args);
  va_end(args);
  printf("(vpm_log_cb) vc:vpm: %s", buffer);
}

static void BroadcastBarkMsg(void)
{
  printf("vc:vpm: %s:enter\n", __func__);
  if (isNeedSilence) {
    return;
  } else {
    ai_push_msg(100);
  }
}

// Call back to handle event from wakeup sdk
// VPM_EVENT_WAKE_NORMAL for wakeup
// VPM_EVENT_WAIT_ASR_TIMEOUT VAD not asr so timeout
// VPM_EVENT_VAD_BEGIN VAD: checked voice begin (NOT USE NOW)
// VPM_EVENT_VAD_END： VAD Checked voice end, asr finished
static void vpm_event_handle(const void *, vpm_event_t event, int)
{
  switch (event) {
    case VPM_EVENT_WAKE_NORMAL:
      /* code */
      printf("vc:vpm: VPM_EVENT_WAKE_NORMAL \n");
      break;
    case VPM_EVENT_WAIT_ASR_TIMEOUT:
      /* code */
      printf("vc:vpm: VPM_EVENT_WAIT_ASR_TIMEOUT \n");
      break;
    case VPM_EVENT_VAD_BEGIN:
      /* code */
      printf("vc:vpm: VPM_EVENT_VAD_BEGIN \n");
      break;
    case VPM_EVENT_VAD_END:
      /* code */
      printf("vc:vpm: vpm_event_handle VPM_EVENT_VAD_END \n");
      break;
    default:
      break;
  }
}

//  typedef enum {
//      TTS_DEV_STATUS_IDLE = 0,
//      TTS_DEV_STATUS_OPEN = 1,
//      TTS_DEV_STATUS_PLAYING = 2,
//      TTS_DEV_STATUS_CLOSE = 3,
//  } tts_dev_status_t;
// tts_dev_status_t  get_tts_device_status(void);
// extern bool ttsStremInterrupt;

static void vpm_wakeup_data_handle(
  const void *, const vpm_callback_data_t * data, int)
{
  float angle = 0;

  printf("vc:vpm: %s [=====>>>>>] check keyword successfully\n", __func__);

  if (audiodata_silence_mode) {
    printf("vc:vpm: %s [=====>>>>>] silence mode return\n", __func__);
    return;
  }

  // close interrupt while audio playing
  // std::shared_ptr<mp3decoder> mp3decoder_ptr = mp3decoder::GetInstance();
  // if ( mp3decoder_ptr->tts_player_is_playing() == true ) {
  // aivs_set_interrupt();
  // tts_stream_flush();
  // mp3decoder_ptr->tts_player_stop_now();
  // printf("vc:vpm: %s [=====>>>>>] tts_player_stop_now\n", __func__);
  // } else  {
  // printf("vc:vpm: %s [=====>>>>>] no sound is playing\n", __func__);
  // }

  BroadcastBarkMsg();

  angle = data->angle;
  printf("vc:vpm: %s: >>> the angle is = %f\n", __func__, angle);
}

/*ASR data
*After wakeup, the data will be transtered to asr module
*VPM_CALLBACK_DATA_HEADER:no need pass to asr
*VPM_CALLBACK_DATA_MIDDLE: will be called per 60 ms
*VPM_CALLBACK_DATA_TAIL: end
*/
static void vpm_asr_data_handle(const void *, const vpm_callback_data_t * data)
{
  if (audiodata_silence_mode) {
    printf("vc:vpm: %s [=====>>>>>] silence mode return\n", __func__);
    return;
  }

  switch (data->type) {
    case VPM_CALLBACK_DATA_HEADER /* constant-expression */:
      printf("vc:vpm: %s, asr VPM_CALLBACK_DATA_HEADER\n", __func__);
      if (isNeedDump) {
        audioDebuggerPtr->audioDumpFileOpen(
          AUDIO_DEBUG_CONFIG_DUMPASR,
          &asrDumpFileFd);
      }
      /*For silence debug mode*/
      if (isNeedSilence) {return;}
      if (get_aivs_engine_status() == true) {  // will check here for on/off line swith
        aivs_recognize_start();  // for online sdk
      } else {
        ai_nativeasr_data_handler(MSM_ASR_START, 0, 0);
      }
      break;
    case VPM_CALLBACK_DATA_TAIL /* constant-expression */:
      printf("vc:vpm: %s, asr VPM_CALLBACK_DATA_TAIL\n", __func__);
      if (isNeedDump) {
        audioDebuggerPtr->audioDumpFileClose(
          AUDIO_DEBUG_CONFIG_DUMPASR,
          &asrDumpFileFd);
      }
      /*For silence debug mode*/
      if (isNeedSilence) {return;}
      if (get_aivs_engine_status() == true) {
        aivs_recognize_stop();    // for online sdk
      } else {
        ai_nativeasr_data_handler(MSM_ASR_STOP, 0, 0);
      }
      set_vpm_status_unwakeup();  // reset the vmp unwakeup status for multi round dialog
      break;
    case VPM_CALLBACK_DATA_MIDDLE /* constant-expression */:
      if (isNeedDump) {
        audioDebuggerPtr->audioDumpFileWrite(
          &asrDumpFileFd,
          data->buffer.u8,
          data->buffer.size);
      }
      /*For silence debug mode*/
      if (isNeedSilence) {return;}
      if (get_aivs_engine_status() == true) {
        if (aivsFinishEventNow == true) {
          aivs_recognize_stop();
        } else {
          aivs_recognize_postData(
            data->buffer.u8,
            data->buffer.size);
        }
      } else {
        ai_nativeasr_data_handler(
          MSM_ASR_PROCESS,
          data->buffer.u8,
          data->buffer.size);
      }
      break;
    default:
      break;
  }
}

// Not support
static void vpm_voip_data_handle(const void *, const vpm_callback_data_t *)
{
  printf("vc:vpm: vpm_voip_data_handle\n");
}

// Not support
static void vpm_processed_data_handle(const void *, const vpm_callback_data_t *)
{
  printf("vc:vpm: vpm_processed_data_handle\n");
}

// Not support
int vpm_stat_point_callback(const void *, const char *, const char *)
{
  printf("vc:vpm: vpm_stat_point_callback\n");
  return 0;
}

// For unwakeup talking case
// VPM_STATUS_WAKEUP
// VPM_STATUS_UNWAKEUP
// Set the device status to wakeup sdk
// VPM_STATUS_DOG_SILENT,
// VPM_STATUS_DOG_STANDUP,
// VPM_STATUS_DOG_WALKING,
// VPM_STATUS_DOG_RUNNING,
static int setStatus(vpm_status_t st)
{
  printf("vc:vpm: setStatus\n");
  vpm_set_status(st);
  return 0;
}

int msg_playback_complete_broadcaster(int msg)
{
  /*1: complete; 0, not complete*/
  std::unique_lock<std::mutex> lock(mMutexPlayComplete);
  mMsgQueuePlayComplete.push_front(msg);
  mCondVarPlayComplete.notify_all();
  return 0;
}

void msg_playback_complete_listener()
{
  int msg = -1;
  std::unique_lock<std::mutex> lock(mMutexPlayComplete);

  printf("vc:vpm: enter %s()\n", __func__);
  mCondVarPlayComplete.wait(lock, [] {return !mMsgQueuePlayComplete.empty();});
  msg = mMsgQueuePlayComplete.back();
  mMsgQueuePlayComplete.pop_back();
  mCondVarPlayComplete.notify_all();

  if (msg == 1) {
    setStatus(VPM_STATUS_WAKEUP);
    printf("vc:vpm: in %s(), set vpm status to VPM_STATUS_WAKEUP\n", __func__);
  }
  printf("vc:vpm: exit %s(), msg[%d]\n", __func__, msg);
}

int set_vpm_status_wakeup()
{
  printf("vc:vpm: set_vpm_status_wakeup()\n");
  return setStatus(VPM_STATUS_WAKEUP);
}

int set_vpm_status_wakeup_postpone(void)
{
  printf("vc:vpm: enter %s()\n", __func__);
  auto th = std::thread(msg_playback_complete_listener);
  th.detach();
  printf("vc:vpm: exit %s()\n", __func__);
  return 0;
}

int set_vpm_status_unwakeup()
{
  // return 0;// debug here now, will change it later
  printf("vc:vpm: set_vpm_status_unwakeup()\n");
  return setStatus(VPM_STATUS_UNWAKEUP);
}

// VPM_PARAS_TYPE_VOICE_WAKEUP_SWITCH  for  enable/disable normal wakeup fuction
// normal wakeup case :1
// enroll case :0
static int setParas(int type, int value)
{
  vpm_paras_t para;
  para.type = (vpm_paras_type_t)type;
  para.value = reinterpret_cast<void *>(&value);
  return vpm_set_paras(&para);
}

void ai_vpm_work_loop()
{
  fprintf(stdout, "vc: ai vp work loop on call\n");
  for (;; ) {
    if (getAiWakeupRunStatus() == false) {
      printf("vc:vpm: getAiWakeupRunStatus shitch off\n");
      sleep(1);
      continue;
    }
    vpm_audio_buffer_t input;
    usleep(1000 * 20);
    if (vpm_buf_ready == false) {
      continue;
    }

    vpm_input_buf_t vpm_data;
    vpm_msg_queue.DeQueue(vpm_data);
    input.raw = reinterpret_cast<void *>(vpm_data.start);
    input.size = vpm_data.length;
    vpm_process(&input);
  }

  /*release vpm*/
  printf("vc:vpm: release  vpm resource\n");
  vpm_stop();
  vpm_release();
}

int ai_vpm_engine_setup(void)
{
  vpm_effect_config_t vpm_cfg;

  audioDebuggerPtr = std::make_shared<std::audioDebugger>();
  isNeedDump = audioDebuggerPtr->getConfig(AUDIO_DEBUG_CONFIG_DUMPASR);
  isNeedSilence = audioDebuggerPtr->getConfig(AUDIO_DEBUG_CONFIG_SILENCE);
  printf("vc:vpm: audioconfig asrDump[%d], silenceMode[%d]\n", isNeedDump, isNeedSilence);

  vpm_cfg.wakeup_prefix = 600;
  vpm_cfg.wakeup_suffix = 600;
  vpm_cfg.wait_asr_timeout = 3000;

  // vad_timeout suggested value 400ms
  vpm_cfg.vad_timeout = 400;  // can adjust here for timeout
  vpm_cfg.vad_switch = 1;
  vpm_cfg.wakeup_data_switch = 1;
  vpm_cfg.processed_data_switch = 0;

  // taget_score not support for now
  vpm_cfg.target_score = 0.3;
  vpm_cfg.effect_mode = VPM_EFFECT_MODE_ASR;
  vpm_cfg.cfg_file_path = VPM_CONFIG_FILE_PATH;
  vpm_cfg.priv = reinterpret_cast<void *>(&vpm_cfg);
  vpm_cfg.log_callback = vpm_log_cb;
  vpm_cfg.event_callback = vpm_event_handle;
  vpm_cfg.wakeup_data_callback = vpm_wakeup_data_handle;
  vpm_cfg.asr_data_callback = vpm_asr_data_handle;
  vpm_cfg.voip_data_callback = vpm_voip_data_handle;
  vpm_cfg.processed_data_callback = vpm_processed_data_handle;
  vpm_cfg.stat_point_callback = vpm_stat_point_callback;
  vpm_cfg.effect_mode = VPM_EFFECT_MODE_ASR;   // 1for asr 2 for VOIP mode

  printf("vc:vpm: start init ....\n");
  if (0 != vpm_init((const vpm_effect_config_t *)&vpm_cfg)) {
    printf("vc:vpm: vpm_init failed ....\n");
  }

  printf("vc:vpm: vpm_start...\n");
  vpm_start();

  // enable wakeup
  setParas(10, 1);

  return 0;
}

void ai_vpm_task(void)
{
  ai_vpm_engine_setup();
  ai_vpm_work_loop();
}
