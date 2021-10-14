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

#include <ifaddrs.h>

#include <mutex>
#include <deque>
#include <vector>
#include <memory>
#include <iostream>
#include <fstream>
#include <thread>
#include <string>
#include <condition_variable>

#include "sys/unistd.h"
#include "netinet/in.h"
#include "arpa/inet.h"
#include "sys/types.h"
#include "sys/ioctl.h"
#include "net/if.h"

#include "xiaoai_sdk/aivs/General.h"
#include "xiaoai_sdk/aivs/Engine.h"
#include "xiaoai_sdk/aivs/InstructionCapability.h"
#include "xiaoai_sdk/aivs/SpeechRecognizer.h"
#include "xiaoai_sdk/aivs/SpeechSynthesizer.h"
#include "xiaoai_sdk/aivs/Template.h"
#include "xiaoai_sdk/aivs/Logger.h"
#include "xiaoai_sdk/aivs/AuthCapabilityImpl.h"
#include "xiaoai_sdk/aivs/Dialog.h"
#include "xiaoai_sdk/aivs/Execution.h"
#include "xiaoai_sdk/aivs/Nlp.h"
#include "xiaoai_sdk/aivs/StorageCapabilityImpl.h"
#include "xiaoai_sdk/aivs/DigestUtils.h"
#include "xiaoai_sdk/aivs/ConnectionCapabilityImpl.h"
#include "xiaoai_sdk/aivs/ErrorCapabilityImpl.h"
#include "xiaoai_sdk/aivs/Settings.h"
#include "xiaoai_sdk/aivs/ClientLoggerHooker.h"
#include "xiaoai_sdk/audio_config/config.h"
#include "audio_assistant/audio_assitant.hpp"
#include "audio_base/audio_queue.hpp"
#include "audio_assistant/mp3decoder.hpp"

#include "xiaoai_sdk/aivs/RobotController.h"

/**/
bool online_sdk_ready = false;
bool aivs_onlie_switch_on = true;
static std::mutex status_mutex;
static std::mutex vad_mutex;

/*for debug*/
static bool mUseTestDeviceId = false;

bool aivsFinishEventNow = false;

// sdk engine实例，建议全局只有一个实例。
std::shared_ptr<aivs::Engine> gEngine;
bool gArsLoop = false;
bool gMainLoop = true;

/*for action transfer*/
static std::mutex mMutex;
static std::condition_variable mCondVar;
static std::deque<int> mMsgQueue;

/*for playback complete sync*/
static std::mutex mMutexNetwork;
static std::condition_variable mCondVarNetwork;
static std::deque<int> mMsgQueueNetwork;

static bool mAivsTtsRunStatus = true;
static bool mAiOnlineRunStatus = true;

typedef enum
{
  TTS_TYPE_STREAM,
  TTS_TYPE_SYNC_START,
  TTS_TYPE_SYNC_STOP,
  TTS_TYPE_INVALID
} tts_type_t;

typedef struct tts_info
{
  tts_type_t type;
  unsigned char * address;
  unsigned int length;
  void free_mem()
  {
    if (address != nullptr) {
      fprintf(stdout, "vc: free ttsinfo memory\n");
      free(address);
    }
  }
} tts_info_t;

typedef struct tts_queue
{
  std::deque<tts_info_t> queue;
  std::mutex mutex;
} tts_queue_t;
static tts_queue_t mMsgQueueTts;

static athena_audio::MsgQueue<tts_info_t> TTSMsgQueue;

bool getAivsTtsRunStatus(void)
{
  return mAivsTtsRunStatus;
}

int setAivsTtsRunStatus(bool on)
{
  mAivsTtsRunStatus = on;
  return 0;
}

bool getAiOnlineRunStatus(void)
{
  return mAiOnlineRunStatus;
}

int setAiOnlineRunStatus(bool on)
{
  mAiOnlineRunStatus = on;
  return 0;
}

int ai_push_msg(int msg)
{
  std::unique_lock<std::mutex> lock(mMutex);
  mMsgQueue.push_front(msg);
  mCondVar.notify_all();
  return 0;
}

int tts_player_decode_paly(void);

int tts_stream_flush(void)
{
  TTSMsgQueue.Clear();
  return 0;
}

int aivs_set_interrupt(void)
{
  std::cout << "vc: enter aivs_set_interrupt()" << std::endl;
  if (gEngine) {
    gEngine->interrupt();
    std::cout << "vc: sent gEngine->interrupt()" << std::endl;
  }
  std::cout << "vc: exit aivs_set_interrupt()" << std::endl;
  return 0;
}

int ai_push_msg_playback(const tts_info_t & ttsInfo)
{
  std::cout << "vc: enter ai_push_msg_playback(), type: " << ttsInfo.type << std::endl;
  tts_info_t ttsInfoPlay = {ttsInfo.type, NULL, ttsInfo.length};

  switch (ttsInfo.type) {
    case TTS_TYPE_STREAM:
      std::cout << "vc: TTS_TYPE_STREAM Case !!" << std::endl;

      if (ttsInfo.address == NULL || ttsInfo.length == 0) {
        std::cout << "vc: ai_push_msg_playback invalid paramter, stream addr is null !!!" <<
          std::endl;
        return -1;
      } else {
        ttsInfoPlay.address = (unsigned char *)malloc(ttsInfo.length);
        if (ttsInfoPlay.address != NULL) {
          memcpy(ttsInfoPlay.address, ttsInfo.address, ttsInfoPlay.length);
          TTSMsgQueue.EnQueue(ttsInfoPlay);
        } else {
          std::cout << "vc: ai_push_msg_playback failed, cannot allocate memory!" << std::endl;
        }
      }
      break;

    case TTS_TYPE_SYNC_START:
      std::cout << "vc: TTS_TYPE_SYNC_START Case !!" << std::endl;
      TTSMsgQueue.EnQueue(ttsInfoPlay);
      break;

    case TTS_TYPE_SYNC_STOP:
      std::cout << "vc: TTS_TYPE_SYNC_STOP Case !!" << std::endl;
      TTSMsgQueue.EnQueue(ttsInfoPlay);
      break;

    default:
      std::cout << "vc: enter ai_push_msg_playback() default case !!" << std::endl;
      break;
  }

  return 0;
}

int aivsTtsHandler(void)
{
  std::cout << "vc: enter aivsTtsHandler() " << std::endl;
  for (;; ) {
    tts_info_t ttsInfoPlayback = {TTS_TYPE_INVALID, NULL, 0};

    if (getAivsTtsRunStatus() == false) {
      printf("[AUDIO][TTS] getAivsTtsRunStatus switch off\n");
      usleep(1000 * 1000);
      continue;
    }

    if (!TTSMsgQueue.DeQueue(ttsInfoPlayback)) {
      std::cout << "vc: aivsTtsHandler get msg with empty, skip once!" << std::endl;
      continue;
    }

    switch (ttsInfoPlayback.type) {
      case TTS_TYPE_STREAM:
        if (ttsInfoPlayback.address == NULL || ttsInfoPlayback.length == 0) {
          usleep(1000 * 100);
          continue;
        } else {
          std::cout << "vc: Process TTS_TYPE_STREAM " << std::endl;
          cyberdog_audio::mp3decoder::GetInstance()->tts_player_accumulate(
            ttsInfoPlayback.address,
            ttsInfoPlayback.length);
          free(ttsInfoPlayback.address);
        }
        break;

      case TTS_TYPE_SYNC_START:
        std::cout << "vc: Process TTS_TYPE_SYNC_START " << std::endl;
        cyberdog_audio::mp3decoder::GetInstance()->tts_player_init();
        ai_push_msg(101);      /*led control for start-dialog status*/
        break;

      case TTS_TYPE_SYNC_STOP:
        std::cout << "vc: Process TTS_TYPE_SYNC_STOP " << std::endl;

        if (cyberdog_audio::mp3decoder::GetInstance() == nullptr) {
          fprintf(stderr, "Process TTS_TYPE_SYNC_STOP Error with invalid pointer!\n");
          break;
        }

        cyberdog_audio::mp3decoder::GetInstance()->tts_player_decode_paly();
        // ai_push_msg(104);/*led control for end -dialog status*/
        break;

      default:
        std::cout << "vc: Process default case !!" << std::endl;
        break;
    }
  }
}

void ttsHandlerTask(void)
{
  printf("vc:ttsplay enter %s\n", __func__);
  aivsTtsHandler();  // loop
}

/*for network status*/
static int network_status = 0;
static int vad_status = -1;

int aivs_recognize_stop(void);
int set_vpm_status_wakeup(void);
int set_vpm_status_wakeup_postpone(void);
int set_vpm_status_unwakeup(void);

int __notify_aivs_network_status(int msg)
{
  /*-1: unconnected; 1: connected*/
  std::unique_lock<std::mutex> lock(mMutexNetwork);
  mMsgQueueNetwork.push_front(msg);
  mCondVarNetwork.notify_all();
  return 0;
}

int get_network_status()
{
  return network_status;
}

int set_network_status(int status)
{
  network_status = status;
  __notify_aivs_network_status(status);
  std::cout << "vc: set_network_status " <<
    network_status <<
    std::endl;
  return 0;
}

int get_vad_status()
{
  int status;
  vad_mutex.lock();
  status = vad_status;
  vad_mutex.unlock();
  return status;
}
int set_vad_status(int status)
{
  vad_mutex.lock();
  vad_status = status;
  vad_mutex.unlock();
  return 0;
}

int set_aivs_onlie_switch(bool on)
{
  aivs_onlie_switch_on = on;
  return 0;
}

int set_aivs_engine_status(bool isEnabled)
{
  status_mutex.lock();
  online_sdk_ready = isEnabled;
  status_mutex.unlock();
  return 0;
}
bool get_aivs_engine_status(void)
{
  bool status = false;

  status_mutex.lock();
  status = (online_sdk_ready && aivs_onlie_switch_on);
  status_mutex.unlock();

  return status;
}

int aivseMsgHandler(void)
{
  int msg = -1;
  std::unique_lock<std::mutex> lock(mMutex);

  mCondVar.wait(lock, [] {return !mMsgQueue.empty();});
  msg = mMsgQueue.back();
  mMsgQueue.pop_back();
  mCondVar.notify_all();
  return msg;
}

void set_useTestDeviceId(bool useTestDeviceId)
{
  mUseTestDeviceId = useTestDeviceId;
}

void handleInstruction(std::shared_ptr<aivs::Instruction> & instruction)
{
  const std::string & ns = instruction->getHeader()->getNamespace();
  const std::string & name = instruction->getHeader()->getName();

  std::cout << "vc: in handleInstruction() !!!" << std::endl;

  if (ns == aivs::SpeechRecognizer::NAMESPACE) {  // ASR相关指令
    if (name == aivs::SpeechRecognizer::RecognizeResult::NAME) {  // ASR结果
      auto payload = std::static_pointer_cast<aivs::SpeechRecognizer::RecognizeResult>(
        instruction->getPayload());
      auto results = payload->getResults();
      if (results.empty()) {
        std::cout << "vc: [WARN]no ASR result" << std::endl;
        return;
      }

      if (payload->isFinal()) {  // ASR final
        std::cout << "vc: [ASR.final]" << results[0]->getText() << std::endl;
      } else {  // ASR partial
        std::cout << "vc: [ASR.partial]" << results[0]->getText() << std::endl;
      }
    } else if (name == aivs::SpeechRecognizer::ExpectSpeech::NAME) {
      std::cout << "vc: [ASR.ExpectSpeech]" << std::endl;
      // 服务端要求客户端做多轮会话。
      // 客户端收到此指令后，需要再次打开麦克风，上次ASR Event，完成多轮会话。
      // 请在另外一个线程里面开启ASR录音。
      set_vpm_status_wakeup_postpone();
    } else if (name == aivs::SpeechRecognizer::StopCapture::NAME) {
      std::cout << "vc: [ASR.StopCapture]" << std::endl;
      // 服务端不再需要上传语音数据，客户端需要关闭麦克风，并发送ASR结束Event
      // XXX sdk vad, need handle , no need to send data to sdk
      aivsFinishEventNow = true;
      gArsLoop = false;
    }
  } else if (ns == aivs::SpeechSynthesizer::NAMESPACE) {  // TTS相关指令
    tts_info_t ttsInfo;
    ttsInfo.address = NULL;
    ttsInfo.length = 0;
    if (name == aivs::SpeechSynthesizer::Speak::NAME) {  // 开始下发TTS流的指令
      // 客户端在收到该指令后，初始化本地TTS播放器
      std::cout << "vc: [TTS]stream begin" << std::endl;
      ttsInfo.type = TTS_TYPE_SYNC_START;
      ai_push_msg_playback(ttsInfo);
    } else if (name == aivs::SpeechSynthesizer::FinishSpeakStream::NAME) {  // TTS流下发完毕的指令
      // 收到此指令后，表示服务的的TTS流已经全部下发完毕。
      // 客户端根据自身TTS播放器特点，来决定后续事情
      std::cout << "vc: [TTS]stream end" << std::endl;
      ttsInfo.type = TTS_TYPE_SYNC_STOP;
      ai_push_msg_playback(ttsInfo);
    }
  } else if (ns == "RobotController") {
    printf("vc: ==> online skd RobotController \n");
    if (name == "Operate") {
      /*    PROCESS INSCTRUCTIONS    */
      auto payload = std::static_pointer_cast<aivs::RobotController::Operate>(
        instruction->getPayload());
      aivs::RobotController::RobotAction action = payload->getAction();
      int message = -1;
      switch (action) {
        case aivs::RobotController::RobotAction::STANDING:
          message = 1;
          break;
        case aivs::RobotController::RobotAction::DOWN:
          message = 2;
          break;
        case aivs::RobotController::RobotAction::COME:
          message = 3;
          break;
        case aivs::RobotController::RobotAction::BACK:
          message = 4;
          break;
        case aivs::RobotController::RobotAction::GO_ROUND:
          message = 5;
          break;
        case aivs::RobotController::RobotAction::HIGH_FIVE:
          message = 6;
          break;
        case aivs::RobotController::RobotAction::BACK_SOMERSAULT:
          message = -1;          // not support
          break;
        case aivs::RobotController::RobotAction::DANCE:
          message = 7;
          break;
        default:
          break;
      }
      printf("vc: ==> Action to decision model: %d\n", message);
      ai_push_msg(message);
    }
  } else if (ns == aivs::Dialog::NAMESPACE) {
    if (name == aivs::Dialog::Finish::NAME) {   // 当前Event的结果全部下发完毕，会话结束
      auto dialogId = instruction->getHeader()->getDialogId();
      // XXX need close mic ?
      std::cout << "vc: [Dialog]finish eventId=" << (dialogId.has_value() ? *dialogId : "null") <<
        std::endl;
      gMainLoop = false;
    }
  }
}

class MyInstructionCapability : public aivs::InstructionCapability
{
public:
  // Instruction回调，回调数据是服务端下发的指令
  virtual bool process(std::shared_ptr<aivs::Instruction> & instruction)
  {
    std::cout << "vc:[Instruction] process: " << instruction->toString() << std::endl;
    handleInstruction(instruction);
    return true;
  }

  // 二进制数据回调，回调数据是服务端下发的TTS二进制流
  // 客户端不需要释放data的内存，此回调返回后，data会被SDK释放。
  virtual bool process(const uint8_t * data, uint32_t length)
  {
    std::cout << "vc: process: data addr=" <<
      static_cast<const void *>(data) <<
      " , length=" << std::dec << length <<
      std::endl;
    tts_info_t ttsInfo;
    ttsInfo.type = TTS_TYPE_STREAM;
    ttsInfo.address = (unsigned char *)data;
    ttsInfo.length = length;
    ai_push_msg_playback(ttsInfo);

    return true;
  }
};

/**
 * 获取设备的deviceid，需要保证每个设备的device id不同；
 * 此处以网卡 eth0 的mac地址为例,集成SDK时根据设备具体情况选择其他网卡，或者使用设备的其他唯一特征计算device id
 */
std::string getDeviceId()
{
  std::cout << "vc: linux_device_id" << std::endl;
  // 对mac地址做md5
  std::string deviceId;
  struct ifreq ifr_mac;
  struct ifconf ifc;
  char mac_addr[32] = {0};
  char buf[1024];
  bool success = false;

  int sock_mac = socket(AF_INET, SOCK_DGRAM, IPPROTO_IP);
  if (sock_mac == -1) {
    perror("open socket failed/n");
    exit(-1);
  }

  ifc.ifc_len = sizeof(buf);
  ifc.ifc_buf = buf;
  if (ioctl(sock_mac, SIOCGIFCONF, &ifc) == -1) {
    perror("socket operation failed/n");
    exit(-1);
  }

  struct ifreq * it = ifc.ifc_req;
  const struct ifreq * const end = it + (ifc.ifc_len / sizeof(struct ifreq));

  for (; it != end; ++it) {
    memset(&ifr_mac, 0, sizeof(ifr_mac));
    snprintf(ifr_mac.ifr_name, sizeof(ifr_mac.ifr_name), "%s", it->ifr_name);
    std::cout << "vc: try to use " << ifr_mac.ifr_name << "vc:  to calculate device id" <<
      std::endl;
    if (ioctl(sock_mac, SIOCGIFFLAGS, &ifr_mac) == 0) {
      if (!(ifr_mac.ifr_flags & IFF_LOOPBACK)) {
        if (ioctl(sock_mac, SIOCGIFHWADDR, &ifr_mac) == 0) {
          success = true;
          break;
        }
      }
    } else {
      perror("socket operation failed/n");
      exit(-1);
    }
  }

  if (!success) {
    perror("get deviceid from mac failed/n");
    exit(-1);
  }

  snprintf(
    mac_addr,
    sizeof(mac_addr),
    "%02x%02x%02x%02x%02x%02x",
    (unsigned char)ifr_mac.ifr_hwaddr.sa_data[0],
    (unsigned char)ifr_mac.ifr_hwaddr.sa_data[1],
    (unsigned char)ifr_mac.ifr_hwaddr.sa_data[2],
    (unsigned char)ifr_mac.ifr_hwaddr.sa_data[3],
    (unsigned char)ifr_mac.ifr_hwaddr.sa_data[4],
    (unsigned char)ifr_mac.ifr_hwaddr.sa_data[5]);

  close(sock_mac);
  std::cout << "vc: get mac addr: " << mac_addr << std::endl;
  DigestUtils::md5((const unsigned char *)mac_addr, strlen(mac_addr), deviceId);

  std::cout << "vc: get device id: " << deviceId.c_str() << std::endl;

  return deviceId;
}

int testAsrTtsNlp(const std::string & file)
{
  std::ifstream input(file.c_str(), std::ifstream::binary);
  if (!input.good()) {
    std::cout << "vc: ERROR: failed to open file " << file << std::endl;
    gMainLoop = true;
    return -1;
  }

  // 发送开始的Recognize Event
  auto recognizePayload = std::make_shared<aivs::SpeechRecognizer::Recognize>();
  std::shared_ptr<Event> recognizeEvent;
  if (!aivs::Event::build(recognizePayload, gEngine->genUUID(), recognizeEvent)) {
    std::cout << "vc: ERROR: failed to build recognizeEvent" << std::endl;
    gMainLoop = false;
    return -1;
  }
  if (!gEngine->postEvent(recognizeEvent)) {
    std::cout << "vc: ERROR: failed to post event " << recognizeEvent->toString() << std::endl;
    gMainLoop = false;
    return -1;
  }

  int bufferSize = 4096;
  char * data = new char[bufferSize];
  gArsLoop = true;
  while (gMainLoop && gArsLoop && !input.eof()) {
    input.read(data, bufferSize);     // 模拟麦克风录音
    int n = input.gcount();
    if (n > 0) {
      if (!gEngine->postData(reinterpret_cast<uint8_t *>(data), n)) {
        std::cout << "vc: ERROR: failed to post data " << n << std::endl;
        gMainLoop = false;
        return -1;
      }
    } else {
      break;
    }
  }

  delete[] data;

  // 发送结束的Event
  auto finishPayload = std::make_shared<aivs::SpeechRecognizer::RecognizeStreamFinished>();
  std::shared_ptr<Event> finishEvent;
  if (!aivs::Event::build(
      finishPayload,
      recognizeEvent->getHeader()->getId(), finishEvent))
  {  // 必须与开始的Event共用ID
    std::cout << "vc: ERROR: failed to build RecognizeStreamFinished" << std::endl;
    gMainLoop = false;
    return -1;
  }
  if (!gEngine->postEvent(finishEvent)) {
    std::cout << "vc: ERROR: failed to post event " << finishEvent->toString() << std::endl;
    gMainLoop = false;
    return -1;
  }

  return 0;
}

int testAsr(const std::string & file)
{
  std::ifstream input(file.c_str(), std::ifstream::binary);
  if (!input.good()) {
    std::cout << "vc: ERROR: failed to open file " << file << std::endl;
    gMainLoop = true;
    return -1;
  }

  // 纯ASR请求，通过Event Context禁掉NLP和TTS
  std::vector<aivs::Execution::RequestControlType> disableTypes;
  disableTypes.push_back(aivs::Execution::RequestControlType::NLP);
  disableTypes.push_back(aivs::Execution::RequestControlType::TTS);
  auto control = std::make_shared<aivs::Execution::RequestControl>();
  control->setDisabled(disableTypes);
  std::shared_ptr<Context> context;
  if (!aivs::Context::build(control, context)) {
    std::cout << "vc: ERROR: failed to build RequestControl" << std::endl;
    gMainLoop = false;
    return -1;
  }

  // 发送开始的Recognize Event
  auto recognizePayload = std::make_shared<aivs::SpeechRecognizer::Recognize>();
  std::shared_ptr<Event> recognizeEvent;
  if (!aivs::Event::build(recognizePayload, gEngine->genUUID(), recognizeEvent)) {
    std::cout << "vc: ERROR: failed to build recognizeEvent" << std::endl;
    gMainLoop = false;
    return -1;
  }
  recognizeEvent->addContext(context);   // 添加禁止NLP和TTS的Context
  if (!gEngine->postEvent(recognizeEvent)) {
    std::cout << "vc: ERROR: failed to post event " << recognizeEvent->toString() << std::endl;
    gMainLoop = false;
    return -1;
  }

  int bufferSize = 4096;
  char * data = new char[bufferSize];
  gArsLoop = true;
  while (gMainLoop && gArsLoop && !input.eof()) {
    input.read(data, bufferSize);     // 模拟麦克风录音
    int n = input.gcount();
    if (n > 0) {
      if (!gEngine->postData(reinterpret_cast<uint8_t *>(data), n)) {
        std::cout << "vc: ERROR: failed to post data " << n << std::endl;
        gMainLoop = false;
        return -1;
      }
    } else {
      break;
    }
  }

  delete[] data;

  // 发送结束的Event
  auto finishPayload = std::make_shared<aivs::SpeechRecognizer::RecognizeStreamFinished>();
  std::shared_ptr<Event> finishEvent;
  if (!aivs::Event::build(
      finishPayload,
      recognizeEvent->getHeader()->getId(), finishEvent))
  {  // 必须与开始的Event共用ID
    std::cout << "vc: ERROR: failed to build RecognizeStreamFinished" << std::endl;
    gMainLoop = false;
    return -1;
  }
  if (!gEngine->postEvent(finishEvent)) {
    std::cout << "vc: ERROR: failed to post event " << finishEvent->toString() << std::endl;
    gMainLoop = false;
    return -1;
  }

  return 0;
}

int testNlpTts()
{
  // 发送Nlp.Request Event
  auto nlpPayload = std::make_shared<aivs::Nlp::Request>();
  std::string query = "今天的天气";
  nlpPayload->setQuery(query);
  std::shared_ptr<Event> nlpEvent;
  gMainLoop = true;
  if (!aivs::Event::build(nlpPayload, gEngine->genUUID(), nlpEvent)) {
    std::cout << "vc: ERROR: failed to build nlpEvent" << std::endl;
    gMainLoop = false;
    return -1;
  }

  if (!gEngine->postEvent(nlpEvent)) {
    std::cout << "vc: ERROR: failed to post event " << nlpEvent->toString() << std::endl;
    gMainLoop = false;
    return -2;
  }

  return 0;
}

int testTts()
{
  std::cout << "vc: in testTts()" << std::endl;
  // 发送Nlp.Request Event
  auto ttsPayload = std::make_shared<aivs::SpeechSynthesizer::Synthesize>();
  std::string text = "今天好冷啊";
  ttsPayload->setText(text);
  std::shared_ptr<Event> ttsEvent;
  gMainLoop = true;
  if (!aivs::Event::build(ttsPayload, gEngine->genUUID(), ttsEvent)) {
    std::cout << "vc: ERROR: failed to build ttsEvent" << std::endl;
    gMainLoop = false;
    return -1;
  }

  if (!gEngine->postEvent(ttsEvent)) {
    std::cout << "vc: ERROR: failed to post event " << ttsEvent->toString() << std::endl;
    gMainLoop = false;
    return -2;
  }

  return 0;
}


void initAivsDeviceOAuth()
{
  /**
   * Device OAuth鉴权
   */
  std::string DEVICE_OAUTH_DEVICE_ID;
  if (mUseTestDeviceId) {
    DEVICE_OAUTH_DEVICE_ID = "ceb58b4e4a54aed0e655af8fa7e411c1";
    std::cout << "vc: Use TestDeviceId: " << DEVICE_OAUTH_DEVICE_ID.c_str() << std::endl;
  } else {
    DEVICE_OAUTH_DEVICE_ID = getDeviceId();
    std::cout << "vc: Use Real DeviceId: " << DEVICE_OAUTH_DEVICE_ID.c_str() << std::endl;
  }


  std::cout << "vc: initAivsDeviceOAuth config" << std::endl;

  std::shared_ptr<Settings::ClientInfo> clientInfo = std::make_shared<Settings::ClientInfo>();
  clientInfo->setDeviceId(DEVICE_OAUTH_DEVICE_ID);

  auto config = getAudioConfig();
  config->putBoolean(aivs::AivsConfig::Auth::REQ_TOKEN_HYBRID, true);
  config->putBoolean(aivs::AivsConfig::Auth::REQ_TOKEN_BY_SDK, true);
  gEngine = aivs::Engine::create(config, clientInfo, aivs::Engine::ENGINE_AUTH_DEVICE_OAUTH);
}

int initEngine()
{
  initAivsDeviceOAuth();

  aivs::Logger::setLevel(aivs::Logger::LOG_LEVEL_DEBUG);
  std::cout << "vc: initEngine" << std::endl;
  // 默认情况下SDK log或打印到标准输出，客户端根据需要可以setLoggerHooker对SDK log进行处理
  // 例如保存到文件等，参考ClientLoggerHooker
  // std::shared_ptr<ClientLoggerHooker> loggerHooker = std::make_shared<ClientLoggerHooker>();
  // gEngine->setLoggerHooker(loggerHooker);

  // 根据需求注册相应的Capability
  gEngine->registerCapability(std::make_shared<MyInstructionCapability>());
  gEngine->registerCapability(std::make_shared<AuthCapabilityImpl>());
  gEngine->registerCapability(std::make_shared<StorageCapabilityImpl>());
  gEngine->registerCapability(std::make_shared<ConnectionCapabilityImpl>());
  gEngine->registerCapability(std::make_shared<ErrorCapabilityImpl>());

  std::cout << "vc: before engine start" << std::endl;
  if (!gEngine->start()) {  // 耗时操作，不建议放到主线程（或UI线程）。
    std::cout << "vc: ERROR: failed to start Engine" << std::endl;
    return -1;
  }
  std::cout << "vc: engine start sucessfully" << std::endl;
  return 0;
}

std::shared_ptr<Event> recognizeEvent = nullptr;
int aivs_recognize_start(void)
{
  std::cout << "vc: >>enter aivs_recognize_start()" << std::endl;
  bool is_ready = false;

  is_ready = get_aivs_engine_status();

  if (!is_ready) {
    std::cout << "vc: ai online is not ready" << std::endl;
    return -1;
  }

  auto state = std::make_shared<aivs::General::RequestState>();
  std::string tts_vendor = "XiaoMi_DOG";
  state->setTtsVendor(tts_vendor);
  std::shared_ptr<Context> context;
  aivs::Context::build(state, context);

  // 发送开始的Recognize Event
  auto recognizePayload = std::make_shared<aivs::SpeechRecognizer::Recognize>();

  if (!aivs::Event::build(recognizePayload, gEngine->genUUID(), recognizeEvent)) {
    std::cout << "vc: ERROR: failed to build recognizeEvent" << std::endl;
    gMainLoop = false;
    return -1;
  }

  recognizeEvent->addContext(context);  // set tts vendor

  if (!gEngine->postEvent(recognizeEvent)) {
    std::cout << "vc: ERROR: failed to post event " << recognizeEvent->toString() << std::endl;
    gMainLoop = false;
    return -1;
  }

  return 0;
}

int aivs_recognize_postData(uint8_t * data, uint32_t length)
{
  std::cout << "vc: >>enter aivs_recognize_postData()" << std::endl;
  bool is_ready = false;

  is_ready = get_aivs_engine_status();

  if (!is_ready) {
    std::cout << "vc: ai online is not ready" << std::endl;
    return -1;
  }

  if (!gEngine->postData(reinterpret_cast<uint8_t *>(data), length)) {
    std::cout << "vc: ERROR: failed to post data " << length << std::endl;
    return -1;
  }

  return 0;
}

int aivs_recognize_stop(void)
{
  std::cout << "vc: >>enter aivs_recognize_stop()" << std::endl;
  bool is_ready = false;

  is_ready = get_aivs_engine_status();

  if (!is_ready) {
    std::cout << "vc: ai online is not ready" << std::endl;
    return -1;
  }

  // 发送结束的Event
  auto finishPayload = std::make_shared<aivs::SpeechRecognizer::RecognizeStreamFinished>();
  std::shared_ptr<Event> finishEvent;

  if (recognizeEvent == nullptr) {
    fprintf(stderr, "vc: ERROR: failed with invalid pointer: recognizeEvent!\n");
    return -1;
  }

  if (!aivs::Event::build(
      finishPayload,
      recognizeEvent->getHeader()->getId(), finishEvent))
  {  // 必须与开始的Event共用ID
    std::cout << "vc: ERROR: failed to build RecognizeStreamFinished" << std::endl;
    gMainLoop = false;
    return -1;
  }

  if (!gEngine->postEvent(finishEvent)) {
    std::cout << "vc: ERROR: failed to post event " << finishEvent->toString() << std::endl;
    gMainLoop = false;
    return -1;
  }

  /*ONLINE VAD*/
  aivsFinishEventNow = false;
  return 0;
}

void releaseEngine()
{
  if (gEngine) {
    gEngine->release();
  }
}

int ai_sdk_init(void)
{
  int result = -1;
  std::string file;

  std::cout << "vc: enter " << __func__ << std::endl;
  result = initEngine();

  return result;
}

int ai_sdk_deinit(void)
{
  std::cout << "vc: exit " << __func__ << std::endl;
  releaseEngine();
  return 0;
}

void ai_online_task(void)
{
  std::cout << "vc: enter " << __func__ << std::endl;

  for (;; ) {
    int msg = -1;
    int i = 0;
    int result = -1;

    if (getAiOnlineRunStatus() == false) {
      std::cout << "[AUDIO][ONLINE] Native asr engine switch off" << std::endl;
      sleep(1);
      continue;
    }

    std::unique_lock<std::mutex> lock(mMutexNetwork);
    mCondVarNetwork.wait(lock, [] {return !mMsgQueueNetwork.empty();});
    msg = mMsgQueueNetwork.back();
    mMsgQueueNetwork.pop_back();
    mCondVarNetwork.notify_all();
    std::cout << "[AUDIO][ONLINE] ENENEN msg " << msg << std::endl;

    if (msg == 1) {
      do {
        result = ai_sdk_init();
        i++;
        std::cout << "vc: ENENEN ai_sdk_init, result = " <<
          result <<
          std::endl;
        sleep(1);
      } while (result != 0 && i < 10);

      if (result == 0) {
        set_aivs_engine_status(true);
        ai_push_msg(111);
      }
    } else if (msg == -1) {
      set_aivs_engine_status(false);
      result = ai_sdk_deinit();
      std::cout << "vc: ai_sdk_deinit, result = " <<
        result <<
        std::endl;
    } else {
      std::cout << "vc: ai_online_task invalid networkstatus " << std::endl;
    }
  }
}
