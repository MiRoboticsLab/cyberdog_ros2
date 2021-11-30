// Copyright (c) 2021  Beijing Xiaomi Mobile Software Co., Ltd.
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

#ifndef CAMERA_ALGO__ALGORITHM_BASE_HPP_
#define CAMERA_ALGO__ALGORITHM_BASE_HPP_

#include <atomic>
#include <Thread.h>
#include "camera_utils/utils.hpp"
#include "camera_utils/queue.hpp"
#include "camera_base/camera_buffer.hpp"

namespace cyberdog_camera
{

using ArgusSamples::Thread;

class AlgorithmBase : public Thread
{
public:
  AlgorithmBase();
  virtual ~AlgorithmBase();
  void enqueueBuffer(ImageBuffer buffer);

  void setBufferDoneCallback(void (* callback)(ImageBuffer, void *), void * arg)
  {
    m_doneCb = callback;
    m_callbackArg = arg;
  }

protected:
  //  Thread methods
  virtual bool threadInitialize();
  virtual bool threadExecute();
  virtual bool threadShutdown();

  virtual bool init();
  virtual bool processImageBuffer(ImageBuffer & buffer) = 0;
  virtual bool isIdle() = 0;
  void bufferDone(ImageBuffer buffer);

private:
  static void * ThreadFunc(void * _this)
  {
    (reinterpret_cast<AlgorithmBase *>(_this))->initThreadFunc();
    return NULL;
  }
  void initThreadFunc();
  Queue<ImageBuffer> m_bufferQ;

  pthread_t m_initThread;
  std::atomic<bool> m_initialized;
  void (* m_doneCb)(ImageBuffer, void *);
  void * m_callbackArg;
};

}  // namespace cyberdog_camera

#endif  // CAMERA_ALGO__ALGORITHM_BASE_HPP_
