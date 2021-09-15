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

#define LOG_TAG "AlgorithmBase"
#include "camera_algo/algorithm_base.hpp"
#include "camera_utils/log.hpp"

namespace cyberdog_camera
{

AlgorithmBase::AlgorithmBase()
: m_initialized(false)
{
}

AlgorithmBase::~AlgorithmBase()
{
}

bool AlgorithmBase::threadInitialize()
{
  if (pthread_create(&m_initThread, NULL, ThreadFunc, this) != 0) {
    return false;
  }

  return true;
}

bool AlgorithmBase::threadExecute()
{
  while (true) {
    ImageBuffer buffer = m_bufferQ.pop();
    if (!processImageBuffer(buffer)) {
      CAM_INFO("end algo thread.");
      break;
    }
  }

  requestShutdown();
  return true;
}

bool AlgorithmBase::threadShutdown()
{
  if (pthread_join(m_initThread, NULL) != 0) {
    return false;
  }

  return true;
}

bool AlgorithmBase::init()
{
  return true;
}

void AlgorithmBase::initThreadFunc()
{
  init();

  m_initialized = true;
}

void AlgorithmBase::enqueueBuffer(ImageBuffer buffer)
{
  /* NULL buffer indicates thread end, so always enqueue it. */
  if (buffer.data == NULL || (m_initialized && isIdle())) {
    m_bufferQ.push(buffer);
  } else {
    bufferDone(buffer);
  }
}

void AlgorithmBase::bufferDone(ImageBuffer buffer)
{
  if (m_doneCb) {
    m_doneCb(buffer, m_callbackArg);
  }
}

}  // namespace cyberdog_camera
