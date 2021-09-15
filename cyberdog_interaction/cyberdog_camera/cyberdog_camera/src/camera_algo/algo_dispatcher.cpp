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

#define LOG_TAG "AlgoDispatcher"
#include <string>
#include <vector>
#include "camera_algo/algo_dispatcher.hpp"
#include "camera_algo/body_detect.hpp"
#include "camera_algo/face_detect.hpp"
#include "camera_utils/log.hpp"

namespace cyberdog_camera
{

static unsigned int ALGO_INTERVAL = 3;

AlgoDispatcher & AlgoDispatcher::getInstance()
{
  static AlgoDispatcher s_instance;

  return s_instance;
}

AlgoDispatcher::AlgoDispatcher()
{
  for (int i = 0; i < ALGO_TYPE_NONE; i++) {
    m_algorithms.push_back(NULL);
  }
}

AlgoDispatcher::~AlgoDispatcher()
{
  m_algorithms.clear();
}

void AlgoDispatcher::setAlgoEnabled(int index, bool enabled)
{
  AlgorithmBase * algo = NULL;

  if (enabled) {
    if (m_algorithms[index] != NULL) {
      CAM_INFO("algo is not NULL");
      return;
    }

    switch (index) {
      case ALGO_FACE_DETECT:
        algo = new FaceDetect();
        break;
      case ALGO_BODY_DETECT:
        algo = new BodyDetect();
        break;
    }

    algo->setBufferDoneCallback(m_doneCb, m_callbackArg);
    m_algorithms[index] = algo;
    algo->initialize();
    algo->waitRunning();
  } else {
    if (m_algorithms[index] == NULL) {
      CAM_INFO("algo is NULL");
      return;
    }

    algo = m_algorithms[index];
    m_algorithms[index] = NULL;

    ImageBuffer emptyBuffer;
    memset(&emptyBuffer, 0, sizeof(emptyBuffer));
    emptyBuffer.data = NULL;
    algo->enqueueBuffer(emptyBuffer);
    algo->shutdown();
    delete algo;
  }
}

bool AlgoDispatcher::getAlgoEnabled(int index)
{
  if (index < ALGO_TYPE_NONE && m_algorithms[index] != NULL) {
    return true;
  }

  return false;
}

bool AlgoDispatcher::needProcess(uint64_t frame_id)
{
  unsigned int index = frame_id % ALGO_INTERVAL;

  if (index < ALGO_TYPE_NONE && m_algorithms[index] != NULL) {
    return true;
  } else {
    return false;
  }
}

void AlgoDispatcher::processImageBuffer(uint64_t frame_id, ImageBuffer buffer)
{
  unsigned int index = frame_id % ALGO_INTERVAL;

  if (index < ALGO_TYPE_NONE && m_algorithms[index] != NULL) {
    m_algorithms[index]->enqueueBuffer(buffer);
  } else {
    m_doneCb(buffer, m_callbackArg);
  }
}

int AlgoDispatcher::startAddingFace(const std::string & name, bool is_host)
{
  CAM_INFO("%s", name.c_str());
  AlgorithmBase * algo = m_algorithms[ALGO_FACE_DETECT];
  if (!algo) {
    return CAM_INVALID_STATE;
  }

  return (reinterpret_cast<FaceDetect *>(algo))->addFace(name, is_host);
}

int AlgoDispatcher::stopAddingFace()
{
  AlgorithmBase * algo = m_algorithms[ALGO_FACE_DETECT];
  if (!algo) {
    return CAM_SUCCESS;
  }

  return (reinterpret_cast<FaceDetect *>(algo))->cancelAddFace();
}

bool AlgoDispatcher::setReidObject(std::vector<int> bbox)
{
  AlgorithmBase * algo = m_algorithms[ALGO_BODY_DETECT];
  if (!algo) {
    return false;
  }

  (reinterpret_cast<BodyDetect *>(algo))->setReidEnabled(true);
  (reinterpret_cast<BodyDetect *>(algo))->setReidObject(bbox);
  return true;
}

bool AlgoDispatcher::isBodyTracked()
{
  AlgorithmBase * algo = m_algorithms[ALGO_BODY_DETECT];
  if (!algo) {
    return false;
  }

  return (reinterpret_cast<BodyDetect *>(algo))->isBodyTracked();
}

}  // namespace cyberdog_camera
