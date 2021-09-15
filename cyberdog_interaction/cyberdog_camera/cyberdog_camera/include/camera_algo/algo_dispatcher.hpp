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

#ifndef CAMERA_ALGO__ALGO_DISPATCHER_HPP_
#define CAMERA_ALGO__ALGO_DISPATCHER_HPP_

#include <string>
#include <vector>
#include "./algorithm_base.hpp"

namespace cyberdog_camera
{

enum
{
  ALGO_FACE_DETECT = 0,
  ALGO_BODY_DETECT,
  ALGO_TYPE_NONE,
};

class AlgoDispatcher
{
public:
  static AlgoDispatcher & getInstance();
  void processImageBuffer(uint64_t frame_id, ImageBuffer buffer);
  void setAlgoEnabled(int index, bool enabled);
  bool getAlgoEnabled(int index);
  bool needProcess(uint64_t frame_id);

  void setBufferDoneCallback(void (* callback)(ImageBuffer, void *), void * arg)
  {
    m_doneCb = callback;
    m_callbackArg = arg;
  }

  int startAddingFace(const std::string & name, bool is_host);
  int stopAddingFace();
  bool setReidObject(std::vector<int> bbox);
  bool isBodyTracked();

private:
  AlgoDispatcher();
  ~AlgoDispatcher();

  std::vector<AlgorithmBase *> m_algorithms;
  std::mutex m_mutex;
  bool m_bodyEnable;
  bool m_faceEnable;
  void (* m_doneCb)(ImageBuffer, void *);
  void * m_callbackArg;
};

}  // namespace cyberdog_camera

#endif  // CAMERA_ALGO__ALGO_DISPATCHER_HPP_
