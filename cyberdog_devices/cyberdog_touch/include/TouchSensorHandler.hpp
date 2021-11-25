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

#ifndef TOUCHSENSORHANDLER_HPP_
#define TOUCHSENSORHANDLER_HPP_

#include <stdint.h>
#include <errno.h>
#include <sys/cdefs.h>
#include <sys/types.h>
#include <poll.h>

#include "InputEventReader.hpp"

#define NUM_TOUCH_EVENT_MAX 10

#define F51_DISCRETE_GESTURE_XM
#ifdef F51_DISCRETE_GESTURE_XM
#define GESTURE_XM_ADDR                              300
#define LPWG_SINGLETAP_DETECTED                      0x01
#define LPWG_DOUBLETAP_DETECTED                      0x03
#define LPWG_TOUCHANDHOLD_DETECTED                   0x07
#define LPWG_CIRCLE_DETECTED                         0x08
#define LPWG_TRIANGLE_DETECTED                       0x09
#define LPWG_VEE_DETECTED                            0x0A
#define LPWG_UNICODE_DETECTED                        0x0B
#define LPWG_SWIPE_DETECTED                          0x0D
#define LPWG_SWIPE_DETECTED_UP_CONTINU               0x0E
#define LPWG_SWIPE_DETECTED_DOWN_CONTINU             0x0F
#define LPWG_SWIPE_DETECTED_LEFT_CONTINU             0x10
#define LPWG_SWIPE_DETECTED_RIGHT_CONTINU            0x11

#define LPWG_SWIPE_FINGER_NUM_MASK                   0xF0
#define LPWG_SWIPE_FINGER_UP_DOWN_DIR_MASK           0x03
#define LPWG_SWIPE_FINGER_LEFT_RIGHT_DIR_MASK        0x0C
#define LPWG_SWIPE_ID_SINGLE                         0x40
#define LPWG_SWIPE_ID_DOUBLE                         0x80
#define LPWG_SWIPE_UP                                0x01
#define LPWG_SWIPE_DOWN                              0x02
#define LPWG_SWIPE_LEFT                              0x04
#define LPWG_SWIPE_RIGHT                             0x08
#endif

class TouchSensorHandler : public InputEventCircularReader
{
public:
  TouchSensorHandler();
  ~TouchSensorHandler();
  int openInput(void);
  int processEvents(input_event * data, int count);
  int pollTouchEvents(input_event * data, int count);
  int getFd() const;
  int data_fd;
  struct input_event mPendingEvent;
  struct pollfd mPollFd;
};

#endif  // TOUCHSENSORHANDLER_HPP_
