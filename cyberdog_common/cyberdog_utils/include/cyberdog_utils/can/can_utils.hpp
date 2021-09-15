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

#ifndef CYBERDOG_UTILS__CAN__CAN_UTILS_HPP_
#define CYBERDOG_UTILS__CAN__CAN_UTILS_HPP_

#include <ctype.h>
#include <errno.h>
#include <libgen.h>
#include <signal.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#include <chrono>
#include <memory>
#include <string>

#include "net/if.h"
#include "sys/ioctl.h"
#include "sys/select.h"
#include "sys/socket.h"
#include "sys/time.h"
#include "sys/types.h"
#include "sys/uio.h"
#include "sys/unistd.h"

#include "linux/can.h"
#include "linux/can/raw.h"
#include "linux/can/error.h"

#include "cyberdog_utils/can/can_proto.h"

#include "cyberdog_utils/can/socket_can_receiver.hpp"
#include "cyberdog_utils/can/socket_can_sender.hpp"

#ifndef ARRAY_SIZE
#define ARRAY_SIZE(arr) (sizeof(arr) / sizeof((arr)[0]))
#endif

namespace cyberdog_utils
{

class can_dev_operation
{
public:
  can_dev_operation();
  ~can_dev_operation();

  int wait_for_can_data();
  int send_can_message(struct can_frame cmd_frame);

  struct can_frame recv_frame;

  std::unique_ptr<drivers::socketcan::SocketCanReceiver> receiver_;
  std::unique_ptr<drivers::socketcan::SocketCanSender> sender_;

private:
  std::string interface_;
};
}  // namespace cyberdog_utils

#endif  // CYBERDOG_UTILS__CAN__CAN_UTILS_HPP_
