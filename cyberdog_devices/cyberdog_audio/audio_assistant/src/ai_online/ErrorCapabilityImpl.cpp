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
#include <iostream>
#include "xiaoai_sdk/aivs/StdStatuses.h"
#include "xiaoai_sdk/aivs/ErrorCapabilityImpl.h"

void ErrorCapabilityImpl::onError(std::shared_ptr<AivsError> & error)
{
  auto eventId = error->getEventId();
  switch (error->getErrorCode()) {
    case StdStatuses::ASR_TIME_OUT:
      std::cout << "[ERROR]ASR_TIME_OUT, eventId=" <<
      (eventId.has_value() ? *eventId : "unknown") <<
        ", msg=" << error->getErrorMessage() <<
        std::endl;
      break;
    case StdStatuses::TTS_TIME_OUT:
      std::cout << "[ERROR]TTS_TIME_OUT, eventId=" <<
      (eventId.has_value() ? *eventId : "unknown") <<
        ", msg=" << error->getErrorMessage() <<
        std::endl;
      break;
    case StdStatuses::CONNECT_FAILED:
      std::cout << "[ERROR]CONNECT_FAILED:" <<
        "msg=" << error->getErrorMessage() <<
        std::endl;
      break;
    case StdStatuses::UNAUTHORIZED:
      std::cout << "[ERROR]UNAUTHORIZED：:" <<
        "msg=" << error->getErrorMessage() <<
        std::endl;
      break;
    default:
      std::cout << "[ERROR]onError, code=" << error->getErrorCode() <<
        ", msg=" << error->getErrorMessage() <<
        std::endl;
      break;
  }
}
