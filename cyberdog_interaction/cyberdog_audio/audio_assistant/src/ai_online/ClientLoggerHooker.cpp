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

#include <thread>
#include <iostream>

#include "sys/time.h"
#include "xiaoai_sdk/aivs/ClientLoggerHooker.h"

namespace aivs
{

ClientLoggerHooker::ClientLoggerHooker()
{
  time_t now = time(NULL);
  struct tm ptm;
  localtime_r(&now, &ptm);

  char logFile[64] = {0};
  strftime(logFile, 64, "aivs-%Y-%m-%d-%H-%M-%S.log", &ptm);
  logger.open(logFile, std::ios::out | std::ios::app);
}

ClientLoggerHooker::~ClientLoggerHooker()
{
  logger.close();
}

void ClientLoggerHooker::i(char * message)
{
  std::unique_lock<std::mutex> lock(mMutex);
  logger << message << std::endl;
}

void ClientLoggerHooker::d(char * message)
{
  std::unique_lock<std::mutex> lock(mMutex);
  logger << message << std::endl;
}

void ClientLoggerHooker::w(char * message)
{
  std::unique_lock<std::mutex> lock(mMutex);
  logger << message << std::endl;
}

void ClientLoggerHooker::e(char * message)
{
  std::unique_lock<std::mutex> lock(mMutex);
  logger << message << std::endl;
}

}  // namespace aivs
