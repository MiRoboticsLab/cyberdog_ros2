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

#include <iostream>
#include <fstream>
#include <sstream>
#include <memory>
#include <string>
#include "xiaoai_sdk/aivs/Logger.h"
#include "xiaoai_sdk/aivs/TextUtils.h"
#include "xiaoai_sdk/aivs/StorageCapabilityImpl.h"

StorageCapabilityImpl::StorageCapabilityImpl()
{
/* #ifndef __NuttX__
  storageFilePath = "data.json";
#else
  storageFilePath = "/tmp/data.json";
#endif */
  storageFilePath = "/opt/ros2/cyberdog/data/token.json";
  loadKeyValuesFromFile();
}

bool StorageCapabilityImpl::writeKeyValue(std::string & key, std::string & value)
{
  std::unique_lock<std::mutex> lock(mMutex);

  mJsonRoot[key] = value;

  // 更新存储信息，以文件存储做示例
  saveKeyValuesToFile();

  return true;
}

bool StorageCapabilityImpl::readKeyValue(std::string & key, std::string & value)
{
  std::unique_lock<std::mutex> lock(mMutex);
  value = mJsonRoot[key].asString();
  if (!value.empty()) {
    return true;
  }

  return false;
}

bool StorageCapabilityImpl::removeKeyValue(std::string & key)
{
  std::unique_lock<std::mutex> lock(mMutex);
  mJsonRoot.removeMember(key);
  saveKeyValuesToFile();

  return true;
}

bool StorageCapabilityImpl::clearKeyValues()
{
  std::unique_lock<std::mutex> lock(mMutex);
  mJsonRoot.clear();
  saveKeyValuesToFile();

  return true;
}

void StorageCapabilityImpl::loadKeyValuesFromFile()
{
  // 从本地文件中读取token
  std::ifstream in(storageFilePath.c_str(), std::ios::in);
  if (!in) {
    std::cout << storageFilePath << " not exist" << std::endl;
    return;
  }

  std::string cachedJson;
  std::string line;
  while (getline(in, line)) {
    cachedJson += line;
  }

  in.close();

  if (!TextUtils::text2json(cachedJson, mJsonRoot)) {
    return;
  }
}

void StorageCapabilityImpl::saveKeyValuesToFile()
{
  std::string jsonData;
  if (!TextUtils::json2text(mJsonRoot, jsonData)) {
    return;
  }

  if (jsonData.empty()) {
    return;
  }

  std::ofstream out(storageFilePath.c_str(), std::ios::trunc | std::ios::out);
  out << jsonData << std::endl;
  out.close();
}

StorageCapabilityImpl::~StorageCapabilityImpl()
{
  LOGD("StorageCapabilityImpl", "~StorageCapabilityImpl", "");
}
