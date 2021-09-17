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

#define LOG_TAG "FaceManager"

#include <sys/stat.h>
#include <unistd.h>
#include <opencv2/opencv.hpp>
#include <string>
#include <map>
#include <vector>
#include <algorithm>
#include <memory>
#include <utility>
#include "camera_service/face_manager.hpp"
#include "camera_service/camera_manager.hpp"
#include "camera_algo/algo_dispatcher.hpp"
#include "camera_service/ncs_client.hpp"
#include "camera_utils/log.hpp"

namespace cyberdog_camera
{

static const char * label_path = "/home/mi/.faces/";

FaceManager * FaceManager::getInstance()
{
  static FaceManager s_instance;

  return &s_instance;
}

FaceManager::FaceManager()
{
  m_inFaceAdding = false;
  m_faceCacheSize = 0;
  initialize();
}

FaceManager::~FaceManager()
{
}

const std::string FaceManager::getFaceDataPath()
{
  return label_path;
}

void FaceManager::initialize()
{
  if (!loadFeatures()) {
    CAM_ERR("Failed to load face features.");
  }

  rclcpp::ServicesQoS qos;
  m_resultPub = CameraManager::getInstance()->create_publisher<FaceResultT>(
    "/face_manager/result",
    qos);
}

std::map<std::string, std::vector<float>> & FaceManager::getFeatures()
{
  std::lock_guard<std::mutex> lock(m_mutex);

  return m_features;
}

bool FaceManager::loadFeatures()
{
  std::lock_guard<std::mutex> lock(m_mutex);
  std::vector<cv::String> filenames;

  if (access(label_path, 0) != 0) {
    printf("faces path not found.\n");
    return false;
  }
  cv::glob(std::string(label_path) + "*.data", filenames);

  for (unsigned int i = 0; i < filenames.size(); i++) {
    int face_name_len = filenames[i].find_last_of(".") - filenames[i].find_last_of("/") - 1;
    std::string face_name = filenames[i].substr(filenames[i].find_last_of("/") + 1, face_name_len);

    size_t feats_size;
    FILE * fp = fopen(filenames[i].c_str(), "rb");
    fread(&feats_size, sizeof(feats_size), 1, fp);
    if (feats_size == 0) {
      continue;
    }

    float * feats_data = new float[feats_size];
    bool is_host;
    fread(feats_data, sizeof(float), feats_size, fp);
    fread(&is_host, sizeof(is_host), 1, fp);

    std::vector<float> feats(feats_size);
    std::copy(feats_data, feats_data + feats_size, feats.begin());
    m_features[face_name] = feats;
    m_hostMap[face_name] = is_host;
    CAM_INFO("Load known face info '%s', host %d", face_name.c_str(), is_host);

    fclose(fp);
    delete[] feats_data;
  }

  return true;
}

bool FaceManager::addFaceData(
  const FaceId & face_id,
  std::vector<float> & feats, void * img, size_t size)
{
  m_faceCacheSize = size;
  m_faceFeatsCached = feats;
  m_faceIdCached = face_id;
  memcpy(m_faceImageCached, img, size);

  publishResult(CAM_SUCCESS, face_id.name, img, size);
  if (pthread_create(&m_thread, NULL, stopThreadFunc, this) != 0) {
    CAM_ERR("Failed to create stop thread.");
    return false;
  }

  m_inFaceAdding = false;

  return true;
}

void FaceManager::publishResult(int result, const std::string & name, void * img, size_t size)
{
  CAM_INFO("publish face input result %d.", result);
  auto msg = std::make_unique<FaceResultT>();

  msg->result = result;
  msg->msg = name;
  if (img != NULL) {
    msg->face_images.resize(1);
    msg->face_images[0].header.frame_id = name;
    msg->face_images[0].format = "jpeg";
    msg->face_images[0].data.resize(size);
    memcpy(&(msg->face_images[0].data[0]), img, size);
  }

  m_resultPub->publish(std::move(msg));
}

int FaceManager::stopFaceDetect()
{
  int ret = CAM_SUCCESS;

  ret = AlgoDispatcher::getInstance().stopAddingFace();
  if (!m_faceDetectEnabled) {
    CameraManager::getInstance()->setVisionAlgoEnabled(ALGO_FACE_DETECT, false);
  }

  return ret;
}

void * FaceManager::stopThreadFunc(void * _this)
{
  FaceManager * manager = static_cast<FaceManager *>(_this);

  manager->stopFaceDetect();

  return NULL;
}

int FaceManager::addFaceInfo(const std::string & name, bool is_host)
{
  m_faceDetectEnabled = AlgoDispatcher::getInstance().getAlgoEnabled(ALGO_FACE_DETECT);
  if (!m_faceDetectEnabled) {
    CameraManager::getInstance()->setVisionAlgoEnabled(ALGO_FACE_DETECT, true);
  }

  m_inFaceAdding = true;

  return AlgoDispatcher::getInstance().startAddingFace(name, is_host);
}

int FaceManager::cancelAddFace(const std::string & args)
{
  if (m_inFaceAdding && args == "timeout") {
    NCSClient::getInstance().play(SoundFaceAddFailed);
    publishResult(CAM_TIMEOUT);
  }
  m_inFaceAdding = false;

  return stopFaceDetect();
}

int FaceManager::confirmFace()
{
  std::string filename;
  FILE * fp;

  CAM_INFO("confirm last face %s:%d", m_faceIdCached.name.c_str(), m_faceIdCached.is_host);
  if (m_faceCacheSize == 0) {
    CAM_ERR("No cached face info.");
    return CAM_INVALID_STATE;
  }

  m_mutex.lock();
  m_features[m_faceIdCached.name] = m_faceFeatsCached;
  m_hostMap[m_faceIdCached.name] = m_faceIdCached.is_host;
  m_mutex.unlock();

  /* save face features */
  if (access(label_path, 0) != 0) {
    umask(0);
    mkdir(label_path, 0755);
  }

  /*
   * face data layout
   * 1. features size     - unsigned long
   * 2. features          - float array
   * 3. host flag         - bool
   */
  size_t feats_size = m_faceFeatsCached.size();
  float * feats_array = new float[feats_size];
  std::copy(m_faceFeatsCached.begin(), m_faceFeatsCached.end(), feats_array);
  filename = std::string(label_path) + m_faceIdCached.name + ".data";
  fp = fopen(filename.c_str(), "wb+");
  fwrite(&feats_size, sizeof(feats_size), 1, fp);
  fwrite(feats_array, sizeof(float), feats_size, fp);
  fwrite(&m_faceIdCached.is_host, sizeof(m_faceIdCached.is_host), 1, fp);
  fclose(fp);
  delete[] feats_array;

  /* clear face cache */
  m_faceFeatsCached.clear();
  m_faceIdCached.name = "";
  m_faceIdCached.is_host = false;
  m_faceCacheSize = 0;

  return CAM_SUCCESS;
}

int FaceManager::updateFaceId(std::string & ori_name, std::string & new_name)
{
  int ret = CAM_SUCCESS;

  if (m_features.find(ori_name) == m_features.end()) {
    CAM_ERR("Face name %s not found", ori_name.c_str());
    return CAM_INVALID_ARG;
  }

  m_mutex.lock();
  m_features[new_name] = m_features[ori_name];
  m_features.erase(ori_name);
  m_hostMap[new_name] = m_hostMap[ori_name];
  m_hostMap.erase(ori_name);
  m_mutex.unlock();

  std::string ori_filename = std::string(label_path) + ori_name + ".data";
  std::string new_filename = std::string(label_path) + new_name + ".data";
  ret = rename_file(ori_filename, new_filename);
  if (ret != 0) {
    return ret;
  }

  return ret;
}

int FaceManager::deleteFace(std::string & face_name)
{
  int ret = CAM_SUCCESS;

  if (m_features.find(face_name) == m_features.end()) {
    CAM_ERR("Face name %s not found", face_name.c_str());
    return CAM_INVALID_ARG;
  }

  m_mutex.lock();
  m_features.erase(face_name);
  m_hostMap.erase(face_name);
  m_mutex.unlock();

  std::string filename = std::string(label_path) + face_name + ".data";
  ret = remove_file(filename);
  if (ret != 0) {
    return ret;
  }

  return ret;
}

bool FaceManager::findFace(const std::string & face_name)
{
  return m_features.find(face_name) != m_features.end();
}

bool FaceManager::isHost(const std::string & face_name)
{
  if (m_hostMap.find(face_name) != m_hostMap.end()) {
    return m_hostMap[face_name];
  }

  return false;
}

}  // namespace cyberdog_camera
