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

#define LOG_TAG "CameraManager"
#include <stdio.h>
#include <sys/stat.h>
#include <sstream>
#include <string>
#include <map>
#include <vector>
#include "camera_service/camera_manager.hpp"
#include "camera_algo/algo_dispatcher.hpp"
#include "camera_utils/utils.hpp"
#include "camera_utils/log.hpp"

#define CAMERA_PATH "/home/mi/Camera/"

namespace cyberdog_camera
{

CameraManager * CameraManager::getInstance()
{
  static CameraManager s_instance;

  return &s_instance;
}

CameraManager::CameraManager()
: m_camera(NULL)
{
}

CameraManager::~CameraManager()
{
}

bool CameraManager::openCamera(int camera_id)
{
  if (m_camera != NULL) {
    CAM_INFO("camera %d already opened\n", camera_id);
    return true;
  }

  m_camera = new ArgusCameraContext(camera_id);

  return true;
}

bool CameraManager::closeCamera()
{
  deinitCamera();
  if (m_camera) {
    delete m_camera;
    m_camera = NULL;
  }

  return true;
}

bool CameraManager::initCamera()
{
  m_camera->createSession();

  return true;
}

void CameraManager::deinitCamera()
{
  m_camera->closeSession();
}

bool CameraManager::startCamera()
{
  m_camera->initCameraContext();

  return true;
}

bool CameraManager::stopCamera()
{
  m_camera->deinitCameraContext();

  return true;
}

int CameraManager::startPreview(std::string & usage)
{
  return m_camera->startPreview(usage);
}

int CameraManager::stopPreview()
{
  return m_camera->stopPreview();
}

int CameraManager::startRecording(std::string & filename, int width, int height)
{
  if (access(CAMERA_PATH, 0) != 0) {
    umask(0);
    mkdir(CAMERA_PATH, 0755);
  }

  filename = get_time_string() + ".mp4";
  return m_camera->startRecording(filename, width, height);
}

int CameraManager::stopRecording(std::string & filename)
{
  return m_camera->stopRecording(filename);
}

bool CameraManager::isRecording()
{
  return m_camera->isRecording();
}

uint64_t CameraManager::getRecordingTime()
{
  return m_camera->getRecordingTime();
}

int CameraManager::takePicture(std::string & filename, int width, int height)
{
  /* create save directory if not exist.*/
  if (access(CAMERA_PATH, 0) != 0) {
    umask(0);
    mkdir(CAMERA_PATH, 0755);
  }

  filename = get_time_string();
  if (filename == m_lastPictureName) {
    filename = filename + "_" + std::to_string(++m_pictureIndex);
  } else {
    m_lastPictureName = filename;
    m_pictureIndex = 0;
  }

  filename += ".jpg";
  std::string path = CAMERA_PATH + filename;

  return m_camera->takePicture(path.c_str(), width, height);
}

int CameraManager::setParameters(std::string & parameters)
{
  std::map<std::string, std::string> params_map = parse_parameters(parameters);
  std::map<std::string, std::string>::iterator it;
  for (it = params_map.begin(); it != params_map.end(); it++) {
    setParameter(it->first, it->second);
  }

  return CAM_SUCCESS;
}

int CameraManager::setParameter(const std::string & key, const std::string & value)
{
  if (key == "face-interval") {
    printf("service: set face detect interval\n");
    if (atoi(value.c_str()) > 0) {
      setVisionAlgoEnabled(ALGO_FACE_DETECT, true);
    } else {
      setVisionAlgoEnabled(ALGO_FACE_DETECT, false);
    }
  }

  if (key == "body-interval") {
    printf("service: set body detect interval\n");
    if (atoi(value.c_str()) > 0) {
      setVisionAlgoEnabled(ALGO_BODY_DETECT, true);
    } else {
      setVisionAlgoEnabled(ALGO_BODY_DETECT, false);
    }
  }

  if (key == "reid-bbox") {
    printf("service: set reid bbox\n");
    std::vector<int> bbox;
    std::stringstream input(value);
    std::string tmp;
    while (getline(input, tmp, ',')) {
      bbox.push_back(atoi(tmp.c_str()));
    }
    if (4 != bbox.size()) {
      return CAM_INVALID_ARG;
    } else {
      if (!AlgoDispatcher::getInstance().setReidObject(bbox)) {
        return CAM_INVALID_STATE;
      }
    }
  }

  return CAM_SUCCESS;
}

void CameraManager::setVisionAlgoEnabled(int algo_type, bool enable)
{
  CAM_INFO("algo %d : %d", algo_type, enable);
  if (enable) {
    m_camera->startRgbStream();
    AlgoDispatcher::getInstance().setAlgoEnabled(algo_type, true);
  } else {
    AlgoDispatcher::getInstance().setAlgoEnabled(algo_type, false);
  }

  processAlgoParam();
}

void CameraManager::processAlgoParam()
{
  bool algo_on = false;

  for (int i = 0; i < ALGO_TYPE_NONE; i++) {
    algo_on |= AlgoDispatcher::getInstance().getAlgoEnabled(i);
  }

  if (!algo_on) {
    m_camera->stopRgbStream();
  }
}

}  // namespace cyberdog_camera
