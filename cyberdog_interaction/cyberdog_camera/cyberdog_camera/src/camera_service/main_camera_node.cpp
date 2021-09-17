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

#define LOG_TAG "MainCameraNode"

#include <memory>
#include <string>
#include <map>
#include <vector>
#include "camera_service/main_camera_node.hpp"
#include "camera_service/camera_manager.hpp"
#include "camera_service/face_manager_node.hpp"
#include "camera_utils/log.hpp"
#include "camera_utils/utils.hpp"

#define ImageWidth 1280
#define ImageHeight 960
#define CAMERA_PATH "/home/mi/Camera/"

namespace cyberdog_camera
{

static const char * camera_cmd_strings[] = {
  "SET_PARAMETERS",
  "TAKE_PICTURE",
  "START_RECORDING",
  "STOP_RECORDING",
  "GET_STATE",
  "DELETE_FILE",
  "GET_ALL_FILES",
  "START_LIVE_STREAM",
  "STOP_LIVE_STREAM",
};

static const char * get_cmd_string(unsigned int index)
{
  if (index >= sizeof(camera_cmd_strings) / sizeof(camera_cmd_strings[0])) {
    return "UNSUPPORTED";
  }

  return camera_cmd_strings[index];
}

CameraServerNode::CameraServerNode()
: LifecycleNode("camera_server")
{
  CAM_INFO("Creating node %s.", get_name());

  initParameters();
  CameraManager::getInstance()->openCamera(0);
  CameraManager::getInstance()->setParent(this);
}

CameraServerNode::~CameraServerNode()
{
  CAM_INFO("Destroy node %s.", get_name());

  CameraManager::getInstance()->closeCamera();
}

void CameraServerNode::initParameters()
{
  width_ = this->declare_parameter("width", ImageWidth);
  height_ = this->declare_parameter("height", ImageHeight);
}

cyberdog_utils::CallbackReturn CameraServerNode::on_configure(const rclcpp_lifecycle::State &)
{
  CAM_INFO("%s Configuring", get_name());

  m_camService = create_service<CameraServiceT>(
    "camera_service",
    std::bind(
      &CameraServerNode::serviceCallback, this,
      std::placeholders::_1, std::placeholders::_2, std::placeholders::_3));

  m_faceManager = create_service<FaceManagerT>(
    "face_manager",
    std::bind(
      &cyberdog_camera::FaceManagerNode::serviceCallback,
      std::make_shared<cyberdog_camera::FaceManagerNode>(),
      std::placeholders::_1, std::placeholders::_2, std::placeholders::_3));

  CameraManager::getInstance()->initCamera();

  return cyberdog_utils::CallbackReturn::SUCCESS;
}

cyberdog_utils::CallbackReturn CameraServerNode::on_activate(const rclcpp_lifecycle::State &)
{
  CAM_INFO("%s Activating", get_name());

  CameraManager::getInstance()->startCamera();

  return CallbackReturn::SUCCESS;
}

cyberdog_utils::CallbackReturn CameraServerNode::on_deactivate(const rclcpp_lifecycle::State &)
{
  CAM_INFO("%s Deactivating", get_name());

  CameraManager::getInstance()->stopCamera();

  return CallbackReturn::SUCCESS;
}

cyberdog_utils::CallbackReturn CameraServerNode::on_cleanup(const rclcpp_lifecycle::State &)
{
  CAM_INFO("%s Cleaning up", get_name());

  CameraManager::getInstance()->deinitCamera();
  m_camService.reset();
  m_faceManager.reset();

  return CallbackReturn::SUCCESS;
}

cyberdog_utils::CallbackReturn CameraServerNode::on_shutdown(const rclcpp_lifecycle::State &)
{
  CAM_INFO("%s Shutting down", get_name());

  CameraManager::getInstance()->closeCamera();

  return CallbackReturn::SUCCESS;
}

void CameraServerNode::serviceCallback(
  const std::shared_ptr<rmw_request_id_t>/*request_header*/,
  const std::shared_ptr<CameraServiceT::Request> request,
  std::shared_ptr<CameraServiceT::Response> response)
{
  CAM_INFO(
    "service received command %s, argument %s\n",
    get_cmd_string(request->command), request->args.c_str());

  switch (request->command) {
    case CameraServiceT::Request::SET_PARAMETERS:
      response->result =
        CameraManager::getInstance()->setParameters(request->args);
      break;
    case CameraServiceT::Request::TAKE_PICTURE:
      response->result = takePicture(request->args, response);
      break;
    case CameraServiceT::Request::START_RECORDING:
      response->result = startRecording(request->args, response);
      break;
    case CameraServiceT::Request::STOP_RECORDING:
      response->result = stopRecording(response);
      break;
    case CameraServiceT::Request::GET_STATE:
      response->result = getState(response);
      break;
    case CameraServiceT::Request::DELETE_FILE:
      response->result = deleteFile(request->args);
      break;
    case CameraServiceT::Request::GET_ALL_FILES:
      response->result = getAllFiles(response);
      break;
    case CameraServiceT::Request::START_LIVE_STREAM:
      response->result =
        CameraManager::getInstance()->startPreview(request->args);
      break;
    case CameraServiceT::Request::STOP_LIVE_STREAM:
      response->result =
        CameraManager::getInstance()->stopPreview();
      break;
    default:
      printf("service unsupport command %d\n", request->command);
      response->result = CameraServiceT::Response::RESULT_INVALID_ARGS;
  }
}

int CameraServerNode::takePicture(
  std::string & args,
  std::shared_ptr<CameraServiceT::Response> response)
{
  int ret = CAM_SUCCESS;
  std::string filename;
  size_t size;
  int width = 0, height = 0;

  std::map<std::string, std::string> params_map = parse_parameters(args);
  std::map<std::string, std::string>::iterator it;
  for (it = params_map.begin(); it != params_map.end(); it++) {
    if (it->first == "width") {
      width = atoi(it->second.c_str());
    }
    if (it->first == "height") {
      height = atoi(it->second.c_str());
    }
  }

  CAM_INFO("width = %d, height = %d", width, height);
  ret = CameraManager::getInstance()->takePicture(filename, width, height);
  if (!ret) {
    size = get_file_size(CAMERA_PATH + filename);
    response->msg = filename + "," + std::to_string(size);
  }
  CAM_INFO("response: %s", response->msg.c_str());

  return ret;
}

int CameraServerNode::startRecording(
  std::string &,
  std::shared_ptr<CameraServiceT::Response> response)
{
  int ret = CAM_SUCCESS;
  std::string filename;
  int width = 0, height = 0;

  ret = CameraManager::getInstance()->startRecording(filename, width, height);
  if (!ret) {
    response->msg = filename + "," + std::to_string(0);
  }

  return ret;
}

int CameraServerNode::stopRecording(std::shared_ptr<CameraServiceT::Response> response)
{
  int ret = CAM_SUCCESS;
  std::string filename;
  size_t size = 0;

  ret = CameraManager::getInstance()->stopRecording(filename);
  if (!ret) {
    size = get_file_size(CAMERA_PATH + filename);
    response->msg = filename + "," + std::to_string(size);
  }
  CAM_INFO("response: %s", response->msg.c_str());

  return ret;
}

int CameraServerNode::getState(std::shared_ptr<CameraServiceT::Response> response)
{
  if (CameraManager::getInstance()->isRecording()) {
    response->msg = "recording,";
    response->msg += std::to_string(CameraManager::getInstance()->getRecordingTime());
  }
  CAM_INFO("response: %s", response->msg.c_str());

  return CAM_SUCCESS;
}

int CameraServerNode::deleteFile(const std::string & filename)
{
  std::string path = CAMERA_PATH + filename;
  return remove_file(path);
}

int CameraServerNode::getAllFiles(std::shared_ptr<CameraServiceT::Response> response)
{
  std::string result;
  std::vector<cv::String> filenames;

  if (access("/home/mi/Camera/", 0) != 0) {
    CAM_INFO("Path not existed.\n");
    return CAM_SUCCESS;
  }

  cv::glob("/home/mi/Camera/", filenames);
  for (unsigned int i = 0; i < filenames.size(); i++) {
    std::string filename = filenames[i].substr(filenames[i].find_last_of("/") + 1);
    result += filename;
    result += ",";
    result += std::to_string(get_file_size(filenames[i]));
    result += ";";
  }
  response->msg = result;

  return CAM_SUCCESS;
}

}  // namespace cyberdog_camera
