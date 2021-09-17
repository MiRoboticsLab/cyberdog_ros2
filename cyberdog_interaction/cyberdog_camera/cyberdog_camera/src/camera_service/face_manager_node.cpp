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

#define LOG_TAG "FaceManagerNode"
#include <unistd.h>
#include <memory>
#include <string>
#include <map>
#include <vector>
#include "camera_service/face_manager_node.hpp"
#include "camera_service/face_manager.hpp"
#include "camera_utils/log.hpp"

namespace cyberdog_camera
{
using namespace std::placeholders;

static const char * face_cmd_strings[] = {
  "ADD_FACE",
  "CANCLE_ADD_FACE",
  "CONFIRM_LAST_FACE",
  "UPDATE_FACE_ID",
  "DELETE_FACE",
  "GET_ALL_FACES",
};

static const char * get_cmd_string(unsigned int index)
{
  if (index >= sizeof(face_cmd_strings) / sizeof(face_cmd_strings[0])) {
    return "UNSUPPORTED";
  }

  return face_cmd_strings[index];
}

FaceManagerNode::FaceManagerNode()
{
}

FaceManagerNode::~FaceManagerNode()
{
}

void FaceManagerNode::serviceCallback(
  const std::shared_ptr<rmw_request_id_t>,
  const std::shared_ptr<FaceManagerT::Request> request,
  std::shared_ptr<FaceManagerT::Response> response)
{
  TimePerf perf(get_cmd_string(request->command));
  CAM_INFO(
    "face service received command %s, argument '%s'",
    get_cmd_string(request->command), request->args.c_str());

  switch (request->command) {
    case FaceManagerT::Request::ADD_FACE:
      response->result = addFaceInfo(request->args);
      break;
    case FaceManagerT::Request::CANCLE_ADD_FACE:
      response->result = cancelAddFace(request->args);
      break;
    case FaceManagerT::Request::CONFIRM_LAST_FACE:
      response->result = confirmLastFace();
      break;
    case FaceManagerT::Request::UPDATE_FACE_ID:
      response->result = updateFaceId(request->args);
      break;
    case FaceManagerT::Request::DELETE_FACE:
      response->result = deleteFace(request->args);
      break;
    case FaceManagerT::Request::GET_ALL_FACES:
      response->result = getAllFaces(response);
      break;
    default:
      CAM_ERR("service unsupport command %d", request->command);
      response->result = FaceManagerT::Response::RESULT_INVALID_ARGS;
  }
}

int FaceManagerNode::addFaceInfo(std::string & args)
{
  int ret = CAM_SUCCESS;
  std::string name;
  bool is_host = false;

  // parse command arguments
  std::map<std::string, std::string> params_map = parse_parameters(args);
  std::map<std::string, std::string>::iterator it;
  for (it = params_map.begin(); it != params_map.end(); it++) {
    if (it->first == "id") {
      name = it->second;
    }
    if (it->first == "host" && it->second == "true") {
      is_host = true;
    }
  }

  if (name.length() == 0) {
    return CAM_INVALID_ARG;
  }

  if (FaceManager::getInstance()->findFace(name)) {
    CAM_INFO("Face %s is already existed.", name.c_str());
  }

  ret = cyberdog_camera::FaceManager::getInstance()->addFaceInfo(name, is_host);

  return ret;
}

int FaceManagerNode::cancelAddFace(const std::string & args)
{
  return cyberdog_camera::FaceManager::getInstance()->cancelAddFace(args);
}

int FaceManagerNode::confirmLastFace()
{
  return cyberdog_camera::FaceManager::getInstance()->confirmFace();
}

int FaceManagerNode::updateFaceId(std::string & args)
{
  std::vector<std::string> names;

  split_string(args, names, ":");
  if (names.size() != 2) {
    return CAM_INVALID_ARG;
  }

  std::string ori_name = names[0];
  std::string new_name = names[1];
  CAM_INFO("change %s to %s", ori_name.c_str(), new_name.c_str());

  return cyberdog_camera::FaceManager::getInstance()->updateFaceId(ori_name, new_name);
}

int FaceManagerNode::deleteFace(std::string & face_name)
{
  if (face_name.length() == 0) {
    return CAM_INVALID_ARG;
  }

  return cyberdog_camera::FaceManager::getInstance()->deleteFace(face_name);
}

int FaceManagerNode::getAllFaces(std::shared_ptr<FaceManagerT::Response> response)
{
  std::vector<cv::String> filenames;

  std::string path = FaceManager::getFaceDataPath();
  if (access(path.c_str(), 0) != 0) {
    CAM_ERR("Face path '%s' not exist", path.c_str());
    return CAM_SUCCESS;
  }
  cv::glob(path + "*.data", filenames);

  CAM_INFO("found %u faces", filenames.size());
  response->face_images.resize(filenames.size());
  for (unsigned int i = 0; i < filenames.size(); i++) {
    int face_name_len = filenames[i].find_last_of(".") - filenames[i].find_last_of("/") - 1;
    std::string face_name = filenames[i].substr(filenames[i].find_last_of("/") + 1, face_name_len);

    if (cyberdog_camera::FaceManager::getInstance()->isHost(face_name)) {
      response->msg = "host=" + face_name;
    }

    response->face_images[i].header.frame_id = face_name;
  }

  return CAM_SUCCESS;
}

}  // namespace cyberdog_camera
