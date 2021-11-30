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

#ifndef CAMERA_SERVICE__FACE_MANAGER_NODE_HPP_
#define CAMERA_SERVICE__FACE_MANAGER_NODE_HPP_

#include <rclcpp/rclcpp.hpp>
#include <string>
#include <memory>
#include "./ros2_service.hpp"

namespace cyberdog_camera
{

class FaceManagerNode
{
public:
  FaceManagerNode();
  ~FaceManagerNode();
  void serviceCallback(
    const std::shared_ptr<rmw_request_id_t>,
    const std::shared_ptr<FaceManagerT::Request> request,
    std::shared_ptr<FaceManagerT::Response> response);

private:
  int addFaceInfo(std::string & args);
  int cancelAddFace(const std::string & args);
  int confirmLastFace();
  int updateFaceId(std::string & args);
  int deleteFace(std::string & face_name);
  int getAllFaces(std::shared_ptr<FaceManagerT::Response> response);

  rclcpp::Service<FaceManagerT>::SharedPtr m_faceManager;
};

}  // namespace cyberdog_camera

#endif  // CAMERA_SERVICE__FACE_MANAGER_NODE_HPP_
