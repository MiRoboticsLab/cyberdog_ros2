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

#ifndef CAMERA_SERVICE__CAMERA_MANAGER_HPP_
#define CAMERA_SERVICE__CAMERA_MANAGER_HPP_

#include <rclcpp_action/rclcpp_action.hpp>
#include <cyberdog_utils/lifecycle_node.hpp>
#include <string>
#include <memory>
#include "./argus_camera_context.hpp"

namespace cyberdog_camera
{

class CameraManager
{
public:
  static CameraManager * getInstance();
  bool openCamera(int camera_id);
  bool closeCamera();
  bool initCamera();
  void deinitCamera();
  bool startCamera();
  bool stopCamera();

  int startPreview(std::string & usage);
  int stopPreview();
  int startRecording(std::string & filename, int width = 0, int height = 0);
  int stopRecording(std::string & filename);
  bool isRecording();
  uint64_t getRecordingTime();
  int takePicture(std::string & filename, int width = 0, int height = 0);
  int setParameters(std::string & parameters);
  void setVisionAlgoEnabled(int algo_type, bool enable);

  void setParent(cyberdog_utils::LifecycleNode * parent)
  {
    m_parentNode = parent;
  }

  template<typename MessageT>
  std::shared_ptr<rclcpp::Publisher<MessageT>> create_publisher(
    const std::string & topic_name,
    const rclcpp::QoS & qos)
  {
    std::shared_ptr<rclcpp_lifecycle::LifecyclePublisher<MessageT>> pub;

    pub = m_parentNode->create_publisher<MessageT>(topic_name, qos);
    pub->on_activate();

    return pub;
  }

  template<typename MessageT, typename CallbackT>
  std::shared_ptr<rclcpp::Subscription<MessageT>> create_subscription(
    const std::string & topic_name,
    const rclcpp::QoS & qos,
    CallbackT && callback)
  {
    return m_parentNode->create_subscription<MessageT>(topic_name, qos, callback);
  }

  template<typename ActionT>
  typename rclcpp_action::Client<ActionT>::SharedPtr create_action_client(
    const std::string & name)
  {
    return rclcpp_action::create_client<ActionT>(m_parentNode, name);
  }

  template<typename ServiceT>
  typename rclcpp::Client<ServiceT>::SharedPtr create_service_client(const std::string & name)
  {
    return m_parentNode->create_client<ServiceT>(name);
  }

private:
  CameraManager();
  ~CameraManager();
  int setParameter(const std::string & key, const std::string & value);
  void processAlgoParam();

  ArgusCameraContext * m_camera;
  cyberdog_utils::LifecycleNode * m_parentNode;
  std::string m_lastPictureName;
  uint32_t m_pictureIndex;
};

}  // namespace cyberdog_camera

#endif  // CAMERA_SERVICE__CAMERA_MANAGER_HPP_
