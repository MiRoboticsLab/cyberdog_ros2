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

#ifndef CAMERA_SERVICE__ROS2_SERVICE_HPP_
#define CAMERA_SERVICE__ROS2_SERVICE_HPP_

/*
 * Camera ros2 service interfaces
 */

#include "interaction_msgs/srv/camera_service.hpp"
#include "interaction_msgs/srv/face_manager.hpp"
#include "interaction_msgs/msg/body_info.hpp"
#include "interaction_msgs/msg/face_info.hpp"
#include "interaction_msgs/msg/face_result.hpp"
#include "interaction_msgs/action/audio_play.hpp"
#include "ception_msgs/srv/sensor_detection_node.hpp"

using CameraServiceT = interaction_msgs::srv::CameraService;
using FaceManagerT = interaction_msgs::srv::FaceManager;
using BodyInfoT = interaction_msgs::msg::BodyInfo;
using BodyT = interaction_msgs::msg::Body;
using FaceInfoT = interaction_msgs::msg::FaceInfo;
using FaceT = interaction_msgs::msg::Face;
using FaceResultT = interaction_msgs::msg::FaceResult;
using AudioPlayT = interaction_msgs::action::AudioPlay;
using LedServiceT = ception_msgs::srv::SensorDetectionNode;

#endif  // CAMERA_SERVICE__ROS2_SERVICE_HPP_
