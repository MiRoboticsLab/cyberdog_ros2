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

#ifndef CAMERA_UTILS__LOG_HPP_
#define CAMERA_UTILS__LOG_HPP_

#include <rclcpp/rclcpp.hpp>

#ifndef LOG_TAG
#define LOG_TAG ""
#endif

#ifndef CAM_DEBUG
#define CAM_DEBUG(fmt, arg ...) do { \
    RCLCPP_DEBUG(rclcpp::get_logger(LOG_TAG), "%s: " fmt, __FUNCTION__, ## arg); \
} while (0)
#endif

#ifndef CAM_INFO
#define CAM_INFO(fmt, arg ...) do { \
    RCLCPP_INFO(rclcpp::get_logger(LOG_TAG), "%s: " fmt, __FUNCTION__, ## arg); \
} while (0)
#endif

#ifndef CAM_ERR
#define CAM_ERR(fmt, arg ...) do { \
    RCLCPP_ERROR(rclcpp::get_logger(LOG_TAG), "%s: " fmt, __FUNCTION__, ## arg); \
} while (0)
#endif

#endif  // CAMERA_UTILS__LOG_HPP_
