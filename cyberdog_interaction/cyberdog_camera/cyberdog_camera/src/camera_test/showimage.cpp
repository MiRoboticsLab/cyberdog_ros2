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

#include <iostream>
#include <string>
#include <vector>
#include <mutex>
#include <memory>

#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "rcl_interfaces/msg/parameter_descriptor.hpp"
#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/image.hpp"
#include "camera_service/ros2_service.hpp"

struct RectId
{
  cv::Rect rect;
  std::string id;
};

class ShowImage : public rclcpp::Node
{
public:
  ShowImage()
  : Node("showimage")
  {
    setvbuf(stdout, NULL, _IONBF, BUFSIZ);
    parse_parameters();
    initialize();
  }

private:
  void initialize()
  {
    // Set quality of service profile based on command line options.
    auto qos = rclcpp::QoS(
      rclcpp::QoSInitialization(
        history_policy_,
        depth_
    ));

    qos.reliability(reliability_policy_);
    auto callback = [this](const sensor_msgs::msg::Image::SharedPtr msg)
      {
        process_image(msg, show_image_);
      };

    RCLCPP_INFO(this->get_logger(), "Subscribing to topic '%s'", topic_.c_str());
    sub_ = create_subscription<sensor_msgs::msg::Image>(topic_, qos, callback);

    auto body_cb = [this](const BodyInfoT::SharedPtr msg)
      {
        process_body_msg(msg);
      };
    auto qos_body = rclcpp::SensorDataQoS();
    RCLCPP_INFO(this->get_logger(), "Subscribing to topic 'body'");
    sub_body = create_subscription<BodyInfoT>("body", qos_body, body_cb);

    auto face_cb = [this](const FaceInfoT::SharedPtr msg)
      {
        process_face_msg(msg);
      };
    auto qos_face = rclcpp::SensorDataQoS();
    RCLCPP_INFO(this->get_logger(), "Subscribing to topic 'face'");
    sub_face = create_subscription<FaceInfoT>("face", qos_face, face_cb);

    if (window_name_ == "") {
      window_name_ = sub_->get_topic_name();
    }
  }

  void parse_parameters()
  {
    // Parse 'reliability' parameter
    rcl_interfaces::msg::ParameterDescriptor reliability_desc;
    reliability_desc.description = "Reliability QoS setting for the image subscription";
    reliability_desc.additional_constraints = "Must be one of: ";
    const std::string reliability_param = this->declare_parameter(
      "reliability", "reliable", reliability_desc);
    reliability_policy_ = RMW_QOS_POLICY_RELIABILITY_RELIABLE;

    // Parse 'history' parameter
    rcl_interfaces::msg::ParameterDescriptor history_desc;
    history_desc.description = "History QoS setting for the image subscription";
    history_desc.additional_constraints = "Must be one of: ";
    const std::string history_param = this->declare_parameter(
      "history", "keep_last", history_desc);
    history_policy_ = RMW_QOS_POLICY_HISTORY_KEEP_LAST;

    // Declare and get remaining parameters
    depth_ = this->declare_parameter("depth", 20);
    show_image_ = this->declare_parameter("show_image", true);
    window_name_ = this->declare_parameter("window_name", "");
  }

  /// Convert a sensor_msgs::Image encoding type (stored as a string) to an OpenCV encoding type.
  /**
   * \param[in] encoding A string representing the encoding type.
   * \return The OpenCV encoding type.
   */
  int encoding2mat_type(const std::string & encoding)
  {
    if (encoding == "mono8") {
      return CV_8UC1;
    } else if (encoding == "bgr8") {
      return CV_8UC3;
    } else if (encoding == "mono16") {
      return CV_16SC1;
    } else if (encoding == "rgba8") {
      return CV_8UC4;
    } else if (encoding == "bgra8") {
      return CV_8UC4;
    } else if (encoding == "32FC1") {
      return CV_32FC1;
    } else if (encoding == "rgb8") {
      return CV_8UC3;
    } else {
      throw std::runtime_error("Unsupported encoding type");
    }
  }

  /// Convert the ROS Image message to an OpenCV matrix and display it to the user.
  // \param[in] msg The image message to show.
  void process_image(
    const sensor_msgs::msg::Image::SharedPtr msg, bool show_image)
  {
    RCLCPP_INFO(get_logger(), "Received image #%s", msg->header.frame_id.c_str());
    if (show_image) {
      // Convert to an OpenCV matrix by assigning the data.
      cv::Mat frame(
        msg->height, msg->width, encoding2mat_type(msg->encoding),
        const_cast<unsigned char *>(msg->data.data()), msg->step);

      if (msg->encoding == "rgb8") {
        cv::cvtColor(frame, frame, cv::COLOR_RGB2BGR);
      }

      cv::Mat cvframe = frame;

      m_body_mutex.lock();
      for (size_t i = 0; i < body_rects.size(); i++) {
        cv::rectangle(frame, body_rects[i].rect, cv::Scalar(255, 0, 0), 3, 8, 0);
        cv::putText(
          frame, body_rects[i].id,
          cv::Point(body_rects[i].rect.x, body_rects[i].rect.y + 30),
          cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(255, 0, 0), 4, 8);
      }
      m_body_mutex.unlock();

      m_face_mutex.lock();
      for (size_t i = 0; i < face_rects.size(); i++) {
        cv::rectangle(frame, face_rects[i].rect, cv::Scalar(0, 255, 0), 3, 8, 0);
        cv::putText(
          frame, face_rects[i].id,
          cv::Point(face_rects[i].rect.x, face_rects[i].rect.y + face_rects[i].rect.height + 30),
          cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 255, 0), 4, 8);
      }
      m_face_mutex.unlock();

      cv::namedWindow(window_name_, cv::WINDOW_NORMAL);
      // Show the image in a window
      cv::imshow(window_name_, cvframe);
      // Draw the screen and wait for 1 millisecond.
      cv::waitKey(1);
    }
  }

  void process_body_msg(const BodyInfoT::SharedPtr msg)
  {
    m_body_mutex.lock();
    body_rects.clear();
    for (size_t i = 0; i < msg->count; i++) {
      RectId item;
      item.rect = cv::Rect(
        msg->infos[i].roi.x_offset / 2,
        msg->infos[i].roi.y_offset / 2,
        msg->infos[i].roi.width / 2,
        msg->infos[i].roi.height / 2);
      item.id = msg->infos[i].reid;
      body_rects.push_back(item);
    }
    m_body_mutex.unlock();
  }

  void process_face_msg(const FaceInfoT::SharedPtr msg)
  {
    m_face_mutex.lock();
    face_rects.clear();
    for (size_t i = 0; i < msg->count; i++) {
      RectId item;
      item.rect = cv::Rect(
        msg->infos[i].roi.x_offset / 2,
        msg->infos[i].roi.y_offset / 2,
        msg->infos[i].roi.width / 2,
        msg->infos[i].roi.height / 2);
      item.id = msg->infos[i].id;
      if (msg->infos[i].is_host) {
        item.id += "(host)";
      }
      face_rects.push_back(item);
    }
    m_face_mutex.unlock();
  }

  rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr sub_;
  size_t depth_ = rmw_qos_profile_default.depth;
  rmw_qos_reliability_policy_t reliability_policy_ = rmw_qos_profile_default.reliability;
  rmw_qos_history_policy_t history_policy_ = rmw_qos_profile_default.history;
  bool show_image_ = true;
  std::string topic_ = "image";
  std::string window_name_;

  std::vector<RectId> body_rects;
  std::mutex m_body_mutex;
  std::vector<RectId> face_rects;
  std::mutex m_face_mutex;
  rclcpp::Subscription<BodyInfoT>::SharedPtr sub_body;
  rclcpp::Subscription<FaceInfoT>::SharedPtr sub_face;
};

int main(int argc, char * argv[])
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<ShowImage>());
  rclcpp::shutdown();

  return 0;
}
