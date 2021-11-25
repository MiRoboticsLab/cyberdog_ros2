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

#include <termios.h>    // termios, TCSANOW, ECHO, ICANON
#include <unistd.h>     // STDIN_FILENO
#include <time.h>
#include <stdio.h>
#include <string.h>
#include <inttypes.h>
#include <syslog.h>
#include <math.h>

#include <memory>
#include <utility>
#include <string>
#include <fstream>
#include <thread>
#include <algorithm>
#include <cctype>
#include <vector>
#include <map>

#include "sys/stat.h"
#include "bream_vendor/patch_downloader.h"
#include "bream_vendor/bream_handler.h"
#include "bream_vendor/bream_helper.h"
#include "rclcpp/rclcpp.hpp"

#include "std_msgs/msg/string.hpp"
#include "motion_msgs/msg/scene.hpp"
#include "ception_msgs/srv/gps_scene_node.hpp"
#include "cyberdog_utils/lifecycle_node.hpp"

detection_scene environment = UNSET;
float lat_global = 0;
float lon_global = 0;
int if_danger = 0;
gps_info * gps_nmea = new gps_info;

static inline void ltrim(std::string & s)
{
  s.erase(
    s.begin(), std::find_if(
      s.begin(), s.end(), [](int ch) {
        return !std::isspace(ch);
      }));
}

// trim from end (in place)
static inline void rtrim(std::string & s)
{
  s.erase(
    std::find_if(
      s.rbegin(), s.rend(), [](int ch) {
        return !std::isspace(ch);
      }).base(), s.end());
}


// trim from both ends (in place)
static inline void trim(std::string & s)
{
  ltrim(s);
  rtrim(s);
}
namespace SceneDetection
{
class GpsPubNode : public cyberdog_utils::LifecycleNode
{
public:
  GpsPubNode();
  ~GpsPubNode();

protected:
  /* Lifecycle stages */
  cyberdog_utils::CallbackReturn on_configure(const rclcpp_lifecycle::State & state) override;
  cyberdog_utils::CallbackReturn on_activate(const rclcpp_lifecycle::State & state) override;
  cyberdog_utils::CallbackReturn on_deactivate(const rclcpp_lifecycle::State & state) override;
  cyberdog_utils::CallbackReturn on_cleanup(const rclcpp_lifecycle::State & state) override;
  cyberdog_utils::CallbackReturn on_shutdown(const rclcpp_lifecycle::State & state) override;

private:
  void gps_data_receiver_callback(void);

  void handle_service(
    const std::shared_ptr<rmw_request_id_t> request_header,
    const std::shared_ptr<ception_msgs::srv::GpsSceneNode::Request> request,
    const std::shared_ptr<ception_msgs::srv::GpsSceneNode::Response> response);

  // message struct define here
  motion_msgs::msg::Scene message;
  rclcpp::TimerBase::SharedPtr timer_;

  bool isthreadrunning;

  /* Service */
  rclcpp::Service<ception_msgs::srv::GpsSceneNode>::SharedPtr scene_detection_cmd_server_;
  rclcpp::CallbackGroup::SharedPtr callback_group_;
  rclcpp_lifecycle::LifecyclePublisher<motion_msgs::msg::Scene>::SharedPtr publisher_;
};  // class GpsPubNode
}  // namespace SceneDetection

namespace SceneDetection
{

GpsPubNode::GpsPubNode()
: cyberdog_utils::LifecycleNode("GpsPubNode")
{
  RCLCPP_INFO(get_logger(), "Creating GpsPubNode.");
}

GpsPubNode::~GpsPubNode()
{
  RCLCPP_INFO(get_logger(), "Destroying GpsPubNode");
}

cyberdog_utils::CallbackReturn GpsPubNode::on_configure(const rclcpp_lifecycle::State &)
{
  RCLCPP_INFO(get_logger(), "Configuring");
  callback_group_ = this->create_callback_group(rclcpp::CallbackGroupType::Reentrant);
  publisher_ = this->create_publisher<motion_msgs::msg::Scene>(
    "SceneDetection",
    rclcpp::SystemDefaultsQoS());
  scene_detection_cmd_server_ = this->create_service<ception_msgs::srv::GpsSceneNode>(
    "SceneDetection",
    std::bind(
      &GpsPubNode::handle_service, this, std::placeholders::_1, std::placeholders::_2,
      std::placeholders::_3),
    rmw_qos_profile_default, callback_group_);
  message = motion_msgs::msg::Scene();

  RCLCPP_INFO(get_logger(), "Configuring,success");
  return cyberdog_utils::CallbackReturn::SUCCESS;
}

cyberdog_utils::CallbackReturn GpsPubNode::on_activate(const rclcpp_lifecycle::State &)
{
  RCLCPP_INFO(get_logger(), "Activaing");
  publisher_->on_activate();
  isthreadrunning = true;

  timer_ = this->create_wall_timer(
    std::chrono::seconds(1),
    std::bind(&SceneDetection::GpsPubNode::gps_data_receiver_callback, this));

  return cyberdog_utils::CallbackReturn::SUCCESS;
}

cyberdog_utils::CallbackReturn GpsPubNode::on_deactivate(const rclcpp_lifecycle::State &)
{
  RCLCPP_INFO(get_logger(), "Deactiving");
  publisher_->on_deactivate();
  isthreadrunning = false;
  return cyberdog_utils::CallbackReturn::SUCCESS;
}

cyberdog_utils::CallbackReturn GpsPubNode::on_cleanup(const rclcpp_lifecycle::State &)
{
  publisher_.reset();
  return cyberdog_utils::CallbackReturn::SUCCESS;
}

cyberdog_utils::CallbackReturn GpsPubNode::on_shutdown(const rclcpp_lifecycle::State &)
{
  publisher_.reset();
  return cyberdog_utils::CallbackReturn::SUCCESS;
}

void GpsPubNode::gps_data_receiver_callback()
{
  if (gps_nmea->flag == 1 && (0.0 != gps_nmea->lat || 0.0 != gps_nmea->lon)) {
    if (gps_nmea->fixtype == 2 || gps_nmea->fixtype == 3) {
      environment = OUTDOOR;
      lat_global = gps_nmea->lat;
      lon_global = gps_nmea->lon;
      // if_danger = comparegeo(lat_global,lon_global);
      if_danger = false;
    } else {
      environment = INDOOR;
    }
    auto message = motion_msgs::msg::Scene();

    message.type = environment;
    message.lat = lat_global;
    message.lon = lon_global;
    message.if_danger = if_danger;

    publisher_->publish(std::move(message));
    gps_nmea->flag = 0;
    gps_nmea->fixtype = 0;
    gps_nmea->lat = 0;
    gps_nmea->lon = 0;
  }
}

void GpsPubNode::handle_service(
  const std::shared_ptr<rmw_request_id_t> request_header,
  const std::shared_ptr<ception_msgs::srv::GpsSceneNode::Request> request,
  const std::shared_ptr<ception_msgs::srv::GpsSceneNode::Response> response)
{
  (void)request_header;
  RCLCPP_INFO(
    get_logger(),
    "request: %d", request->command);
  response->success = true;

  switch (request->command) {
    case ception_msgs::srv::GpsSceneNode::Request::GPS_START:
      BreamHelper::GetInstance().GnssStart();
      break;
    case ception_msgs::srv::GpsSceneNode::Request::GPS_STOP:
      BreamHelper::GetInstance().GnssStop();
      LD2OS_delay(500);
      LD2OS_closeLog();
      LD2OS_openLog();
      break;
    default:
      response->success = false;
      break;
  }
}
}  // namespace SceneDetection

extern void RegisterBreamCallbacks(void);
bool threadRunning = false;

void Run()
{
  RegisterBreamCallbacks();
  threadRunning = true;

  while (threadRunning) {
    uint8_t rxBuff[1024];
    uint32_t rxLen = 0;

    bool ret = LD2OS_readFromSerial(rxBuff, &rxLen, sizeof(rxBuff), -1);
    if (!ret) {break;}
    if (rxLen) {
      BreamHandler::GetInstance().AsicData(rxBuff, rxLen);
    }
  }
}

char _getch_generic()
{
  return static_cast<char>(getchar());
}

void ConfigGnss()
{
  // Config GNSS
  LD2_LOG("ConfigGnss()\n");
  uint8_t infMsgMask[6] = {0x1F, 0x11, 0x0, 0x0, 0x0, 0x0};  // full logging(LOG_GLLIO, LOG_RAWDATA)
  BreamHelper::GetInstance().SetLogging(infMsgMask);
  BreamHelper::GetInstance().EnableBlindGalSearch();
  BreamHelper::GetInstance().SetPowerModePreset(1, 0);   // Default L1L5 Best Performance mode
  BreamHelper::GetInstance().SetMsgRate(0xF0, 0x00, 1);  // Report GPGGA for every measurment
  BreamHelper::GetInstance().SetMsgRate(0xF0, 0x04, 1);  // Report GPRMC for every measurment
  BreamHelper::GetInstance().SetMsgRate(0xF0, 0x03, 1);  // Report GPGSV for every measurment
  BreamHelper::GetInstance().SetMsgRate(0x01, 0x61, 1);  // Report NAVEOE for every measurment
  BreamHelper::GetInstance().SetMsgRate(0x01, 0x07, 1);  // Report NAVPVT for every measurment
  BreamHelper::GetInstance().SetMsgRate(0xF1, 0x00, 1);  // Report PGLOR SPEED for every measurment
  BreamHelper::GetInstance().SetMsgRate(0xF1, 0x01, 1);  // Report PGLOR FIX for every measurment
  BreamHelper::GetInstance().SetMsgRate(0xF1, 0x02, 1);  // Report PGLOR SAT for every measurment
  BreamHelper::GetInstance().SetMsgRate(0xF1, 0x03, 1);  // Report PGLOR LSQ for every measurment
  BreamHelper::GetInstance().SetMsgRate(0xF1, 0x04, 1);  // Report PGLOR PWR for every measurment
  BreamHelper::GetInstance().SetMsgRate(0xF1, 0x05, 1);  // Report PGLOR STA for every measurment
  BreamHelper::GetInstance().SetMsgRate(0x01, 0x09, 1);  // Report NAVODO for every measurment
  BreamHelper::GetInstance().SetMsgRate(0x01, 0x35, 1);  // Report NAVSAT for every measurment
  BreamHelper::GetInstance().SetMsgRate(0x01, 0x04, 1);  // Report NAVDOP for every measurment
  BreamHelper::GetInstance().SetMsgRate(0x01, 0x60, 1);  // Report CBEE status for every measurment
  BreamHelper::GetInstance().SetMsgRate(0x02, 0x13, 1);  // Report ASC SUBFRAMES forevery measurment
  BreamHelper::GetInstance().SetMsgRate(0x02, 0x15, 1);  // Report ASC MEAS for every measurment
  BreamHelper::GetInstance().SetMsgRate(0x02, 0x80, 1);  // Report ASC AGC for every measurment

  BreamHelper::GetInstance().SetAckAiding(true);
  BreamHelper::GetInstance().GetVer();
}

int main(int argc, char * argv[])
{
  const char * tty = nullptr;
  // setvbuf(stdout, NULL, _IONBF, 0);
  std::thread first;
  rclcpp::init(argc, argv);
  rclcpp::executors::MultiThreadedExecutor exec_;
  auto node = std::make_shared<SceneDetection::GpsPubNode>();

  LD2OS_openLog();

  system("echo 0 > /sys/devices/bcm4775/nstandby");
  sleep(1);
  system("echo 1 > /sys/devices/bcm4775/nstandby");
  std::vector<std::string> argVector(&argv[1], &argv[argc]);
  std::map<std::string, std::string> args;

  int portNumber = -1;
  bool bSkipDownload = false;
  LoDi2SerialConnection connType = LODI2_SERIAL_UART;
  std::map<std::string, std::string>::iterator it;

  static struct termios oldt, newt;
  tcgetattr(STDIN_FILENO, &oldt);
  newt = oldt;
  newt.c_lflag &= ~(ICANON);
  tcsetattr(STDIN_FILENO, TCSANOW, &newt);

  tty = "/dev/ttyTHS0";
  bSkipDownload = false;

  const int baudrate = 3000000;
  exec_.add_node(node->get_node_base_interface());


  if (!bSkipDownload) {
    // Open serial port
    LD2OS_open(connType, portNumber, baudrate, tty);
    // Download patch
    if (false == Bream_LoadPatch("/usr/sbin/bream.patch")) {
      goto errgo;
    }
  }

  if (connType == LODI2_SERIAL_UART) {
    // Reopen and send command to switch baudrate from default 115200
    LD2OS_open(connType, portNumber, 115200, tty);

    BreamHelper::GetInstance().SetBaudrate(baudrate);
    // delay needed because MCU needs some time in receiving CFG-PRT and handling it.
    // This delay can be replaced by checking ACK for CFG-PRT
    LD2OS_delay(100);
    // Reopen in new baudrate
    LD2OS_open(connType, portNumber, baudrate, tty);
  }
  // Register callbacks and start listener loop
  first = std::thread(Run);
  // If SPI connection, run read thread first and then send first packet to mcu
  if (connType == LODI2_SERIAL_SPI) {
    LD2OS_delay(200);
    BreamHelper::GetInstance().SetBaudrate(baudrate, 4);
  }

  // Config GNSS
  ConfigGnss();
  // Run command loop
  LD2_LOG(
    "==========================================================================================\n");
  LD2_LOG("  Hello Bream ! \n");
  BreamHelper::GetInstance().GnssStart();
  LD2OS_delay(10);
errgo:
  LD2_LOG("we going to errgo");
  exec_.spin();
  rclcpp::shutdown();

  threadRunning = false;
  first.join();


  LD2OS_close();

  LD2OS_closeLog();

  // restore the old terminal settings
  tcsetattr(STDIN_FILENO, TCSANOW, &oldt);

  return 0;
}
