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

#include <lcm/lcm-cpp.hpp>
#include <rclcpp_action/rclcpp_action.hpp>
#include <stdio.h>
#include <unistd.h>
#include <memory>
#include <string>
#include <chrono>
#include "bms_common.hpp"
#include "time_interval.hpp"
#include "rclcpp/rclcpp.hpp"

#include "lcm_translate_msgs/bms_request_lcmt.hpp"
#include "lcm_translate_msgs/bms_response_lcmt.hpp"
#include "ception_msgs/srv/sensor_detection_node.hpp"
#include "ception_msgs/msg/shut_down.hpp"
#include "interaction_msgs/msg/touch.hpp"
#include "interaction_msgs/action/audio_play.hpp"


#define LPWG_SINGLETAP_DETECTED                      0x01
#define LPWG_DOUBLETAP_DETECTED                      0x03
#define LPWG_TOUCHANDHOLD_DETECTED                   0x07
#define LPWG_CIRCLE_DETECTED                         0x08
#define LPWG_TRIANGLE_DETECTED                       0x09
#define LPWG_VEE_DETECTED                            0x0A
#define LPWG_UNICODE_DETECTED                        0x0B
#define LPWG_SWIPE_DETECTED                          0x0D
#define LPWG_SWIPE_DETECTED_UP_CONTINU               0x0E
#define LPWG_SWIPE_DETECTED_DOWN_CONTINU             0x0F
#define LPWG_SWIPE_DETECTED_LEFT_CONTINU             0x10
#define LPWG_SWIPE_DETECTED_RIGHT_CONTINU            0x11

#define LPWG_SWIPE_FINGER_NUM_MASK                   0xF0
#define LPWG_SWIPE_FINGER_UP_DOWN_DIR_MASK           0x03
#define LPWG_SWIPE_FINGER_LEFT_RIGHT_DIR_MASK        0x0C
#define LPWG_SWIPE_ID_SINGLE                         0x40
#define LPWG_SWIPE_ID_DOUBLE                         0x80
#define LPWG_SWIPE_UP                                0x01
#define LPWG_SWIPE_DOWN                              0x02
#define LPWG_SWIPE_LEFT                              0x04
#define LPWG_SWIPE_RIGHT                             0x08
#define AUDIO_NOTICE_LOW_POWER  101
#define AUDIO_NOTICE_CHARGING_BASE 200
#define AUDIO_NOTICE_CHARGING_BEGIN 102
#define AUDIO_NOTICE_CHARGING_FULL 104
#define AUDIO_NOTICE_DISCHARGING_BASE 300
#define AUDIO_NOTICE_SHUTDOWN 2
#define MIN_BOOT_TIME 30


typedef enum
{
  LED_OFF,
  LED_BLINK,
  LED_NORMAL,
} LED_STATUS_T;

/*
BIT0:识别到有线充电桩
BIT1:应用板低电保护
BIT2:电池充电中
BIT3:电池已充满
BIT4:短路故障
BIT5:过温故障
BIT6:BMS通信故障0：未发生  1：已发生
*/

using std::placeholders::_1;
using namespace std::chrono_literals;
rclcpp::Node::SharedPtr node_handle = nullptr;
FILE * info_file = NULL;


#define BMS_INFO_FILE_PATH "/tmp/bms_info"
#define BATTERY_SOC_LOW 10
#define BATTERY_SOC_MEDIUM 60
#define BMS_CLIENT_REAR_LED 1
#define BMS_CLIENT_HEAD_LED 1
#define SETBIT(x, y) ((x) |= (1 << (y)))
#define CLRBIT(x, y) ((x) &= !(1 << (y)))
#define SEND_CMD_TTL        2
#define RECV_CMD_TTL        12
#define RETYR_TIME 1
using AudioPlayT = interaction_msgs::action::AudioPlay;
using GoalHandleAudio = rclcpp_action::ClientGoalHandle<AudioPlayT>;
class Bms : public rclcpp::Node
{
public:
  Bms();

private:
  int lcm_cnt;
  ception_msgs::msg::Bms message_;
  ception_msgs::msg::ShutDown shutdown_msg;
  void bms_info_local_save(void);
  void bms_led_control(LED_STATUS_T status);
  void shutdown_process(uint8_t reason);
  void bms_receive_msg(
    const lcm::ReceiveBuffer * rbuf,
    const std::string & chan,
    const bms_response_lcmt * lcm_data);
  void set_buzze(const ception_msgs::msg::Bms::SharedPtr msg);
  void get_touch(const interaction_msgs::msg::Touch::SharedPtr msg);
  void bms_pub_shutdown_msg(uint8_t reason);
  rclcpp::Publisher<ception_msgs::msg::Bms>::SharedPtr bms_pub;
  rclcpp::Publisher<ception_msgs::msg::ShutDown>::SharedPtr shutdown_pub;
  rclcpp::Subscription<ception_msgs::msg::Bms>::SharedPtr buzze_sub;
  rclcpp::Subscription<interaction_msgs::msg::Touch>::SharedPtr touch_sub;
  rclcpp::Service<ception_msgs::srv::BmsInfo>::SharedPtr bms_info_server_;
  rclcpp::CallbackGroup::SharedPtr callback_group_;
  std::shared_ptr<lcm::LCM> bms_response;
  void handle_service(
    const std::shared_ptr<rmw_request_id_t> request_header,
    const std::shared_ptr<ception_msgs::srv::BmsInfo::Request> request,
    std::shared_ptr<ception_msgs::srv::BmsInfo::Response> response);
  std::shared_ptr<std::thread> lcm_handle_thread_;
  void recv_lcm_handle();
  bool shutting_down;
  void send_shutdown_lcm();
  bool needblink;
  std_msgs::msg::Header returnHeader();
  rclcpp::Client<ception_msgs::srv::SensorDetectionNode>::SharedPtr led_client_;
  std::shared_ptr<lcm::LCM> bms_request;
  bms_request_lcmt bms_lcm_data;
  std::string getLcmUrl(int ttl);
  int8_t last_status;
  TimeInterval time_interval;
  TimeInterval boot_time_interval;
  rclcpp_action::Client<AudioPlayT>::SharedPtr audio_client;
  bool sendGoal(int audio_id);
  bool is_in_ota();
  void goal_response_callback(std::shared_future<GoalHandleAudio::SharedPtr> future);
  void feedback_callback(
    GoalHandleAudio::SharedPtr,
    const std::shared_ptr<const AudioPlayT::Feedback> feedback);
  void result_callback(const GoalHandleAudio::WrappedResult & result);
  bool processing_audo;
  uint8_t last_soc;
  bool need_retry;
  std::shared_ptr<ception_msgs::srv::SensorDetectionNode::Request> last_rear_request;
  std::shared_ptr<ception_msgs::srv::SensorDetectionNode::Request> last_head_request;
  void set_head_led(
    int head_command, int head_priority,
    uint64_t timeout = ception_msgs::srv::SensorDetectionNode::Request::COMMAND_ALWAYSON);
  void set_rear_led(
    int rear_command, int rear_priority,
    uint64_t timeout = ception_msgs::srv::SensorDetectionNode::Request::COMMAND_ALWAYSON);
};
std::string Bms::getLcmUrl(int ttl)
{
  return "udpm://239.255.76.67:7672?ttl=" + std::to_string(ttl);
}
Bms::Bms()
: Node("bms_recv"), lcm_cnt(0), shutting_down(false), needblink(false),
  last_status(CHARGING_STATUS_UNKNOWN), processing_audo(false), last_soc(BATTERY_SOC_LOW + 1),
  need_retry(false),
  last_rear_request(nullptr), last_head_request(nullptr)
{
  callback_group_ = this->create_callback_group(rclcpp::CallbackGroupType::Reentrant);
  bms_pub = this->create_publisher<ception_msgs::msg::Bms>("bms_recv", rclcpp::SystemDefaultsQoS());
  shutdown_pub = this->create_publisher<ception_msgs::msg::ShutDown>(
    "shuting_down",
    rclcpp::SystemDefaultsQoS());


  RCLCPP_INFO(node_handle->get_logger(), "bms timer created");

  buzze_sub = this->create_subscription<ception_msgs::msg::Bms>(
    "buzze", rclcpp::SystemDefaultsQoS(), std::bind(&Bms::set_buzze, this, _1));

  touch_sub = this->create_subscription<interaction_msgs::msg::Touch>(
    "TouchState", rclcpp::SystemDefaultsQoS(), std::bind(&Bms::get_touch, this, _1));

  bms_response = std::make_shared<lcm::LCM>(getLcmUrl(RECV_CMD_TTL));
  bms_response->subscribe("bms_data", &Bms::bms_receive_msg, this);

  bms_info_server_ = this->create_service<ception_msgs::srv::BmsInfo>(
    "get_bms_info",
    std::bind(
      &Bms::handle_service, this, std::placeholders::_1, std::placeholders::_2,
      std::placeholders::_3),
    rmw_qos_profile_default, callback_group_);

  led_client_ = this->create_client<ception_msgs::srv::SensorDetectionNode>("cyberdog_led");
  audio_client = rclcpp_action::create_client<AudioPlayT>(this, "audio_play");
  lcm_handle_thread_ = std::make_shared<std::thread>(&Bms::recv_lcm_handle, this);
  bms_request = std::make_shared<lcm::LCM>(getLcmUrl(SEND_CMD_TTL));
  time_interval.init();
  boot_time_interval.init();
}
void Bms::handle_service(
  const std::shared_ptr<rmw_request_id_t> request_header,
  const std::shared_ptr<ception_msgs::srv::BmsInfo::Request> request,
  std::shared_ptr<ception_msgs::srv::BmsInfo::Response> response)
{
  (void)request_header;
  (void)request;
  response->header = returnHeader();
  RCLCPP_INFO(node_handle->get_logger(), "bms handle_service");
  response->info.batt_volt = message_.batt_volt;
  response->info.batt_curr = message_.batt_curr;
  response->info.batt_temp = message_.batt_temp;
  response->info.batt_soc = message_.batt_soc;
  response->info.status = message_.status;
  response->info.key_val = message_.key_val;
  response->success = true;
}
std_msgs::msg::Header Bms::returnHeader()
{
  std_msgs::msg::Header msg;
  msg.frame_id = "Bms";
  msg.stamp = this->get_clock()->now();
  return msg;
}
void Bms::set_head_led(int head_command, int head_priority, uint64_t timeout)
{
  auto head_request = std::make_shared<ception_msgs::srv::SensorDetectionNode::Request>();

  // cancle priveous led request first
  if (last_head_request != nullptr &&
    timeout == ception_msgs::srv::SensorDetectionNode::Request::COMMAND_ALWAYSON)
  {
    head_request = last_head_request;
    head_request->timeout = ception_msgs::srv::SensorDetectionNode::Request::COMMAND_OFF;
    led_client_->async_send_request(head_request);
  }

  // set new led request
  head_request->clientid = BMS_CLIENT_HEAD_LED;
  head_request->command = head_command;
  head_request->priority = head_priority;
  head_request->timeout = timeout;
  if (timeout == ception_msgs::srv::SensorDetectionNode::Request::COMMAND_ALWAYSON) {
    last_head_request = head_request;
  }
  led_client_->async_send_request(head_request);
}

void Bms::set_rear_led(int rear_command, int rear_priority, uint64_t timeout)
{
  auto rear_request = std::make_shared<ception_msgs::srv::SensorDetectionNode::Request>();
  // cancle priveous led request first
  if (last_rear_request != nullptr &&
    timeout == ception_msgs::srv::SensorDetectionNode::Request::COMMAND_ALWAYSON)
  {
    rear_request = last_rear_request;
    rear_request->timeout = ception_msgs::srv::SensorDetectionNode::Request::COMMAND_OFF;
    led_client_->async_send_request(rear_request);
  }

  // set new led request
  rear_request->clientid = BMS_CLIENT_REAR_LED;
  rear_request->command = rear_command;
  rear_request->priority = rear_priority;
  rear_request->timeout = timeout;
  if (timeout == ception_msgs::srv::SensorDetectionNode::Request::COMMAND_ALWAYSON) {
    last_rear_request = rear_request;
  }
  led_client_->async_send_request(rear_request);
}

void Bms::bms_led_control(LED_STATUS_T type)
{
  switch (type) {
    case LED_BLINK:
      {
        RCLCPP_INFO(node_handle->get_logger(), "bms_led_control report error");
        set_head_led(
          ception_msgs::srv::SensorDetectionNode::Request::HEAD_LED_RED_BLINK,
          ception_msgs::srv::SensorDetectionNode::Request::TYPE_ALARM);

        set_rear_led(
          ception_msgs::srv::SensorDetectionNode::Request::REAR_LED_RED_BLINK,
          ception_msgs::srv::SensorDetectionNode::Request::TYPE_ALARM);
        return;
      }
    case LED_OFF:
      {
        RCLCPP_INFO(node_handle->get_logger(), "bms_led_control off");
        set_head_led(
          ception_msgs::srv::SensorDetectionNode::Request::HEAD_LED_OFF,
          ception_msgs::srv::SensorDetectionNode::Request::TYPE_EFFECTS);

        set_rear_led(
          ception_msgs::srv::SensorDetectionNode::Request::REAR_LED_OFF,
          ception_msgs::srv::SensorDetectionNode::Request::TYPE_EFFECTS);
        return;
      }
    case LED_NORMAL:
      {
        if (!message_.status && boot_time_interval.check_once(MIN_BOOT_TIME) &&
          message_.batt_soc < BATTERY_SOC_LOW && last_soc >= BATTERY_SOC_LOW)
        {
          RCLCPP_INFO(node_handle->get_logger(), "bms_led_control battery low");
          set_head_led(
            ception_msgs::srv::SensorDetectionNode::Request::HEAD_LED_RED_ON,
            ception_msgs::srv::SensorDetectionNode::Request::TYPE_ALARM);

          set_rear_led(
            ception_msgs::srv::SensorDetectionNode::Request::REAR_LED_RED_BLINK,
            ception_msgs::srv::SensorDetectionNode::Request::TYPE_ALARM);

          sendGoal(AUDIO_NOTICE_LOW_POWER);
          last_soc = message_.batt_soc;
          return;
        }
        RCLCPP_INFO(
          node_handle->get_logger(), "last_status %d, message_.status:%d, need_retry:%d",
          last_status, message_.status, need_retry);

        if (last_status == message_.status && !need_retry) {
          return;
        }

        switch (message_.status) {
          case CHARGING_STATUS_FULL:
            if (last_status == CHARGING_STATUS_CHARGING || last_status == CHARGING_STATUS_UNKNOWN) {
              RCLCPP_INFO(node_handle->get_logger(), "bms_led_control charged");
              set_head_led(
                ception_msgs::srv::SensorDetectionNode::Request::HEAD_LED_DARKBLUE_ON,
                ception_msgs::srv::SensorDetectionNode::Request::TYPE_FUNCTION);

              set_rear_led(
                ception_msgs::srv::SensorDetectionNode::Request::REAR_LED_OFF,
                ception_msgs::srv::SensorDetectionNode::Request::TYPE_FUNCTION);
              sendGoal(AUDIO_NOTICE_CHARGING_FULL);
            }
            break;

          case CHARGING_STATUS_CHARGING:
            RCLCPP_INFO(node_handle->get_logger(), "bms_led_control charging");
            set_head_led(
              ception_msgs::srv::SensorDetectionNode::Request::HEAD_LED_DARKBLUE_BREATH,
              ception_msgs::srv::SensorDetectionNode::Request::TYPE_FUNCTION);

            set_rear_led(
              ception_msgs::srv::SensorDetectionNode::Request::REAR_LED_RED_BREATH,
              ception_msgs::srv::SensorDetectionNode::Request::TYPE_FUNCTION);
            sendGoal(AUDIO_NOTICE_CHARGING_BEGIN);
            break;

          case CHARGING_STATUS_DISCHARGE:
            RCLCPP_INFO(node_handle->get_logger(), "bms_led_control discharging");
            if (boot_time_interval.check_once(MIN_BOOT_TIME) &&
              message_.batt_soc < BATTERY_SOC_LOW && last_status == CHARGING_STATUS_CHARGING)
            {
              RCLCPP_INFO(node_handle->get_logger(), "bms_led_control battery low");

              set_head_led(
                ception_msgs::srv::SensorDetectionNode::Request::HEAD_LED_RED_ON,
                ception_msgs::srv::SensorDetectionNode::Request::TYPE_ALARM);

              set_rear_led(
                ception_msgs::srv::SensorDetectionNode::Request::REAR_LED_RED_BLINK,
                ception_msgs::srv::SensorDetectionNode::Request::TYPE_ALARM);

              sendGoal(AUDIO_NOTICE_LOW_POWER);
              break;
            }
            set_head_led(
              ception_msgs::srv::SensorDetectionNode::Request::HEAD_LED_DARKBLUE_ON,
              ception_msgs::srv::SensorDetectionNode::Request::TYPE_EFFECTS);

            set_rear_led(
              ception_msgs::srv::SensorDetectionNode::Request::REAR_LED_RED_ON,
              ception_msgs::srv::SensorDetectionNode::Request::TYPE_EFFECTS);
            break;

          default:
            RCLCPP_INFO(node_handle->get_logger(), "unknown charing status");
            break;
        }
        last_status = message_.status;
        break;
      }
    default:
      RCLCPP_INFO(node_handle->get_logger(), "unknown led status");
      break;
  }
}

void Bms::bms_info_local_save(void)
{
  if (NULL == info_file) {
    info_file = fopen(BMS_INFO_FILE_PATH, "w+");
  }

  if (NULL != info_file) {
    fseek(info_file, 0, SEEK_SET);
    fprintf(info_file, "volt:%d\n", message_.batt_volt);
    fprintf(info_file, "curr:%d\n", message_.batt_curr);
    fprintf(info_file, "temp:%d\n", message_.batt_temp);
    fprintf(info_file, "soc:%d\n", message_.batt_soc);
    fprintf(info_file, "status:%d\n", message_.status);
    fprintf(info_file, "key_val:%d\n", message_.key_val);
    fprintf(info_file, "batt_health:%d\n", message_.batt_health);
    fprintf(info_file, "batt_loop_number:%d\n", message_.batt_loop_number);
    fprintf(info_file, "powerBoard_status:%d\n", message_.powerboard_status);
    fflush(info_file);
  }
}

void Bms::set_buzze(const ception_msgs::msg::Bms::SharedPtr msg)
{
  RCLCPP_INFO(node_handle->get_logger(), "bms set buzz :%d", msg->buzze);
}
void Bms::get_touch(const interaction_msgs::msg::Touch::SharedPtr msg)
{
  RCLCPP_INFO(node_handle->get_logger(), "get_touch");

  if (msg->touch_state == LPWG_SINGLETAP_DETECTED) {
    if (message_.batt_soc > 0 && message_.batt_soc <= 100) {
      if (message_.status == CHARGING_STATUS_CHARGING) {
        sendGoal(message_.batt_soc + AUDIO_NOTICE_CHARGING_BASE);
        set_head_led(
          ception_msgs::srv::SensorDetectionNode::Request::HEAD_LED_DARKBLUE_BLINK,
          ception_msgs::srv::SensorDetectionNode::Request::TYPE_ALARM, 3000000000);
      } else {
        sendGoal(message_.batt_soc + AUDIO_NOTICE_DISCHARGING_BASE);
      }
    }
  }
}
void Bms::send_shutdown_lcm()
{
  rclcpp::WallRate r(50ms);
  while (true) {
    SETBIT(bms_lcm_data.power_supply, 2);
    bms_request->publish("bms_command", &bms_lcm_data);
    r.sleep();
  }
}

void Bms::shutdown_process(uint8_t reason)
{
  if (shutting_down || is_in_ota()) {
    return;
  }
  shutting_down = true;
  rclcpp::WallRate r(1500ms);
  bms_led_control(LED_OFF);
  RCLCPP_INFO(node_handle->get_logger(), "shutdown_process");
  sendGoal(AUDIO_NOTICE_SHUTDOWN);
  std::shared_ptr<std::thread> shutdown_thread = std::make_shared<std::thread>(
    &Bms::send_shutdown_lcm, this);
  bms_pub_shutdown_msg(reason);
  r.sleep();
  system("echo '123' | sudo -S shutdown -h now");
}
void Bms::bms_pub_shutdown_msg(uint8_t reason)
{
  shutdown_msg.header = returnHeader();
  shutdown_msg.type = reason;
  shutdown_pub->publish(shutdown_msg);
}
void Bms::bms_receive_msg(
  const lcm::ReceiveBuffer * rbuf,
  const std::string & chan,
  const bms_response_lcmt * lcm_data)
{
  (void)(rbuf);
  (void)(chan);

  if (IS_BIT_SET(lcm_data->powerBoard_status, 0) && !needblink) {
    // motion_board_error
    needblink = true;
    bms_led_control(LED_BLINK);
  } else {
    if (IS_BIT_SET(lcm_data->status, BATTERY_LOW_BIT) && !needblink) {
      // low power protect
      needblink = true;
      bms_led_control(LED_BLINK);
    } else {
      needblink = false;
    }
  }

  if (IS_BIT_SET(lcm_data->key, SOFT_SHUTDOWN_BIT)) {
    shutdown_process(ception_msgs::msg::ShutDown::USER_TRIGGERED);
    return;
  }

  RCLCPP_INFO(node_handle->get_logger(), "bms_receive_msg");
  bms_log_store(lcm_data);
  RCLCPP_INFO(node_handle->get_logger(), "bms_receive_msg status:%d", lcm_data->status);
  int status = convert_status(lcm_data);
  if (message_.batt_volt != lcm_data->batt_volt ||
    message_.batt_curr != lcm_data->batt_curr ||
    message_.batt_temp != lcm_data->batt_temp ||
    message_.batt_soc != lcm_data->batt_soc ||
    message_.status != status)
  {
    message_.batt_volt = lcm_data->batt_volt;
    message_.batt_curr = lcm_data->batt_curr;
    message_.batt_temp = lcm_data->batt_temp;
    if (status == CHARGING_STATUS_FULL) {
      message_.batt_soc = 100;
    } else {
      message_.batt_soc = lcm_data->batt_soc;
    }
    RCLCPP_INFO(
      node_handle->get_logger(), "soc:%d, volt:%d", message_.batt_soc,
      message_.batt_volt);
    /*fix bms i2c issue for debug*/
    if (message_.batt_soc == 0 && message_.batt_volt > 20000) {
      message_.batt_soc = 55;
    }

    message_.status = status;
    message_.key_val = lcm_data->key;
    message_.batt_health = lcm_data->batt_health;
    message_.batt_loop_number = lcm_data->batt_loop_number;
    message_.powerboard_status = lcm_data->powerBoard_status;

    bms_info_local_save();
    bms_pub->publish(message_);
    bms_led_control(needblink ? LED_BLINK : LED_NORMAL);
  }
}
void Bms::recv_lcm_handle()
{
  rclcpp::WallRate r(100);
  while (0 == bms_response->handle()) {
    r.sleep();
  }
}
bool Bms::sendGoal(int audio_id)
{
  using namespace std::placeholders;

  if (!audio_client->wait_for_action_server(std::chrono::seconds(1))) {
    RCLCPP_INFO(node_handle->get_logger(), "Audio server not available after waiting");
    need_retry = true;
    return false;
  }
  if (processing_audo) {
    RCLCPP_INFO(node_handle->get_logger(), "previous audio request is processing");
    return false;
  }
  auto goal_msg = AudioPlayT::Goal();
  goal_msg.order.name.id = audio_id;
  goal_msg.order.user.id = 5;

  RCLCPP_INFO(node_handle->get_logger(), "Send audio play request");

  auto send_goal_options = rclcpp_action::Client<AudioPlayT>::SendGoalOptions();
  send_goal_options.goal_response_callback =
    std::bind(&Bms::goal_response_callback, this, _1);
  send_goal_options.feedback_callback =
    std::bind(&Bms::feedback_callback, this, _1, _2);
  send_goal_options.result_callback =
    std::bind(&Bms::result_callback, this, _1);
  audio_client->async_send_goal(goal_msg, send_goal_options);
  processing_audo = true;
  need_retry = false;
  return true;
}
void Bms::goal_response_callback(std::shared_future<GoalHandleAudio::SharedPtr> future)
{
  auto goal_handle = future.get();
  if (!goal_handle) {
    RCLCPP_INFO(node_handle->get_logger(), "Goal was rejected by server");
    processing_audo = false;
  } else {
    RCLCPP_INFO(node_handle->get_logger(), "Goal accepted by server, waiting for result");
  }
}

void Bms::feedback_callback(
  GoalHandleAudio::SharedPtr,
  const std::shared_ptr<const AudioPlayT::Feedback> feedback)
{
  RCLCPP_INFO(node_handle->get_logger(), "status: %u", feedback->feed.status);
}

void Bms::result_callback(const GoalHandleAudio::WrappedResult & result)
{
  RCLCPP_INFO(node_handle->get_logger(), "result: %u", result.result->result.error);
  processing_audo = false;
}
#define OTA_STATUS_IDLE "idle"
#define OTA_STATUS_DOWNLOAD "download"
#define OTA_STATUS_UPDATE "update"
#define OTA_FILE_PATH "/etc/ota_server/status"
bool Bms::is_in_ota()
{
  char buffer[32];
  bool result = false;
  int bufflen = sizeof(buffer) / sizeof(char);
  FILE * ota_status_file = NULL;
  ota_status_file = fopen(OTA_FILE_PATH, "r");
  memset(buffer, 0, bufflen);
  if (NULL == ota_status_file) {
    RCLCPP_INFO(node_handle->get_logger(), "get ota status open failed");
    return false;
  }

  fgets(buffer, bufflen, ota_status_file);
  RCLCPP_INFO(node_handle->get_logger(), "get ota status:%s", buffer);
  if (strncmp(OTA_STATUS_DOWNLOAD, buffer, strlen(OTA_STATUS_DOWNLOAD)) == 0 ||
    strncmp(OTA_STATUS_UPDATE, buffer, strlen(OTA_STATUS_UPDATE)) == 0)
  {
    result = true;
  }

  fclose(ota_status_file);
  return result;
}
int main(int argc, char * argv[])
{
  rclcpp::init(argc, argv);

  node_handle = rclcpp::Node::make_shared("bms_recv_node");
  info_file = fopen(BMS_INFO_FILE_PATH, "w+");
  RCLCPP_INFO(node_handle->get_logger(), "bms recv");
  log_file_status_check();

  rclcpp::spin(std::make_shared<Bms>());
  fclose(info_file);
  remove(BMS_INFO_FILE_PATH);
  rclcpp::shutdown();
  return 0;
}
