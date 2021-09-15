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

#include <cmath>
#include <iostream>

#include "dirent.h"  // NOLINT
#include "sys/unistd.h"  // NOLINT
#include "sys/time.h"
#include "sys/fcntl.h"  // NOLINT

#include "cyberdog_utils/can/can_proto.h"

#include "rclcpp/rclcpp.hpp"
#include "cyberdog_utils/sensor/sensor_utils.hpp"

#define DOUBLE_DATA_SIZE 2
#define TRIPLE_DATA_SIZE 3
#define FOUR_DATA_SIZE   4
#define HEART_ATTACK_INTERVAL 1000000000

#ifndef UNUSED_VAR
#define UNUSED_VAR(var)  ((void)(var));
#endif

namespace cyberdog_utils
{

cyberdog_sensor::cyberdog_sensor(int sensor_count)
{
  filter_sensor_count = sensor_count * 2;
  filter_id_table = reinterpret_cast<uint32_t *>(malloc(sizeof(uint32_t) * filter_sensor_count));
  filter_id_table_index = 0;
  publish_sensor_bitmask = 0;

  sync_time_nv2head = get_current_nv_time();
  sync_time_nv2bot = sync_time_nv2head;
  sync_time_nv2rear = sync_time_nv2head;
  sync_time_head_board = sync_time_nv2head;
  sync_time_bot_board = sync_time_nv2head;
  sync_time_rear_board = sync_time_nv2head;
  memset(previous_sensor_data, 0, sizeof(sensors_event_t) * SENSOR_TYPE_MAX);
  // get_regulater_name(3);
  // get_regulater_name(2);
  // get_regulater_name(0);
}

cyberdog_sensor::~cyberdog_sensor()
{
  free(filter_id_table);
}

void
cyberdog_sensor::get_regulater_name(int boardID)
{
  DIR * dir;
  struct dirent * ptr;
  char * regulatordir = nullptr;

  if (boardID == 3) {
    regulatordir = head_board_regulator_prefix;
  } else if (boardID == 2) {
    regulatordir = bot_board_regulator_prefix;
  } else if (boardID == 0) {
    regulatordir = rear_board_regulator_prefix;
  }

  memset(current_regulatorname, 0, strlen(current_regulatorname));

  if ((dir = opendir(regulatordir)) == NULL) {
    printf("Open dir error: %s, errno: %d\n\r", regulatordir, errno);
    return;
  }

  while ((ptr = readdir(dir)) != NULL) {
    // current dir OR parrent dir
    if (strcmp(ptr->d_name, ".") == 0 || strcmp(ptr->d_name, "..") == 0) {
      continue;
    } else if (ptr->d_type == 8) {  // file
      printf("d_name:%s%s\n", regulatordir, ptr->d_name);
    } else if (ptr->d_type == 10) {  // link file
      printf("d_name:%s%s\n", regulatordir, ptr->d_name);
    } else if (ptr->d_type == 4) {  // dir
      printf("current folder: %s \n\r", ptr->d_name);
      strncpy(current_regulatorname, regulatordir, strlen(regulatordir));
      snprintf(current_regulatorname, sizeof(current_regulatorname), "%s", ptr->d_name);
      snprintf(current_regulatorname, sizeof(current_regulatorname), "%s", "/state");
      break;
    }
  }
  printf("current regulater path: %s \n\r", current_regulatorname);
  closedir(dir);
}

int
cyberdog_sensor::set_convert_sensortype(uint16_t sensortype_bitmask)
{
  publish_sensor_bitmask |= sensortype_bitmask;
  return 0;
}

int
cyberdog_sensor::clear_convert_sensortype(uint16_t sensortype_bitmask)
{
  publish_sensor_bitmask &= ~sensortype_bitmask;
  return 0;
}

// this function should be stay aline with stm32 sensor dev struct
int
cyberdog_sensor::get_sensor_data_size(int sensortype)
{
  switch (sensortype) {
    case SENSOR_TYPE_MAGNETIC_FIELD:
      return TRIPLE_DATA_SIZE;
    case SENSOR_TYPE_SPEED_VECTOR:
    case SENSOR_TYPE_LIGHT_SPEED:
    case SENSOR_TYPE_PROXIMITY_HEAD:
    case SENSOR_TYPE_PROXIMITY_BOT:
    case SENSOR_TYPE_PROXIMITY_REAR:
      return DOUBLE_DATA_SIZE;
    case SENSOR_TYPE_ACCELEROMETER:
    case SENSOR_TYPE_GYROSCOPE:
    case SENSOR_TYPE_ROTATION_VECTOR:
    case SENSOR_TYPE_LIGHT:
      return FOUR_DATA_SIZE;
    default:
      return 1;
  }
}

int
cyberdog_sensor::find_sensor_filter_stdid()
{
  for (int i = 0; i < SENSOR_TYPE_MAX; i++) {
    if (publish_sensor_bitmask & (1 << i)) {
      std::cout << "find_sensor_filter_stdid, publish_sensor_bitmask: " <<
        std::hex << publish_sensor_bitmask << ", filter_id_table_index: " <<
        filter_id_table_index << std::endl;
      filter_id_table[filter_id_table_index] =
        SENSOR_DATA_MESSAGE << SENSOR_EVENT_MESSAGE_BIT_SHIFT |
        i << SENSOR_TYPE_BIT_SHIFT |
        get_sensor_data_size(i);
      filter_id_table_index++;
      filter_id_table[filter_id_table_index] =
        SENSOR_TIMESTAMP_MESSAGE << SENSOR_EVENT_MESSAGE_BIT_SHIFT |
        i << SENSOR_TYPE_BIT_SHIFT |
        get_sensor_data_size(i);
      filter_id_table_index++;
    }
  }
  return 0;
}

uint8_t
cyberdog_sensor::find_sensor_boardID(int sensor_type)
{
  switch (sensor_type) {
    case SENSOR_TYPE_ACCELEROMETER:
    case SENSOR_TYPE_MAGNETIC_FIELD:
    case SENSOR_TYPE_GYROSCOPE:
    case SENSOR_TYPE_LIGHT:
    case SENSOR_TYPE_LED_HEAD:
    case SENSOR_TYPE_PROXIMITY_HEAD:
    case SENSOR_TYPE_ROTATION_VECTOR:
      return 3;
    case SENSOR_TYPE_PROXIMITY_BOT:
    case SENSOR_TYPE_LIGHT_SPEED:
    case SENSOR_TYPE_SPEED_VECTOR:
      return 2;
    case SENSOR_TYPE_PROXIMITY_REAR:
    case SENSOR_TYPE_LED_REAR:
      return 0;
    default:
      return 3;
  }
}

uint64_t
cyberdog_sensor::timer_converter_board2nv(uint8_t boardID, uint64_t stm32_timestamp)
{
  switch (boardID) {
    case 3:
      return sync_time_nv2head + stm32_timestamp - sync_time_head_board;
    case 2:
      return sync_time_nv2bot + stm32_timestamp - sync_time_bot_board;
    case 0:
      return sync_time_nv2rear + stm32_timestamp - sync_time_rear_board;
    default:
      return sync_time_nv2head + stm32_timestamp - sync_time_head_board;
  }
}

uint64_t
cyberdog_sensor::get_current_nv_time()
{
  struct timespec ts;
  clock_gettime(CLOCK_REALTIME, &ts);

  return ts.tv_sec * 1000000000 + ts.tv_nsec;
}

bool
cyberdog_sensor::convert_timestamp_data(struct can_frame * recv_frame)
{
  bool ret = false;
  uint64_t stm32_timestamp_temp = 0;
  uint64_t nv_timestamp_temp = 0;
  if ((recv_frame->can_id & 0x0F00) >> SENSOR_EVENT_MESSAGE_BIT_SHIFT == SENSOR_TIME_SYNC_MSG &&
    (recv_frame->can_id & 0x000F) == 0xF)
  {
    nv_timestamp_temp = get_current_nv_time();
    unsigned int board_id = ((recv_frame->can_id & 0x00F0) >> SENSOR_TYPE_BIT_SHIFT);
    printf("board_id: %d\n\r", board_id);
    switch (board_id) {
      case 3:
        memcpy(&stm32_timestamp_temp, recv_frame->data, sizeof(uint64_t));
        sync_time_head_board = stm32_timestamp_temp * 1000000;
        sync_time_nv2head = nv_timestamp_temp;
        printf(
          "SENSOR_TIMESTAMP_SYNC, head: %lu, nv: %lu \n\r", sync_time_head_board,
          sync_time_nv2head);
        ret = true;
        break;
      case 2:
        memcpy(&stm32_timestamp_temp, recv_frame->data, sizeof(uint64_t));
        sync_time_bot_board = stm32_timestamp_temp * 1000000;
        sync_time_nv2bot = nv_timestamp_temp;
        printf(
          "SENSOR_TIMESTAMP_SYNC, bot: %lu, nv: %lu \n\r", sync_time_bot_board,
          sync_time_nv2bot);
        ret = true;
        break;
      case 0:
        memcpy(&stm32_timestamp_temp, recv_frame->data, sizeof(uint64_t));
        sync_time_rear_board = stm32_timestamp_temp * 1000000;
        sync_time_nv2rear = nv_timestamp_temp;
        printf(
          "SENSOR_TIMESTAMP_SYNC, rear: %lu, nv: %lu \n\r", sync_time_rear_board,
          sync_time_nv2rear);
        ret = true;
        break;
      default:
        break;
    }
  }
  return ret;
}

bool
cyberdog_sensor::check_sensor_data_valid(
  sensor_data_converter * sensor_data_converter,
  int sensor_type)
{
  switch (sensor_type) {
    case SENSOR_TYPE_ACCELEROMETER:
    case SENSOR_TYPE_MAGNETIC_FIELD:
    case SENSOR_TYPE_GYROSCOPE:
      return true;
    case SENSOR_TYPE_LIGHT:
      if (sensor_data_converter->sensor_data[SENSOR_TYPE_LIGHT].sensor_data_t.vec.data[0] < 0) {
        return false;
      } else {
        return true;
      }
    case SENSOR_TYPE_PROXIMITY_HEAD:
    case SENSOR_TYPE_PROXIMITY_REAR:
      if (sensor_data_converter->sensor_data[sensor_type].sensor_data_t.vec.data[1] <
        ULTRASOUND_AMPLITUDE_THRESHOLD)
      {
        return false;
      } else {
        return true;
      }
    case SENSOR_TYPE_PROXIMITY_BOT:
      return true;
    case SENSOR_TYPE_LIGHT_SPEED:
    case SENSOR_TYPE_SPEED_VECTOR:
      return true;
    case SENSOR_TYPE_ROTATION_VECTOR:
      return true;
    default:
      return true;
  }
}

bool
cyberdog_sensor::convert_sensor_data(
  sensor_data_converter * sensor_data_converter,
  struct can_frame * recv_frame)
{
  int i = 0;
  // int len = (recv_frame->len > CAN_MAX_DLEN)? CAN_MAX_DLEN : recv_frame->len;

  for (i = 0; i < filter_sensor_count; i++) {
    if (recv_frame->can_id == filter_id_table[i]) {
      unsigned int can_message_type = (recv_frame->can_id & 0x0F00) >>
        SENSOR_EVENT_MESSAGE_BIT_SHIFT;
      unsigned int sensor_type = (recv_frame->can_id & 0x00F0) >> SENSOR_TYPE_BIT_SHIFT;
      unsigned int sensor_data_size = recv_frame->can_id & 0x000F;
      uint8_t sensor_data_index = recv_frame->data[0];
      // std::cout << "convert_sensor_data, msg id: "
      //  << std::hex << recv_frame->can_id << std::endl;

      if (can_message_type == SENSOR_DATA_MESSAGE) {
        // std::cout << "SENSOR_DATA_MESSAGE, type: " << sensor_type << " ,sensor_data_bitmask: "
        //  << std::hex << sensor_data_converter->sensor_data_bitmask[sensor_type] << std::endl;

        if (!(sensor_data_converter->sensor_data_bitmask[sensor_type] & (1 << sensor_data_index)) &&
          sensor_type == sensor_data_converter->sensor_data[sensor_type].sensor_type)
        {
          memcpy(
            &sensor_data_converter->sensor_data[sensor_type].sensor_data_t.vec.data[
              sensor_data_index],
            &recv_frame->data[1],
            sizeof(float));
          sensor_data_converter->sensor_data_bitmask[sensor_type] |= (1 << sensor_data_index);
        } else if ((sensor_data_converter->sensor_data_bitmask[sensor_type] & // NOLINT
          (1 << sensor_data_index)) &&
          sensor_type == sensor_data_converter->sensor_data[sensor_type].sensor_type &&
          sensor_data_converter->sensor_data_bitmask[sensor_type] ==
          ((uint16_t)(pow(2, sensor_data_size) - 1)))
        {
          sensor_data_converter->sensor_data_bitmask[sensor_type] = 0x00;
          std::cout << " abnormal sensor data index found! reset!" << std::endl;
        }
      } else if (can_message_type == SENSOR_TIMESTAMP_MESSAGE) {
        // std::cout << "SENSOR_TIMESTAMP_MESSAGE, type: " << sensor_type << std::endl;
        if (sensor_data_converter->sensor_data_bitmask[sensor_type] ==
          ((uint16_t)(pow(2, sensor_data_size) - 1)) &&
          sensor_type == sensor_data_converter->sensor_data[sensor_type].sensor_type)
        {
          sensor_data_converter->sensor_data_bitmask[sensor_type] = 0x00;
          if ((1 << sensor_type) & publish_sensor_bitmask) {
            // std::cout << "data, type: " << sensor_type << "bit mask:"
            // << publish_sensor_bitmask << std::endl;
            uint8_t boardID = find_sensor_boardID(sensor_type);
            uint64_t stm32_timestamp = 0;
            memcpy(&stm32_timestamp, recv_frame->data, sizeof(uint64_t));
            stm32_timestamp = stm32_timestamp * 1000000;
            sensor_data_converter->sensor_data[sensor_type].timestamp = timer_converter_board2nv(
              boardID, stm32_timestamp);
            /*
              printf("receive sensordata, type: %d, {%f, %f, %f, %f}, @ %llu \n\r", sensor_type,
              sensor_data_converter->sensor_data[sensor_type].sensor_data_t.vec.data[0],
              sensor_data_converter->sensor_data[sensor_type].sensor_data_t.vec.data[1],
              sensor_data_converter->sensor_data[sensor_type].sensor_data_t.vec.data[2],
              sensor_data_converter->sensor_data[sensor_type].sensor_data_t.vec.data[3],
              sensor_data_converter->sensor_data[sensor_type].timestamp);
            */
            return true;
          }
        }
      }
    }
  }
  if (recv_frame->can_id == 0x70A) {
    RCLCPP_INFO(rclcpp::get_logger("rclcpp"), "SENSOR_BUS_ERROR.");
    if (((1 << SENSOR_TYPE_ACCELEROMETER) | (1 << SENSOR_TYPE_GYROSCOPE) |
      (1 << SENSOR_TYPE_ROTATION_VECTOR)) & publish_sensor_bitmask)
    {
      // trigger corrsponding sensor board power off/on
      board_regulator_control(find_sensor_boardID(SENSOR_TYPE_ACCELEROMETER), 0);
      usleep(500 * 1000);
      board_regulator_control(find_sensor_boardID(SENSOR_TYPE_ACCELEROMETER), 1);
      RCLCPP_INFO(rclcpp::get_logger("rclcpp"), "complete reset head board.!!!");
    }
  }
  return false;
}

int
cyberdog_sensor::board_regulator_control(int board_id, int enable)
{
  int fd = -1, ret = -1;

  char * enable_command = const_cast<char *>("enable");
  char * disable_command = const_cast<char *>("disable");

  char target_regulator[500] = {0};
  char enable_disable[20] = {0};

  get_regulater_name(board_id);
  strncpy(target_regulator, current_regulatorname, strlen(current_regulatorname));


  if (enable == 0) {
    memcpy(enable_disable, disable_command, strlen(disable_command));
  } else {
    memcpy(enable_disable, enable_command, strlen(enable_command));
  }

  fd = open(target_regulator, O_RDWR);
  if (fd < 0) {
    printf("path %s open error! error: %d\n", target_regulator, errno);
    return fd;
  }

  ret = write(fd, enable_disable, strlen(enable_disable));
  if (ret < 0) {
    printf("write %s failed for: %s\n", target_regulator, (enable == 0) ? "disable" : "enable");
    close(fd);
    return ret;
  }

  ret = close(fd);
  return 0;
}


void
cyberdog_sensor::log_prev_sensor_data(sensors_event_t sensor_data, int sensortype)
{
  memcpy(&previous_sensor_data[sensortype], &sensor_data, sizeof(sensors_event_t));
}

bool
cyberdog_sensor::sensor_data_heart_attack_checker(int sensortype)
{
  // printf("heart attack check, type: %d, timestamp now: %llu, timestamp prev: %llu \n\r",
  //  sensortype, get_current_nv_time(), previous_sensor_data[sensortype].timestamp);
  // todo: check timestamp between prev sensor data and current sensor time.
  // If there is a large gap such as 1s, it stands for sensor is stopping reporting data,
  //   then we should power on/off the stm32 board
  int64_t heart_attack_interval = 0;
  heart_attack_interval = get_current_nv_time() - previous_sensor_data[sensortype].timestamp;

  if (heart_attack_interval > HEART_ATTACK_INTERVAL) {
    // trigger corrsponding sensor board power off/on
    printf("reboot stm32 occured, target: %d \n\r", find_sensor_boardID(sensortype));
    board_regulator_control(find_sensor_boardID(sensortype), 0);
    usleep(500 * 1000);
    board_regulator_control(find_sensor_boardID(sensortype), 1);
    previous_sensor_data[sensortype].timestamp = get_current_nv_time();
    usleep(1000 * 1000);
  }

  return false;
}

}  // namespace cyberdog_utils
