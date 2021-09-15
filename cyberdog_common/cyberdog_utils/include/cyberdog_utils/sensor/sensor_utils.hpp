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

#ifndef CYBERDOG_UTILS__SENSOR__SENSOR_UTILS_HPP_
#define CYBERDOG_UTILS__SENSOR__SENSOR_UTILS_HPP_

#include <ctype.h>
#include <errno.h>
#include <libgen.h>
#include <signal.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

#include <linux/can.h>
#include <linux/can/raw.h>
#include <linux/can/error.h>

#include "cyberdog_utils/sensor/sensor.h"

#ifndef ARRAY_SIZE
#define ARRAY_SIZE(arr) (sizeof(arr) / sizeof((arr)[0]))
#endif

#define ULTRASOUND_AMPLITUDE_THRESHOLD 0.0f

typedef struct sensor_data_converter
{
  sensors_event_t sensor_data[SENSOR_TYPE_MAX];
  uint16_t sensor_data_bitmask[SENSOR_TYPE_MAX];
} sensor_data_converter;

namespace cyberdog_utils
{

class cyberdog_sensor
{
public:
  explicit cyberdog_sensor(int sensor_count);
  ~cyberdog_sensor();

  bool convert_sensor_data(
    sensor_data_converter * sensor_data_converter,
    struct can_frame * recv_frame);
  void log_prev_sensor_data(sensors_event_t sensor_data, int sensortype);
  bool convert_timestamp_data(struct can_frame * recv_frame);
  bool sensor_data_heart_attack_checker(int sensortype);
  bool check_sensor_data_valid(sensor_data_converter * sensor_data_converter, int sensor_type);

  int set_convert_sensortype(uint16_t sensortype_bitmask);
  int clear_convert_sensortype(uint16_t sensortype_bitmask);
  int find_sensor_filter_stdid();

  sensors_event_t previous_sensor_data[SENSOR_TYPE_MAX];
  uint64_t get_current_nv_time();

private:
  int get_sensor_data_size(int sensortype);
  int board_regulator_control(int board_id, int enable);
  void get_regulater_name(int boardID);

  uint8_t find_sensor_boardID(int sensor_type);
  uint64_t timer_converter_board2nv(uint8_t boardID, uint64_t stm32_timestamp);

  uint32_t * filter_id_table;
  int32_t filter_sensor_count;
  int32_t filter_id_table_index;

  uint16_t publish_sensor_bitmask;
  uint64_t sync_time_head_board;  // timestamp sync message on head stm32
  uint64_t sync_time_bot_board;  // timestamp sync message on bot stm32
  uint64_t sync_time_rear_board;  // timestamp sync message on rear stm32
  uint64_t sync_time_nv2head;  // timestamp sync message mapping NV/head board
  uint64_t sync_time_nv2bot;  // timestamp sync message mapping NV/bot board
  uint64_t sync_time_nv2rear;  // timestamp sync message mapping NV/rear board

  char * head_board_regulator_prefix =
    const_cast<char *>(
    "/sys/devices/fixed-regulators/fixed-regulators:reulator@26/regulator/");
  char * bot_board_regulator_prefix =
    const_cast<char *>(
    "/sys/devices/fixed-regulators/fixed-regulators:reulator@29/regulator/");
  char * rear_board_regulator_prefix =
    const_cast<char *>(
    "/sys/devices/fixed-regulators/fixed-regulators:reulator@25/regulator/");

  char current_regulatorname[500];
};
}  // namespace cyberdog_utils
#endif  // CYBERDOG_UTILS__SENSOR__SENSOR_UTILS_HPP_
