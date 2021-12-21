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

#ifndef DEVICE_EMWAVE__GNSS_HPP_
#define DEVICE_EMWAVE__GNSS_HPP_

#include <ctime>
#include <variant>  // NOLINT

#include "common_base/common_type.hpp"
#include "common_base/input_device.hpp"

namespace cyberdog
{
namespace device
{

struct GNSSTargetT
{
  uint16_t id;
  uint16_t type;
  PoseT relat_pose;
};

/* NMEA cardinal direction types */
typedef char NMEACardinalT;
#define NMEA_CARDINAL_DIR_NORTH         (NMEACardinalT) 'N'
#define NMEA_CARDINAL_DIR_EAST          (NMEACardinalT) 'E'
#define NMEA_CARDINAL_DIR_SOUTH         (NMEACardinalT) 'S'
#define NMEA_CARDINAL_DIR_WEST          (NMEACardinalT) 'W'
#define NMEA_CARDINAL_DIR_UNKNOWN       (NMEACardinalT) '\0'

struct NMEAPositionT
{
  NMEACardinalT cardinal;
  uint32_t degree;
  double minutes;
};
// Ref to https://docs.novatel.com/OEM7/Content/Logs/GPGGA.htm
struct GPGGA
{
  std::time_t utc_time;
  NMEAPositionT lontitude;
  NMEAPositionT latitude;
  uint8_t position_fix;
  uint8_t n_satellites;
  float hdop;
  double altitude;
  uint8_t altitude_units;
  double undulation;
  uint8_t undulation_units;
  uint8_t diff_age_of_correction;
  uint16_t diff_base_station_id;
};
// Ref to https://docs.novatel.com/OEM7/Content/Logs/GPGLL.htm
struct GPGLL
{
  std::time_t utc_time;
  NMEAPositionT lontitude;
  NMEAPositionT latitude;
  uint8_t data_status;
  uint8_t mode_indicator;
};
// Ref to https://docs.novatel.com/OEM7/Content/Logs/GPGSA.htm
struct GPGSA
{
  uint8_t mode;
  uint8_t fixtype;
  int sat_id_00;
  int sat_id_01;
  int sat_id_02;
  int sat_id_03;
  int sat_id_04;
  int sat_id_05;
  int sat_id_06;
  int sat_id_07;
  int sat_id_08;
  int sat_id_09;
  int sat_id_10;
  int sat_id_11;
  double pdop;
  double hdop;
  double vdop;
};
// Ref to https://docs.novatel.com/OEM7/Content/Logs/GPGSV.htm
struct GPGSV
{
  std::time_t utc_time;
  NMEAPositionT lontitude;
  NMEAPositionT latitude;
  uint8_t n_satellites;
  struct
  {
    int prn;
    int elevation;
    int azimuth;
    int snr;
  } sat[4];
};
// Ref to https://docs.novatel.com/OEM7/Content/Logs/GPRMC.htm
struct GPRMC
{
  std::time_t utc_time;
  NMEAPositionT lontitude;
  NMEAPositionT latitude;
  double speed_knots;
  double track_deg;
  double magvar_deg;
  NMEAPositionT magvar_cardinal;
  uint8_t magvar_direction;
  uint8_t mode_indicator;
};
// https://docs.novatel.com/OEM7/Content/Logs/GPVTG.htm
struct GPVTG
{
  double mag_track_deg;
  double speed_knots;
  double speed_kmph;
  uint8_t mag_track_indicator;
  uint8_t speed_knots_indicator;
  uint8_t speed_kmph_indicator;
  uint8_t mode_indicator;
};

typedef std::variant<GPGGA, GPGLL, GPGSA, GPGSV, GPRMC, GPVTG> GNSSDataT;
typedef uint32_t GNSSModeT;
typedef bool GNSSArgK;
typedef bool GNSSArgV;
typedef bool GNSSCalibT;

/**
 * @brief GNSS is designed for Global Navigation Satellite System devices with many type of
 * data. Like GPGGA, GPGLL, GPGSA, GPGSV, GPRMC and GPVTG.
 * You will got sequential data from this device after setting callback function.
 * You must initialize device with function init(), set device modules informations, and
 * synchronize data if you need. The synchronization mechanism is up to devices and protocol.
 * After initialization, set callback function, please.
 * Argument Key to Argument Map is designed for calibration test online.
 */
class GNSS : public virtual InputDevice
  <GNSSTargetT, GNSSDataT, GNSSModeT, GNSSArgK, GNSSArgV, GNSSCalibT> {};
}  // namespace device
}  // namespace cyberdog

#endif  // DEVICE_EMWAVE__GNSS_HPP_
