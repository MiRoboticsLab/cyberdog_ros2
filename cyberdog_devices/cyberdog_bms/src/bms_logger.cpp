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


#include <fcntl.h>
#include <unistd.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <errno.h>
#include <sys/stat.h>
#include <dirent.h>
#include <time.h>
#include <sys/time.h>
#include "bms_common.hpp"

#define BMS_LOG_PATH "/home/mi/log"
#define BMS_LOG_FILE BMS_LOG_PATH "/charge_logger_a.csv"
#define READ_BUF_SIZE 64
#define LOG_TEXT_MAX_SIZE 2048
#define LOG_DELIMITER ","
#define DATE_STAMP_FMT "%04d/%02d/%02d"
#define TIME_STAMP_FMT "%02d:%02d:%02d"

#define LOCAL_ARRAY_SIZE(x) (static_cast<int>(sizeof(x) / sizeof((x)[0])))


static const char * basic_title_name[] = {
  "date",
  "time",
  "voltage",
  "current",
  "capacity",
  "temperture",
  "status",
  "key"};

int file_exists(const char * path)
{
  struct stat s;
  int rv;

  rv = stat(path, &s);
  return rv ? 0 : S_ISREG(s.st_mode);
}

int get_file_line(const char * fpath)
{
  char * ret;
  FILE * fp;
  int rc = -1, cnt = 0;
  char buf[LOG_TEXT_MAX_SIZE];

  if (fpath) {
    fp = fopen(fpath, "r");
    if (!fp) {
      printf("Open error: '%s'", fpath);
      goto error;
    }

    while (1) {
      ret = fgets(buf, LOG_TEXT_MAX_SIZE, fp);
      if (!ret) {
        break;
      }
      cnt++;
    }
    fclose(fp);
    rc = cnt;
  } else {
    printf("FILE fpath is NULL\n");
  }

error:
  return rc;
}

int file_access(const char * path, void * pdata, int length, int opt)
{
  int rc = -1;
  int fd = -1, idx = 0, last = length;
  char * ptr = reinterpret_cast<char *>(pdata);

  if (path && ptr) {
    fd = open(path, opt, S_IRUSR | S_IWUSR | S_IRGRP | S_IROTH);
    if (fd < 0) {
      printf("Open error: %s\n", path);
      goto error;
    }

    while (last) {
      rc = write(fd, ptr + idx, last);
      if (rc < 0) {
        if (errno != EINTR) {
          printf("Write error: %s\n", strerror(errno));
          break;
        } else {
          printf("EINTR occured errno:%s\n", strerror(errno));
        }
      } else {
        last -= rc;
        idx += rc;
        rc = 0;
      }
    }
  }

error:
  if (fd >= 0) {
    fsync(fd);
    close(fd);
  }
  return rc;
}

int create_log_title(const char * path)
{
  char buf[READ_BUF_SIZE] = {0};
  char line_buf[LOG_TEXT_MAX_SIZE] = {0};
  char line_buf_tmp[LOG_TEXT_MAX_SIZE] = {0};
  int rc = -1, i;

  if (path) {
    rc = mkdir(BMS_LOG_PATH, 0777);
    printf("mkdir return rc:%d\n", rc);
    for (i = 0; i < LOCAL_ARRAY_SIZE(basic_title_name); i++) {
      if (!i) {
        snprintf(
          buf, sizeof(buf) - 1, "%s",
          basic_title_name[i]);
      } else {
        snprintf(
          buf, sizeof(buf) - 1, LOG_DELIMITER "%s",
          basic_title_name[i]);
      }
      snprintf(line_buf_tmp, sizeof(line_buf_tmp), "%s", buf);
    }
    snprintf(line_buf, sizeof(line_buf), "%s%s", line_buf_tmp, "\n");
    rc = file_access(
      path, line_buf, strlen(line_buf),
      O_CREAT | O_TRUNC | O_RDWR | O_SYNC);
  }

  return rc;
}

int log_file_status_check(void)
{
  int cnt_active = 0;
  int rc;

  rc = file_exists(BMS_LOG_FILE);
  if (rc > 0) {
    cnt_active = get_file_line(BMS_LOG_FILE);
  }

  if (cnt_active <= 0) {
    rc = create_log_title(BMS_LOG_FILE);
  }

  printf("cnt_active1:%d\n", rc);

  return rc;
}

int bms_log_store(const bms_response_lcmt * msg)
{
  char line_buf[LOG_TEXT_MAX_SIZE] = {0};
  struct tm stm;
  time_t recv_time;
  int rc = 0;

  /*save the bms log 10s one time*/
  time(&recv_time);
  localtime_r(&recv_time, &stm);
  int status = convert_status(msg);
  int soc = msg->batt_soc;
  if (status == CHARGING_STATUS_FULL) {
    soc = 100;
  }

  snprintf(
    line_buf, sizeof(line_buf) - 1,
    DATE_STAMP_FMT LOG_DELIMITER TIME_STAMP_FMT LOG_DELIMITER
    "%d,%d,%d,%d,%d,%d,%d,%d,%d\n",
    1900 + stm.tm_year, 1 + stm.tm_mon, stm.tm_mday,
    stm.tm_hour, stm.tm_min, stm.tm_sec,
    msg->batt_volt, msg->batt_curr, soc, msg->batt_temp,
    status, msg->key, msg->batt_health, msg->batt_loop_number, msg->powerBoard_status);

  rc = file_access(
    BMS_LOG_FILE, line_buf, strlen(line_buf),
    O_WRONLY | O_APPEND);

  return rc;
}

int convert_status(const bms_response_lcmt * lcm_data)
{
  if (IS_BIT_SET(lcm_data->status, CHARING_BIT)) {
    return CHARGING_STATUS_CHARGING;
  } else if (IS_BIT_SET(lcm_data->status, CHARGED_BIT)) {
    return CHARGING_STATUS_FULL;
  }
  return CHARGING_STATUS_DISCHARGE;
}
