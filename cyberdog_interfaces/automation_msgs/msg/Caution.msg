# Message denote nav state
#
#
# Nav Mode 
#
uint8 NAV_NORMAL        = 3
uint8 NAV_RELOCATING    = 0
uint8 NAV_TARGET_REACH  = 5

#
#
# Error Type
#
# 11 - robot trapped           # 机器人被困
# 12 - load map fail           # 地图读取失败
# 13 - build map fail          # 地图建立失败
# 14 - slam fail               # slam定位失败
# 15 - path planning fail      # 路径规划失败
# 16 - body tracking fail      # 人体跟踪失败
# 17 - obstacle block          # 障碍物无法通过
# 18 - realsense no data       # realsense没有数据
#
uint8 NO_ERROR             = 10
uint8 ROBOT_TRAPPED        = 11
uint8 MAP_LOAD_FAILED      = 12
uint8 MAP_BUILD_FAILED     = 13
uint8 SLAM_FAILED          = 14
uint8 PATH_PLAN_FAILED     = 15
uint8 BODY_TRACKING_FAILED = 16
uint8 OBSTACLE_BLOCK       = 17
uint8 REALSENSE_FAILED     = 18
#
uint8 robot_mode
uint8 error_type
