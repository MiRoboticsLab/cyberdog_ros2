## BMS模块简介

BMS模块是Cyberdog上的电池管理模块，主要负责电池信息的接收与分发。
## 主要功能
* 通过LCM接收来自与运动控制板的BMS消息，消息主要包括
    *  电池电量
    *  充电状态
    *  充电电流
    *  电池温度
    *  电池健康程度
    *  电池充放电循环次数
* 作为一个ROS2节点，利用ROS2 Service向其他节点提供BMS信息
    *  其他节点作为Client向本模块发送请求
* 依据充电状态控制头部和尾部LED灯
    *  电池电量低，LED闪烁
    *  充电过程中，LED呼吸
    *  充电完成，LED常亮
* 依据电池状态播放对应语音
    *  单击狗头时播放当前电量
    *  低电时进行语音播报
    *  开始充电时播报开始充电
    *  播报关机音量
* 保存充电信息到文件系统
* 保存充电信息到临时文件系统

## 编译
使用了colcon编译系统
*   colcon build --merge-install --install-base /opt/ros2/cyberdog --packages-up-to cyberdog_bms

## 文件
*   ./include/
    *   bms_common.hpp
        *   通用头文件
    *   time_interval.hpp
        *   用于检查时间间隔的类
*   ./src/
    *   bms_recv.cpp
        *   接收来自底层的LCM信息，转化并处理信息，包括声音控制，灯效控制，关机控制等
    *   bms_logger.cpp
        *   日子保存类
    *   bms_control.cpp
        *   LCM消息发送类
    *   bms_test.cpp
        *   测试代码
*   ./CMakefile.txt
    *   编译脚本
*   ./package.xml
    *   模块声明和依赖声明
