# Xiaomi CyberDog ROS 2

[![License](https://img.shields.io/badge/License-Apache%202.0-orange)](https://choosealicense.com/licenses/apache-2.0/)

![CyberDogDog](https://cdn.cnbj2m.fds.api.mi-img.com/cyberdog-package/packages/doc_materials/cyberdogdog.jpeg)

> ***文档包含简体中文和English***

## 简介 - Introduction

本项目包含小米铁蛋®的ROS 2主要功能包. 

This project contains the main ROS 2 packages of Xiaomi CyberDog®.

## 基本信息 - Basic Information

- 铁蛋默认用户是`mi`, 密码为`123`
- The default user of CyberDog is `mi`, dafault password is `123`
- 使用USB线连接`Download`接口, 可通过`ssh mi@192.168.55.1`连接铁蛋进行内部操作
- You can use a USB cable to connect to the `Download` interface, and use `ssh mi@192.168.55.1` to connect to CyberDog for internal operations

## 软件架构 - Software Architecture 

我们基于ROS 2实现了大部分的机器人应用, 如架构图所示, 包括多设备链接、多模态感知、多模态人机交互、自主决策、空间定位、导航和目标追踪等功能. 目前使用的DDS中间件是`Cyclone DDS`, ROS 2的版本为`Foxy`. 

Most of the robot applications are implemented based on ROS -As shown in the framework diagram, it includes functions such as multi-device connection, multi-modal perception, multi-modal human-computer interaction, autonomous decision-making, localization, navigation, and target tracking. Currently, the DDS middleware in use is `Cyclone DDS`, and the ROS 2 Package is implemented at `Foxy`.

![SoftwareArchitecture](tools/docs/soft_arch.svg)

由于NVIDIA对Jetson系列截至目前（202109）只提供了Ubuntu 18.04的支持, 故我们对Ubuntu 18.04进行了ROS 2的适配和修改. 具体的修改内容可以通过[mini.repos](tools/ros2_fork/mini.repos)进行拉取, 我们去除了部分没必要的仓, 并添加了一些需要使用的仓库（文件是Galcatic版本的, 我们会在后续适配到该版本）. 

Due to NVIDIA only provides support for Ubuntu 18.04 for the Jetson series (202109) as yet, we have adapted and modified ROS 2 for Ubuntu 18.04. The specific modification content can be pulled through [mini.repos](tools/ros2_fork/mini.repos). We have removed some unnecessary repositories, and added some repositories that need to be used (The file is of the Galactic version, and we will adapt to this version in the future).

本项目的详细文档都在各个子模块的根目录里, 如有需要可以直接点击进行了解:

The documentation of this project is in the root directory of each submodule. If necessary, feel free to read the README files.

- 通用类 - Common
  - [cyberdog_bringup](cyberdog_bringup): 启动系统相关, 我们在ROS 2的Launch启动系统上设计了更简约的启动项管理, 对启动脚本（Python3）和启动内容进行了隔离. 通过简单的参数配置, 新的节点或新的进程便可被添加到启动项, 而无需修改脚本内容. 同时也支持参数管理、调试和开关等功能. 
  - [cyberdog_bringup](cyberdog_bringup): Startup program. We designed simpler management on the ROS 2 Launch system and isolated the startup script (Python3) and startup content. Through simple parameter configuration, new nodes or processes can be added to the startup item without modifying the script content. It also supports functions such as parameter management, debugging and switching, etc.
  - [cyberdog_grpc](cyberdog_common/cyberdog_grpc): 机器人与外部通讯的媒介, 目前与手机App进行连接是基于GRPC的. 在未来将支持多机识别和多机通讯. 
  - [cyberdog_grpc](cyberdog_common/cyberdog_grpc): The communication agent between the robot and the real world. Currently, the connection with the mobile app is based on GRPC. In the future, it will support multi-device identification and multi-device communication.
  - [cyberdog_utils](cyberdog_common/cyberdog_utils): 本项目的通用功能仓, 包括基于[cascade_lifecycle](https://github.com/fmrico/cascade_lifecycle)修改的LifecycleNode基类, 和传感器相关节点共用的基类等. 
  - [cyberdog_utils](cyberdog_common/cyberdog_utils): The general function repository of this project. It contains the base class of LifecycleNode [cascade_lifecycle](https://github.com/fmrico/cascade_lifecycle) and the base class shared with sensor-related nodes, etc.
  - [media_vendor](cyberdog_common/media_vendor): 多媒体相关应用需要使用的`CMake`配置项. 
  - [media_vendor](cyberdog_common/media_vendor): The `CMake` configuration items that multimedia related applications need to use.
  - [toml11_vendor](cyberdog_common/toml11_vendor): [toml11](https://github.com/ToruNiina/toml11)的桥接包. 
  - [toml11_vendor](cyberdog_common/toml11_vendor): Bridge package of [toml11](https://github.com/ToruNiina/toml11).
- 感知类 - Perception
  - [cyberdog_bms](cyberdog_ception/cyberdog_bms): CyberDog上的电池管理模块, 主要负责电池信息的接收与分发. 
  - [cyberdog_bms](cyberdog_ception/cyberdog_bms): The battery management module on CyberDog. It is mainly responsible for receiving and distributing battery information.
  - [cyberdog_body_state](cyberdog_ception/cyberdog_body_state):该模块实现了整机运动状态的感知功能, 并通过BodyState上报posequat和speed_vector两种message的数据（posequat表示整机姿态四元数；speed_vector表示整机运动的瞬时速度, 单位:m/s）. 
  - [cyberdog_body_state](cyberdog_ception/cyberdog_body_state): This module realizes the perception function of the whole machine motion state, and reports the data of posequat and speed_vector through BodyState (posequat represents the posture quaternion of the robot; speed_vector represents the instantaneous speed of the robot motion, unit: m/s).
  - [cyberdog_lightsensor](cyberdog_ception/cyberdog_lightsensor): 该模块创建了机器人感知系统中Light Sensor的service和publisher, 当机器人需要感知周围环境光的强度时, 可通过上层决策层启动该service和publisher, 获取环境光照度信息. 
  - [cyberdog_lightsensor](cyberdog_ception/cyberdog_lightsensor): This module creates the service and publisher of the Light Sensor in the robot perception system. When the robot needs to perceive the intensity of the ambient light, it can start the service and publisher through the upper decision making layer to obtain the ambient light information.
  - [cyberdog_obstacledetection](cyberdog_ception/cyberdog_obstacledetection): 该模块创建了机器人感知系统中Ultrasonic Sensor的service和publisher. 
  - [cyberdog_obstacledetection](cyberdog_ception/cyberdog_obstacledetection): This module creates the service and publisher of Ultrasonic Sensor in the robot perception system.
  - [cyberdog_scenedetection](cyberdog_ception/cyberdog_scenedetection): 该模块为机器人提供场景检测功能, 通过定位芯片获取当前位置信息, 同时通过卫星信号质量判断机器人当前处于室内或室外. 
  - [cyberdog_scenedetection](cyberdog_ception/cyberdog_scenedetection): This module provides a scene detection function for the robot. Obtain the current position information through the position chip. Determine whether the robot is currently indoors or outdoors based on the quality of the satellite signal.
 
- 交互类 - Interaction
  - [cyberdog_audio](cyberdog_interaction/cyberdog_audio)
    - [audio_assistant](cyberdog_interaction/cyberdog_audio/audio_assistant): 该模块集成了小米小爱SDK的功能, 并作为ROS 2节点桥接了这部分功能. 
    - [audio_assitant](cyberdog_interaction/cyberdog_audio/audio_assistant): This module integrates the functions of Xiaomi Xiaoai SDK and bridges these functions as a ROS 2 node.
    - [audio_base](cyberdog_interaction/cyberdog_audio/audio_base): 该模块提供播放多段或单段不定长PCM音频数据和wav格式音频文件的方法, 其底层实现为SDL2与SDL_mixer. 
    - [audio_base](cyberdog_interaction/cyberdog_audio/audio_base): This module is used to play multi-segment or single-segment variable length PCM audio data,  audio files with wav format. Its bottom layer is implemented as SDL2 and SDL_mixer.
    - [audio_interaction](cyberdog_interaction/cyberdog_audio/audio_interaction): 该模块实现了整个音频模块的交互部分, 主要包括与其它模块之间关于通用播放的交互、语音助手相关控制与交互功能和与APP(grpc)之间关于音量调解的交互功能. 
    - [audio_interaction](cyberdog_interaction/cyberdog_audio/audio_interaction): This module is used to implement the interactive part of the audio module, which mainly includes the interaction with other modules on common playback, the control and interaction functions of the voice assistant, and the interaction function with APP (grpc) about volume adjustment.
    - [cyberdog_audio](cyberdog_interaction/cyberdog_audio/cyberdog_audio): 音频功能包的完整包, 负责整合所有功能. 
    - [cyberdog_audio](cyberdog_interaction/cyberdog_audio/cyberdog_audio): The complete package of the audio function package is responsible for integrating all functions.
    - [xiaoai_sdk_vendor](cyberdog_interaction/cyberdog_audio/xiaoai_sdk_vendor): 该模块提供了小爱SDK的完整打包, 并集成ROS 2的编译体系. 
    - [xiaoai_sdk_vendor](cyberdog_interaction/cyberdog_audio/xiaoai_sdk_vendor): This module provides a complete package of Xiaoai SDK and integration the ROS 2 compilation system.
  - [cyberdog_camera](cyberdog_interaction/cyberdog_camera)
    - [cyberdog_camera](cyberdog_interaction/cyberdog_camera/cyberdog_camera): 该模块基于NVIDIA Argus和ROS 2实现了相机的基础功能, 包括拍照和录像等, 并为其他模块（视觉SDK、图传等）提供调用接口. 
    - [cyberdog_camera](cyberdog_interaction/cyberdog_camera/cyberdog_camera): This module implements the basic functions of the camera based on NVIDIA Argus and ROS 2, including taking pictures and videos, and provides a calling interface for other modules (visual SDK, image transmission, etc.).
    - [cyberdog_vision](cyberdog_interaction/cyberdog_camera/cyberdog_vision): 该模块集成了小米AI视觉的功能, 包括人脸、人体和手势识别, 并编译成库供使用. 
    - [cyberdog_vision](cyberdog_interaction/cyberdog_camera/cyberdog_vision): This module integrates the functions of Xiaomi AI vision, including face, human body and gesture recognition, and is compiled into a library for use.
    - [vision_sdk_vendor](cyberdog_interaction/cyberdog_camera/vision_sdk_vendor): 该模块提供了小米AI视觉SDK的完整打包, 并集成ROS 2的编译体系. 
    - [vision_sdk_vendor](cyberdog_interaction/cyberdog_camera/vision_sdk_vendor): This module provides a complete package of Xiaomi AI Vision SDK and integration the ROS 2 compilation system.
  - [cyberdog_led](cyberdog_interaction/cyberdog_led): 该模块用于统一决策系统所有的LED灯效请求, 基于ROS2 Service以及sensor_utils类实现, 通过定义不同client的LED消息优先级以及timeout来实现对CyberDog头灯和尾灯的控制显示功能. 
  - [cyberdog_led](cyberdog_interaction/cyberdog_led): This module is used to unify all the LED lighting requests of the decision-making system. It is based on ROS2 Service and sensor_utils. It realizes the control and display function of CyberDog headlights and taillights by defining the LED message priority and timeout of different clients.
  - [cyberdog_livestream](cyberdog_interaction/cyberdog_livestream): 图传模块的库. 
  - [cyberdog_livestream](cyberdog_interaction/cyberdog_livestream): Library of video stream transmission modules.
  - [cyberdog_touch](cyberdog_interaction/cyberdog_touch): 该模块提供了topic为TouchState的publisher, 继承于cyberdog_utils::LifecycleNode. 目前支持单指单击报 `LPWG_SINGLETAP_DETECTED` 和单指长按3s报`LPWG_TOUCHANDHOLD_DETECTED`事件. 
  - [cyberdog_touch](cyberdog_interaction/cyberdog_touch): This module provides a publisher of topic TouchState, and is a subclass of  cyberdog_utils::LifecycleNode. Currently, it supports to report `LPWG_SINGLETAP_DETECTED` event with a single finger to single-click and report `LPWG_TOUCHANDHOLD_DETECTED` event with a single finger to hold for 3 seconds.
  - [cyberdog_wireless](cyberdog_interaction/cyberdog_wireless)
    - [bluetooth](cyberdog_interaction/cyberdog_wireless/bluetooth): 该模块实现了:注册蓝牙GATT service, 发送BLE广播, 等待手机APP连接成功后, 接收APP发送的Wifi信息, 实现Wifi的联网操作, 并返回联网结果, 以及通过APP提供的命令, 实现蓝牙手柄的扫描, 连接, 达到通过手柄控制CyberDog的目的. 
    - [bluetooth](cyberdog_interaction/cyberdog_wireless/bluetooth): The module realizes: register Bluetooth GATT service; send BLE broadcast; after waiting for the mobile APP to connect successfully, receive the Wifi information sent by the APP; realize the Wifi networking operation; return the networking results; realize the Bluetooth handle through the commands provided by the APP Scan and connect. Achieve the purpose of controlling CyberDog through the handle.
    - [wifirssi](cyberdog_interaction/cyberdog_wireless/wifirssi): 该模块的主要功能是创建定时任务, 实时查询当前已连接Wifi的信号强度, 并通过ros2的topic通讯进行传输. 
    - [wifirssi](cyberdog_interaction/cyberdog_wireless/wifirssi): The main function of this module is to create a timed task, query the signal strength of the currently connected Wifi in real time, and transmit it through the topic of ros2.
- 决策类 - Decision
  - [cyberdog_decisionmaker](cyberdog_decision/cyberdog_decisionmaker): 面向业务层, 负责实现具体业务功能. 目前提供了`automation_manager`、`ception_manager`、`interaction_manager`和`motion_manager`. 分别用于自动化功能、感知功能、人机交互功能和运动功能的管理和决策. 该四个模块均继承`cascade_manager`, 并在基础上根据业务功能稍作改动. 
  - [cyberdog_decisionmaker](cyberdog_decision/cyberdog_decisionmaker): Towards business layer, responsible for realizing specific business functions. Currently it provides `automation_manager`, `ception_manager`, `interaction_manager` and `motion_manager`, respectively used for automation functions, perception functions, human-computer interaction functions, management and decision making of motion functions. These four modules all are subclasses of the `cascade_manager` and make slight changes based on the business functions.
  - [cyberdog_decisionutils](cyberdog_decision/cyberdog_decisionutils): 决策相关功能的基类和工具类集合, 负责实现通用功能. 目前提供了`cascade_manager`, 该模块继承于`cyberdog_utils::LifecycleNode`, 具备级联/并联和单点控制的功能, 可以快速管控其作用域下的节点的启动和关闭. 
  - [cyberdog_decisionutils](cyberdog_decision/cyberdog_decisionutils): A collection of base classes and tool classes for decision making, responsible for implementing general functions. Currently, it provides `cascade_manager`, which inherits from `cyberdog_utils::LifecycleNode` and has the functions of cascade/parallel connection and single point of control. It can control the startup and shutdown of nodes under its scope quickly.
- 接口类 - Interface
  - [cyberdog_interfaces](cyberdog_interfaces/cyberdog_interfaces): 接口总抽象包. 
  - [cyberdog_interfaces](cyberdog_interfaces/cyberdog_interfaces): Interfaces total abstract package.
  - [automation_msgs](cyberdog_interfaces/automation_msgs): 自动化功能相关接口. 
  - [automation_msgs](cyberdog_interfaces/automation_msgs): Interfaces related to automation functions.
  - [cascade_lifecycle_msgs](cyberdog_interfaces/cascade_lifecycle_msgs): 级联节点相关接口. 
  - [cascade_lifecycle_msgs](cyberdog_interfaces/cascade_lifecycle_msgs): Interfaces of cascading nodes.
  - [ception_msgs](cyberdog_interfaces/ception_msgs): 感知功能相关接口. 
  - [ception_msgs](cyberdog_interfaces/ception_msgs): Interfaces related to perception function.
  - [interaction_msgs](cyberdog_interfaces/interaction_msgs): 交互功能相关接口. 
  - [interaction_msgs](cyberdog_interfaces/interaction_msgs): Interfaces related to interactive functions.
  - [lcm_translate_msgs](cyberdog_interfaces/lcm_translate_msgs): 定制化的LCM消息定义. 
  - [lcm_translate_msgs](cyberdog_interfaces/lcm_translate_msgs): Customized LCM Messages.
  - [motion_msgs](cyberdog_interfaces/motion_msgs): 运动功能相关接口. 
  - [motion_msgs](cyberdog_interfaces/motion_msgs): Interfaces related to motion functions.

## 前置条件 - Precondition

如在交叉环境进行编译, 可参考[交叉编译铁蛋源码](TBD)进行了解环境的配置. 

If you are compiling in a cross environment, you can refer to [Cross Compiling CyberDog Source Code](TBD) to understand the environment configuration.

如在目标设备上直接编译, 需要保证已连接互联网. 首选环境是铁蛋, 次选环境是`NVIDIA Jetson系列`的开发环境. 

If you are directly compiling on the target device, you need to ensure that you have connected to the Internet. The preferred environment is CyberDog, and the second choice is the development environment of the `NVIDIA Jetson series`.

如是后者, 需要保证安装:

If the latter, you need to ensure the following packages are installed:

- ROS 2: 最小必须. 并且需要至少包含[mini.repos](tools/ros2_fork/mini.repos)中的功能包. 
- ROS 2 (Foxy): Required for construction of minimal package. And it needs to contain at least the package [mini.repos](tools/ros2_fork/mini.repos).
- LCM:最小必须. 可通过下载源码编译安装. 
- LCM: Required for construction of minimal package. It can be compiled and installed by downloading the source code.
- 媒体库: mpg123,SDL2和SDL2_Mixer: 基础必须. 最好通过源码安装, 并确保按照前面所写的顺序编译安装. 
- Media libraries: mpg123, SDL2 and SDL2_Mixer: Required for construction of basic package. It is best to install through source code and make sure to compile and install in the order written above.
- NV相关库: 基础必须. 可`nvidia-l4t-jetson-multimedia-api`和`cuda-compiler-10-2`. 
- NV-related libraries: Required for construction of basic package. You can use `nvidia-l4t-jetson-multimedia-api` and `cuda-compiler-10-2`.
- 其他库: OpenCV和librealsense. 
- Other libraries: OpenCV and librealsense.

## 构建 & 部署 - Build & Deploy

本项目支持两种构建策略:

This project supports two construction strategies:

- 最小功能包: 只编译影响整机启动和运动的相关功能包. 
- Minimal package: Only compile the relevant packages that affect the startup and motion of the whole machine.
- 基础功能包: 编译本仓（cyberdog_ros2）的全部功能包. 
- Basic package: Compile all packages of this repository(cyberdog_ros2).

### 最小功能包的构建 - Construction of Minimal Package

编译方法:

Compilation process:

- 下载`cyberdog_ros2`. 
- Download `cyberdog_ros2`.

```
$ mkdir -p ros_apps/src
$ cd ros_apps/src
$ git clone https://github.com/MiRoboticsLab/cyberdog_ros2.git
$ cd ..
```

- 使用`--packages-up-to`编译（[确保source过ROS 2的环境变量](TBD)）
- Use `--packages-up-to` to compile（[Ensure the ROS 2 environment is sourced](TBD)）

```
$ colcon build --merge-install --packages-up-to cyberdog_bringup
```

或者, 编译到指定目录, 注意: 如有需要请替换`/opt/ros2/cyberdog`的值为其他. 

Or, compile to the specified directory. Note: If necessary, please replace `/opt/ros2/cyberdog` with your path.

```
$ export OUTPUT_DIR=/opt/ros2/cyberdog
$ colcon build --merge-install --install-base $OUTPUT_DIR --packages-up-to cyberdog_bringup
```

### 基础功能包 - Basic package

编译方法:

Compilation process:

- 下载`cyberdog_ros2`. 
- Download `cyberdog_ros2`.

```
$ mkdir -p ros_apps/src
$ cd ros_apps/src
$ git clone https://github.com/MiRoboticsLab/cyberdog_ros2.git
$ cd ..
```

- 直接编译所有的包（[确保source过ROS 2的环境变量](TBD)）

- Compile all packages ([Ensure the ROS 2 environment is sourced](TBD))

```
$ colcon build --merge-install
```

或者, 编译到指定目录, 注意:如有需要请替换`/opt/ros2/cyberdog`的值为其他. 

Or, compile to the specified directory. Note: If necessary, please replace `/opt/ros2/cyberdog` with your path.

```
$ export OUTPUT_DIR=/opt/ros2/cyberdog
$ colcon build --merge-install --install-base $OUTPUT_DIR
```

### 通用的部署方式 - General deployment method

如果使用的是`/opt/ros2/cyberdog`路径进行编译, 且环境是铁蛋, 重启机器或服务即可部署完毕. 

If you use the `/opt/ros2/cyberdog` path to compile and the environment is cyberdog, restart the machine or service to complete the deployment.

重启服务的方式:

To restart the service:

```
$ sudo systemctl restart cyberdog_ros2.service
```

## 相关项目 - Related Projects

- [CyberDog_Ctrl](https://github.com/Karlsx/CyberDog_Ctrl):使用GRPC控制铁蛋 - Control your CyberDog with GRPC

## 相关资源 - Related Resources

- [CyberDogAPP下载链接 - CyberDogAPP download link](http://cdn.cnbj1.fds.api.mi-img.com/ota-packages/apk/cyberdog_app.apk)
- [铁蛋躯干Step文件 - Step file of CyberDog body](https://cdn.cnbj2m.fds.api.mi-img.com/cyberdog-package/packages/doc_materials/cyber_dog_body.stp)

## 向铁蛋贡献力量！ - Contribute to CyberDog!

浏览页面[CONTRIBUTING.md](CONTRIBUTING.md)了解如何向铁蛋贡献力量！

Go through the page [CONTRIBUTING.md](CONTRIBUTING.md) to learn how to contribute to CyberDog!
