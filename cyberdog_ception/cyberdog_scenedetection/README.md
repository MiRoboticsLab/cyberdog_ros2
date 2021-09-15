# SceneDetection

该模块为机器人提供场景检测功能，通过定位芯片获取当前位置信息，同时通过卫星信号质量判断机器人当前处于室内或室外。

通过SceneDetection server接受系统对本模块的控制命令，通过SceneDetection publisher向系统发布当前位置信息。

## 主要功能

### SceneDetection server

- 接收系统发送的控制命令，命令类型如下：
  - 开启定位
  - 关闭定位


### SceneDetection publisher

- 定期（1S）向系统发布机器人当前位置信息，包含以下内容
	- 经度
	- 纬度
	- 场景判别结果
	
- 过滤非必要的定位芯片上报信息，减少CPU使用率

### 其他

* 完成定位芯片初始化
* 解析定位芯片上报的数据

## 编译

使用了colcon编译系统
*   colcon build --merge-install --install-base /opt/ros2/cyberdog --packages-up-to cyberdog_scene_detection

## 文件

*   ./src/
    *   bream
        *   定位芯片头文件
    *   hal
        *   定位芯片头文件
    *   scene_detection.cpp
        *   实现SceneDetection server与SceneDetection publisher
    *   patch_downloader
        *   定位芯片头文件
*   lib_bream
    *   定位芯片库
*   ./CMakefile.txt
    *   编译脚本
*   ./package.xml
    *   模块声明和依赖声明
