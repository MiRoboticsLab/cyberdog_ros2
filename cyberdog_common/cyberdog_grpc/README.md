### cyberdog_grpc 机器狗与应用通信模块简介
本模块是一个通信模块，实现了GPRC server，GRPC client以及ROS2 节点的角色，本模块可以
通过GRPC server接收来自于应用的请求，并将请求转化为ROS2系统的消息发送给相应的ROS2节点完成请求处理。
通过ROS2 topic订阅者来完成ROS2系统内的信息收集，并通过GRPC client发送给应用。
通过本模块，运行在手机端的应用可以完成对Cyberdog的控制，并完成对Cyberdog信息的手机以向用户展示。

## cyberdog_grpc 功能介绍

* 接收手机端应用的请求并发送到Cyberdog系统中的对应ROS2节点处理
    *  运动控制消息
        *  切换狗的模式，比如手动模式，跟随模式，锁定模式等
        *  切换狗的步态，比如小跑，缓跑，跳跑等
        *  发送狗的运动控制信息，比如X轴速度，Y轴速度等
    *  相机和导航相关请求
        * 人脸录入请求
        * 跟谁模式请求
        * AB点导航请求
    *  音频请求
        *   获取声纹请求
        *   音量设置请求

* 发送狗的信息到手机端应用
    *  WIFI信号
    *  电池电量
    *  狗当前的状态
    *  导航状态
    *  地图信息
    *  声纹结果

## 编译
使用colcon编译系统

*   需要GPRC环境，狗内已经内置. 更多信息请访问 https://github.com/grpc/grpc.
*   需要ROS2环境，狗内已经内置. 更多信息请访问 https://github.com/ros2.
*   编译命令: colcon build --merge-install --install-base /opt/ros2/cyberdog --packages-up-to cyberdog_cyberdog_app.

## 目录树
*   ./include/
    *   cyberdog_app_client.hpp
        *   GRPC 客户端代码头文件
    *   cyberdog_app_server.hpp
        *   GRPC 服务器端代码头文件
    *   cyberdog_app.hpp
        *   Ros2 节点代码头文件
    *   msgdispatcher.hpp
        *   线程安全的消息分发类，用户分发GRPC消息到手机端应用
    *   net_avalible.hpp
        *   网络是否可达监测类，用于定期检查网络状态。
    *   threadsafe_queue.hpp
        *   线程安全的队列，用于处理ROS2 Action的结果
*   ./src/
    *   action_clients.cpp
        *   Action 请求的发起类，用于向ROS2系统发起Action请求，并处理返回结果   
    *   cyberdog_app_client.cpp
        *   GRPC 客户端程序，cyberdog_app.cpp通过该类完成ROS2消息到手机端应用的发送
    *   cyberdog_app_server.cpp
        *   GRPC 服务端程序，用于接收手机端应用发送的请求并转化处ROS2请求，最终通过cyberdog_app.cpp完成请求处理
    *   cyberdog_app.cpp
        *   ROS2 节点的实现类，该类完成对手机端应用所需要的信息的监听，并通过GRPC客户端发送到手机端APP
        *   实现了GRPC 服务端的处理代码，将从手机端应用发送到狗的请求分发给ROS2系统内的对应模块，完成请求处理并返回结果
    *   main.cpp
        *   程序入口，启动了ROS2节点
*   ./CMakefile.txt
    *   编译脚本
*   ./package.xml
    *   模块依赖及自身信息
