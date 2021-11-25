# CyberDog 整机运动状态感知模块

## **简介**
该模块具体实现了topic为'BodyState'的publisher, 和名为cyberdog_body_state的server. 模块继承于`cyberdog_utils::LifecycleNode`.
BodyState上报posequat和speed_vector两种message的数据（posequat表示整机姿态四元数；speed_vector表示整机运动的瞬时速度，单位：m/s）
整个模块通过cyberdog_utils中的can和sensor与整机头部stm32进行通讯。
