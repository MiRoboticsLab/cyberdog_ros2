CyberDog 前后置LED灯效控制模块

[简介]

该模块用于统一决策系统所有的LED灯效请求，基于ROS2 Service以及sensor_utils类实现，通过定义不同client的LED消息优先级以及timeout来实现对CyberDog头灯和尾灯的控制显示功能.

sensor_utils部分的功能实现描述，参考cyberdog_utils这个project中的描述.

[功能特性]
功能基于SensorDetectionNode.srv这个service功能实现，对client端来说，涉及到的数据包括下面这个数据结构中的4个变量：timeout, priority, clientId, ledId
typedef struct {
  uint64_t timeout;
  uint64_t finishTimestamp;
  uint64_t arriveTimestamp;
  uint8_t  priority;
  uint8_t  clientId;
  uint8_t  ledId;
} led_command;

分别对应SensorDetectionNode中的：
uint8 command
# clientID: BMS,1;BLUETOOTH,2;AUDIO,3; ..., used to decide response priority, smaller means higher.
uint8  clientid
uint8  priority
# timeout uint in nanoseconds, 0 stands for disable, 0xFFFFFFFFFFFFFFFF stands for always on
uint64 timeout

Client在向ROS2 Service发送LED命令时,填充上面的四个字段进入request, clientid根据描述定义填入，command对应为SensorDetectionNode中的
uint8 REAR_LED_OFF = 6
uint8 REAR_LED_RED_ON = 7
uint8 REAR_LED_RED_BREATH = 8
uint8 REAR_LED_RED_BLINK = 9
uint8 REAR_LED_GREEN_ON = 10
uint8 REAR_LED_GREEN_BREATH = 11
uint8 REAR_LED_GREEN_BLINK = 12

uint8 HEAD_LED_OFF = 17
uint8 HEAD_LED_POWER_ON = 18
uint8 HEAD_LED_POWER_OFF = 19
uint8 HEAD_LED_DARKBLUE_ON = 20
uint8 HEAD_LED_SKYBLUE_ON = 21
uint8 HEAD_LED_ORANGE_ON = 22
uint8 HEAD_LED_RED_ON = 23
uint8 HEAD_LED_DARKBLUE_BREATH = 24
uint8 HEAD_LED_SKYBLUE_BREATH = 25
uint8 HEAD_LED_DARKBLUE_BLINK = 26
uint8 HEAD_LED_ORANGE_BLINK = 27
uint8 HEAD_LED_RED_BLINK = 28
这些具体定义, timeout值以纳秒为单位, priority对应下面的这些定义
uint8 TYPE_EFFECTS = 1
uint8 TYPE_FUNCTION = 2
uint8 TYPE_ALARM = 3
其中TYPE_ALARM的优先级最高.

Service端的响应逻辑为: 优先相应最高优先级的command, 对于同样优先级的command, 选取最后一次到达的生效，当前生效命令超时后, 选取上一个同等优先级的命令进行响应, 直到响应队列中不再包含任何新的命令.

对于timeout设置为常亮的case, 常亮命令会一直存在于响应队列中, 直到收到对应的取消命令为止，取消命令需要确保和常亮命令的内容除timeout外的参数取值保持完全一致



