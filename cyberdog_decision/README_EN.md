# CyberDog Decisionmaker

## **[Introduction]**
---
This project contains two sub-projects: `decision_maker` and `decision_utils`. `decision_utils` is a collection of base classes and tool classes for decision-related functions and is responsible for implementing general functions; the former is oriented to the business layer and is responsible for implementing specific business functions.

Up to now：

`decision_utils` provides `cascade_manager`, this module inherits from `cyberdog_utils::LifecycleNode`, has the functions of cascade/parallel connection and single-point control, and can quickly control the startup and shutdown of nodes under its scope.

`automation_manager`, `ception_manager`, `interaction_manager` and `motion_manager` are provided in `decision_maker`. They are used for the management and decision-making of automation functions (not yet autonomous), perception functions, human-computer interaction functions, and sports functions. The four modules all inherit the `cascade_manager` and make slight changes based on the business functions.

## **[Features]**
---
### **Cascade Manager**

#### **Introduction**

Cascade manager is implemented in [decision_utils](decision_utils/src/cascade_manager.cpp). The purpose is to realize a function that can manage the state of its child nodes in parallel through self-start and self-pause. It is inherited from `cyberdog_utils::LifecycleNode`, the principle of this node can be jumped to [cyberdog_utils](../cyberdog_common/cyberdog_utils) for understanding.

Cascade manager implements `manager_configure`, `manager_activate`, `manager_deactivate`, `manager_cleanup`, `manager_shutdown` and `manager_error` in order according to the mechanism of `ROS 2` life cycle node, and the nodes that need to inherit it are in the corresponding The transition state calls the corresponding function. Support parameter configuration.

#### **Parameters Configuration**

[] is the variable name.

- The name of the cascading node list [node_list_name_], the parameter name is determined according to the input string.
- Manager timeout time [timeout_manager_], the parameter name is `timeout_manager_s`.

#### **Basic Mechanism**

The cascade manager distinguishes its own function types by identifying external parameters. There are currently three types of functions implemented, namely: `single manager mode`, `single list management mode` and `multiple list management mode`.

- Single manager mode: There are no child nodes that need to be connected, and all operations only need to consider the own node.
- Single list management mode: When the node is configured, the external parameter (YAML) list will be read through the cascading node list name, and the corresponding node name will be added in turn according to the list, and the status will be synchronized. And during the activation and suspension operations, it will check whether each node in the list has reached the corresponding state, and update the `chainnodes_state_` variable in time.
- Multi-list management mode: similar to single-list, when the node is configured, it will be read through the external parameter list, but it can be read by entering a different parameter list during configuration. The use of this management mode needs to be leveled when instantiating Copy the name of the associated node list to `multi` to activate the multi-list management mode, and pass the correct string of the list name to the `manager_configure` function during configuration.

All nodes that inherit the cascade manager can use `message_info`, `message_warn` and `message_error` for ROS-related record output, and input the content in the format of `std::string` (use `+` to combile strings if multiple strings are needed).

### **Automation Management**

#### **Introduction**

At present, this node is only responsible for activating and suspending automation related nodes, including the nodes specially needed in `Explor` mode and `Track` mode. Support parameter configuration.

#### **Parameters Configuration**

- Explore the node list of mapping mode [explor_map_node_list_name_], the parameter name is determined according to the input string.
- Explore the navigation mode node list [explor_nav_node_list_name_], the parameter name is determined according to the input string.
- Tracking mode node list [track_node_list_name_], the parameter name is determined according to the input string.

### **Interoception & Exteroception Management**

#### **Introduction**

At present, this node is responsible for activating and suspending perception related function nodes, and is responsible for collecting important perception information, fusing perception information, and processing and classifying the information.

#### **Parameters Configuration**

- Tick ​​frequency [rate_tik_], the parameter name is `rate_tik_hz`.
- Low battery threshold [soc_limit_], the parameter name is `soc_limit_perc`.

#### Safety Guardian

At present, the security protection function detects the robot status at the frequency of rate_tik_, and currently only supports the identification of low battery status, and broadcasts notifications to external nodes. New features will be added in the future.

### **Human-computer Interaction Management**

#### **Introduction**

At present, this node is responsible for activating and suspending human-computer interaction related function nodes.

### **Motion Management**

#### **Introduction**

At present, this node is responsible for all movement-related instruction collection, instruction processing and instruction issuance, all movement-related state collection, state processing and state broadcasting, as well as all sports-related Parameters Configuration and functional configuration.

- Instructions include: mode checking, gait checking, order execution and movement instructions.
- Status includes: current mode, gait, timely gait, order, speed, pose and parameters, etc.
- Parameters include: gait parameters and intra-node parameters.

Since the current motion control level uses `Lightweight Communications and Marshalling (LCM)` for communication, this module will also call LCM related interfaces for communication.

Customized orders will be designed through `Tom's Obvious, Minimal Language (TOML)`. At present, this module is designed by motion control developers through the framework of `Mini Cheetah` and provides `txt` format. We designed A `Python` script automatically converts `txt` to `TOML`. *In the future, the function of automatically generating order `TOML` files will be developed. Developers do not need to write these files manually. *

Since the robot has only one main body, the following `mode check`, `gait check` and `order execution` all have preemptive functions.

#### **Intra-node Parameters Configuration**
##### Frequency
- common rate[rate_common_] -- `rate_common_hz`
- control rate[rate_control_] -- `rate_control_hz`
- output rate[rate_output_] -- `rate_output_hz`
- waiting rate[rate_wait_loop_] -- `rate_wait_loop_hz`
- odom rate[rate_odom_] -- `rate_odom_hz`
- LCM broadcast rate constant[rate_lcm_const_] -- `rate_lcm_const_hz`
##### Timeout
- manager timeout[timeout_manager_] -- `timeout_motion_ms`
- motion timeout[timeout_motion_] -- `timeout_manager_s`
- gait checking timeout[timeout_gait_] -- `timeout_gait_s`
- order executing timeout[timeout_order_] -- `timeout_order_ms`
- LCM receive timeout[timeout_lcm_] -- `timeout_lcm_ms`
##### Value
- x-axis speed extreme value[cons_abs_lin_x_]
- y-axis speed extreme value[cons_abs_lin_y_]
- x-axis angular velocity extreme value[cons_abs_ang_r_]
- y-axis angular velocity extreme value[cons_abs_ang_p_]
- z-axis angular velocity extreme value[cons_abs_ang_y_]
- z-axis angular acceleration extreme value[cons_abs_aang_y_]
- maximum body height[cons_max_body_]
- maximum plantar height[cons_max_gait_]
- default body height[cons_default_body_]
- default gait height[cons_default_gait_]
- default moving linear speed[cons_speed_l_normal_]
- default angular velocity of movement[cons_speed_a_normal_]
##### Ratio
- low battery control ratio[scale_low_btr_]
##### Connection
- motion receive channel[ttl_recv_from_motion_]
- motion send channel[ttl_send_to_motion_]
- odom receive channel[ttl_from_odom_]
- motion receive port[port_recv_from_motion_]
- motion send port[port_send_to_motion_]
- odom receive port[port_from_odom_]

#### **Motion Parameters Configuration**

Data format: reference[motion_msgs](../cyberdog_interfaces/motion_msgs/README.md)

Motion parameter modification currently only supports dynamic modification of body height and plantar height.

Legitimate request: motion parameter with a newer time stamp.

- The newer timestamp means that the timestamp of this parameter modification is newer than the timestamp of the last parameter modification. Generally, the current system time in the same time zone can be used.
- The parameter range is within the constraints of `cons_max_body_` and `cons_max_gait_` above, and is a positive number.



#### **Mode Checking**

Data format: reference[motion_msgs](../cyberdog_interfaces/motion_msgs/README.md)

Legitimate request: a mode with a newer timestamp.
- The newer timestamp means that the timestamp of this switching request is newer than the timestamp of the last mode switching. Generally, the current system time in the same zone can be used.
- The mode needs to ensure that it meets the preset options. The mode switch supports the default mode (MODE_DEFAULT), lock mode (MODE_LOCK), manual mode (MODE_MANUAL), exploration mode (MODE_EXPLOR) and tracking mode (MODE_TRACK). The modes are divided into two fields of `control_mode` and `mode_type` to distinguish them. The `mode_type` of the first three modes are all `DEFAULT_TYPE(0)`, and the latter two modes can identify different sub-modes according to `mode_type`.

Preemption: Newly initiated legal mode switching requests can preempt the old mode switching requests that are running, that is, the new priority is higher. The same mode (both fields are the same) does not preempt.

Cancellation: the handle that initiated the application can initiate cancellation at any time.

Feedback: Once a legal request is received, the mode Action service will detect the current switching status at the frequency of `rate_common_`Hz, and return the current switching status at this frequency.

Result:
- If succeed: Update the value of `modestamped` in `robot_control_state_`.
- If failed: The previous state will be restored. If the gait has been switched, the previous gait will be restored (except for being preempted); if other nodes have been activated, other nodes will be suspended.

#### **Gait Checking**

Data format: reference[motion_msgs](../cyberdog_interfaces/motion_msgs/README.md)

Legitimate request: a gait with a legal motivation and a newer timestamp
- Legal motivation means that the value of `motivation` must conform to the default value `cyberdog_utils::GaitChangePriority`
- The newer timestamp means that the timestamp of this switching request is newer than the timestamp of the last gait switching. Generally, the current system time in the same time zone can be used.
- The gait needs to ensure that it meets the preset options. The gait currently supports power-off (GAIT_PASSIVE), slow down (GAIT_KNEEL), resume standing (GAIT_STAND_R), posture standing (GAIT_STAND_B), slow walking (GAIT_WALK), low speed trotting (GAIT_SLOW_TROT) ), medium-speed trot (GAIT_TROT), fast trot (GAIT_FLYTROT), double-leg bouncing (GAIT_BOUND) and four-leg bouncing (GAIT_PRONK).

Preemption: first consider motivation (priority), and secondly consider gait.
- According to the value of motivation (uint8), from large to small, the priority is from high to low.
- If the gait of the same priority is different, the new request preempts the old request.

Cancellation: the handle that initiated the application can initiate cancellation at any time.

Interrupt: If the mode changes during the gait switching, the current gait switching will be stopped immediately.

Feedback: Once a legal request is received, the gait action service will detect the current switching gait at the frequency of `rate_common_`Hz, and return to the current switching situation at this frequency.

Result：
- If succeed: update the value of `gaitstamped` in `robot_control_state_`.
- If failed: Restore the default value of `gait_cached_`.

#### **Order Execution**

Data format: reference[motion_msgs](../cyberdog_interfaces/motion_msgs/README.md)

Legitimate request: Order with a newer timestamp and its parameters (if required)
- The newer timestamp means that the timestamp of this execution request is newer than the timestamp of the last order execution. Generally, the current system time in the same time zone can be used.
- The order needs to comply with the preset options. The order currently supports standing up (MONO_ORDER_STAND_UP), lying down (MONO_ORDER_PROSTRATE), backing (MONO_ORDER_STEP_BACK), turning in place (MONO_ORDER_TURN_AROUND), shaking hands (MONO_ORDER_HI_FIVE), dancing (MONO_ORDER_DANCE), welcome（MONO_ORDER_WELCOME）,scroll (MONO_ORDER_TURN_OVER) and sit down (MONO_ORDER_SIT).
- Among them, the parameters can be set for backing and turning in place, that is, the value given by the `para` field is valid. The effect of the order execution will be directly related to the parameters.

Preemption: Except for the two orders of welcome and scroll, other orders can be preempted by new order execution requests.

Cancellation: Except for the two orders of New Year greetings and scrolling, other orders can be cancelled at any time by the handle that initiated the application.

Interruption: Except for the two orders of New Year's greetings and rolling, other orders can be directly interrupted by gait switching requests, mode switching requests and movement instructions.

External tuning: The current orders are divided into three categories:
- ROS layer direct encapsulation: use existing gait for operation.
- The ROS layer sets the parameter table, and the motion control layer receives the parameter table and then parses the parameters in turn and executes it: create a parameter file of `TOML`, and send the parameter table after reading the file. *The current method of establishing and tuning the parameter table requires engineers to manually adjust after experimentation. In the future, an automated programming software will be designed. *
- The ROS layer only issues a specific ID, and the motion control layer is hard-coded and executed directly: the bottom layer is hard-coded, and two orders that cannot be interrupted and preempted are the mechanism.

Feedback: Once a legal request is received, the order execution service will detect the progress of the order execution at the frequency of `rate_common_`Hz (from 0 to 100), and return the current execution at this frequency, and update the `orderstamped` of `robot_control_state_` at the same time.

Result：
- If succeed：Update the value of `id` in `robot_control_state_` to `MONO_ORDER_NULL`, restore the state, and return success.
- If failed：Update the value of `id` in `robot_control_state_` to `MONO_ORDER_NULL`, restore the state, and return failure.

#### **Movement Instruction**

Data format: reference[motion_msgs](../cyberdog_interfaces/motion_msgs/README.md)

Legitimate request: Movement instruction with appropriate source id and specified frame id, and with a newer timestamp
- Appropriate source id: In different modes, the movement command accepted by the decision-making end will be distinguished by the value of source id. As shown below, the macro definition is taken from motion_msgs/msg/SE3VelocityCMD.msg.
  - Remote control mode: accept INTERNAL and REMOTEC
  - Exploration mode: accept INTERNAL, REMOTEC and NAVIGATOR
  - Tracking mode: accept INTERNAL and NAVIGATOR
- Specified frame id: The frame id of all movement commands must be BODY_FRAME (taken from motion_msgs/msg/Frameid.msg)
- The newer timestamp means that the timestamp of this command initiated is newer than the timestamp of the last movement command. Generally, the current system time in the same time zone can be used.

## **Run & Debug**
---
This module supports independent operation and startup system operation, and can be combined with GRPC or handle two control modes.

### **Stand-alone Running**

Stand-alone operation means using `ros2 run` to start, which is often used for single-function testing and single-function debugging.

The complete command is：

```Bash
$ ros2 run cyberdog_decisionmaker decisionmaker
```

In this starting state, since there are no external parameters, only motion-related functions can be tested.

### **Launch Running**

When the system is running, use `ros2 launch` to start, the CyberDog will automatically call this script to start, in addition to starting the decision node, it also starts several other nodes at the same time. For details, please refer to [cyberdog_bringup](../cyberdog_bringup) for in-depth understanding .

The complete command is：

```Bash
$ ros2 launch cyberdog_bringup lc_bringup_launch.py
```

This startup state is the process of the robot's normal startup, and all functions can be tested.

### **Debug Methods**

This module supports built-in debugging mode and supports GDB debugging.

#### Debug Mode

`motion_manager` will detect three macro definitions, which are the general switch to turn on debugging, motion debugging and simulated motion data

```
DEBUG_ALL  // for complete debug
DEBUG_MOTION // for gait & motion debug
DEBUG_MOCK  // for mock lcm messages
```

To enter debug mode, you need to turn on `DEBUG_ALL`:
- If you need to output movement-related logs, you can turn on `DEBUG_MOTION`
- If there is no LCM data input source, you can open `DEBUG_MOCK`
- `DEBUG_MOTION` and `DEBUG_MOCK` can be opened at the same time

The debugging function can be turned on by creating `.debug_xxx` in the root directory of `decision_maker`, such as

```shell
$ touch .debug_all
```

The relationship is as follows

|DEF|File|
|---|----|
|DEBUG_ALL|.debug_all|
|DEBUG_MOTION|.debug_motion|
|DEBUG_MOCK|.debug_mock|

#### Debug with GDB

1. First, you need to add the compilation flag of `-g` in `CMakeLists.txt`, generally in the `add_compile_options` function
2. Then modify it according to [cyberdog_bringup](../cyberdog_bringup), use the `gdb` prefix to start
3. Determine whether there is a debugging terminal tool in the system, including `gdb` and `xterm`, etc. If not, you need to install it
4. After ensuring that the same node does not exist under the same `Domain ID` and the same `namespace` in the system, use `Launch` to start it in an environment with a graphical interface.
5. Start debugging.

## **[Future]**
---
-Improve perception function decision-making
-Improve interactive function decision
-Improve the decision-making of automated functions
-Modular & plug-in single point function
-Dynamic parameter tuning
-Online programming of orders
-Custom configuration of functions