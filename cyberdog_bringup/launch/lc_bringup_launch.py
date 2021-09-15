#!/usr/bin/python3
#
# Copyright (c) 2021 Beijing Xiaomi Mobile Software Co., Ltd. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os

from ament_index_python.packages import get_package_share_directory
from cyberdog_pycommon import Get_Namespace, Load_Yaml, RewrittenYaml, Yaml_launcher
from launch.actions import DeclareLaunchArgument, LogInfo, RegisterEventHandler
from launch.event_handlers.on_shutdown import OnShutdown
from launch.substitutions import LaunchConfiguration


def generate_launch_description():

    # Define arguments
    bringup_dir = get_package_share_directory('cyberdog_bringup')
    Get_Namespace_cmd = DeclareLaunchArgument(
        'namespace',
        default_value=Get_Namespace(),
        description='Namespace of all nodes')
    set_parameter_cmd = DeclareLaunchArgument(
        'params_file',
        default_value=os.path.join(bringup_dir, 'params',
                                   'default_param.yaml'),
        description='Path to paramaters YAML file')
    shutdown_cmd = RegisterEventHandler(
        OnShutdown(on_shutdown=[LogInfo(msg='ROS Apps is exiting.')]))

    namespace = LaunchConfiguration('namespace')
    params_file = LaunchConfiguration('params_file')

    configured_params = RewrittenYaml(source_file=params_file,
                                      root_key=namespace,
                                      convert_types=True)

    launch_nodes = Load_Yaml(
        os.path.join(bringup_dir, 'params', 'launch_nodes.yaml'), True)
    remappings = Load_Yaml(
        os.path.join(bringup_dir, 'params', 'remappings.yaml'))
    launch_groups = Load_Yaml(
        os.path.join(bringup_dir, 'params', 'launch_groups.yaml'))

    return Yaml_launcher(
        launch_param=[Get_Namespace_cmd, set_parameter_cmd, shutdown_cmd],
        launch_nodes_yaml=launch_nodes,
        launch_groups_yaml=launch_groups,
        nodes_param=configured_params,
        namespace=namespace,
        remappings=remappings)
