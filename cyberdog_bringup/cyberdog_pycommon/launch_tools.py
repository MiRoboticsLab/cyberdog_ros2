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

import subprocess

from typing import List

import launch

from launch import LaunchContext, LaunchDescription
from launch_ros.actions import Node
from launch_ros.substitutions.executable_in_package import ExecutableInPackage

import yaml


def runcmd(command):
    ret = subprocess.run(command,
                         shell=True,
                         stdout=subprocess.PIPE,
                         stderr=subprocess.PIPE,
                         encoding='utf-8',
                         timeout=1)
    if ret.returncode == 0:
        print('success:', ret)
    else:
        print('error:', ret)
    return ret.returncode, ret.stdout


def Get_Namespace():
    ret, ns = runcmd('cat /sys/firmware/devicetree/base/serial-number')
    if ret == 0:
        ns = ns.strip()
    else:
        ns = ''
    ns = ns[0:-1]
    n = len(ns)
    if n > 0:
        n = min(n, 7)
        ns = ns[-n:]
    ns_ = 'mi' + ns

    print('serial number:%s' % (ns_))
    return ns_


def Load_Yaml(path, threwError=False):
    file_name = str(path).split('/')[-1]
    try:
        with open(path, 'r') as f:
            yaml_file = yaml.safe_load(f.read())  # ros2_foxy use
            # yaml_file = yaml.load(f.read(), Loader=yaml.FullLoader) # ros2_galactic use
            f.close()
    except IOError as e:
        yaml_file = None
        raise e('[launcher][Load_Yaml_error] Cant find yaml file [%s]' %
                file_name)
    except yaml.YAMLError as e:
        print('Error parsing file [%s]' % file_name)
        raise e
    return yaml_file


def Yaml_launcher(launch_param: List, launch_nodes_yaml: dict,
                  launch_groups_yaml: dict, nodes_param, namespace,
                  remappings) -> LaunchDescription:
    """Load launch nodes detail from "launch_nodes.yaml"."""
    except_nodes: List = None
    include_nodes: List = None
    debug_param = launch_nodes_yaml['launch_nodes'][
        'debug_param'] if 'debug_param' in launch_nodes_yaml[
            'launch_nodes'] else None
    if (launch_groups_yaml is not None):
        launch_groups = launch_groups_yaml['launch_groups']
        target_launch_group = launch_groups[
            'target_launch_group'] if 'target_launch_group' in launch_groups else None
        if (target_launch_group in launch_groups['groups']):
            print(
                '[launcher][Yaml_launcher_info] Launch nodes base on group[%s]'
                % target_launch_group)
            target = launch_groups['groups'][target_launch_group]
            if (target is not None):
                if ('except' in target):
                    except_nodes = target['except'] if target[
                        'except'] is not None else []
                if ('launch' in target):
                    include_nodes = target['launch'] if target[
                        'launch'] is not None else []
            if (except_nodes is None and include_nodes is None):
                print('[launcher][Yaml_launcher_info] Empty launch group[%s], '
                      'all nodes will be launch' % target_launch_group)
            if (except_nodes is not None and include_nodes is not None):
                error_t = '[launcher][Yaml_launcher_error] Group type conflict, '
                'only one type can be used (except/launch)'
                print(error_t)
                raise Exception('\n' + error_t)
        else:
            error_t = '[launcher][Yaml_launcher_error] Cant find target_launch_group'
            print(error_t)
            raise Exception('\n' + error_t)
    else:
        except_nodes = []
        print(
            '[launcher][Yaml_launcher_info] Empty launch_groups.yaml, launch all nodes'
        )

    ld = LaunchDescription()
    if (launch_param is not None and len(launch_param) > 0):
        for p in launch_param:
            ld.add_action(p)
    context = LaunchContext()
    nodes_for_check = []
    if (launch_nodes_yaml['launch_nodes']['base_nodes'] is not None):
        for n in launch_nodes_yaml['launch_nodes']['base_nodes']:
            n.update({'launch_type': 'Base'})
            nodes_for_check.append(n)
    if (launch_nodes_yaml['launch_nodes']['other_nodes'] is not None):
        for n in launch_nodes_yaml['launch_nodes']['other_nodes']:
            n.update({'launch_type': 'Other'})
            nodes_for_check.append(n)

    nodes_checked = []
    checked_no_error = {'Base': True, 'Other': True}
    for n in nodes_for_check:
        if (n is None):
            continue
        node_def_name = list(n.keys())[0]
        if ((except_nodes is not None and node_def_name in except_nodes)
                or (include_nodes is not None
                    and node_def_name not in include_nodes)):
            print('[launcher][Yaml_launcher_info] Disable launch node[%s]' %
                  node_def_name)
            continue
        param = n[node_def_name]
        if ('package' not in param or 'executable' not in param):
            checked_no_error[n['launch_type']] = False
            error_t = '[launcher][Yaml_launcher_error] '
            '%s_nodes:[%s] package or executable missing' % (n['launch_type'],
                                                             node_def_name)
            print(error_t)
            if (n['launch_type'] == 'Base'):
                raise Exception('\n' + error_t)
            continue
        try:
            checker = ExecutableInPackage(package=param['package'],
                                          executable=param['executable'])
            checker.perform(context)
        except (AttributeError, launch.substitutions.SubstitutionFailure, LookupError) as e:
            checked_no_error[n['launch_type']] = False
            error_t = ('[launcher][Launch_Check_Error] %s_nodes:[%s] '
                       'check error [package=%s,executable=%s]' %
                       (n['launch_type'], node_def_name, param['package'],
                        param['executable']))
            print(error_t)
            if (n['launch_type'] == 'Base'):
                raise e('\n' + error_t)
            continue
        remappings_tuplelist = []
        if (remappings is not None):
            try:
                if (remappings['remappings'] is not None
                        and node_def_name in remappings['remappings']
                        and remappings['remappings'][node_def_name] is not None
                        and len(remappings['remappings'][node_def_name]) != 0):
                    for r in remappings['remappings'][node_def_name]:
                        t = tuple(r)
                        if (len(t) != 2):
                            raise IOError('Remapping format error')
                        remappings_tuplelist.append(t)
            except IOError:
                remappings_tuplelist = []
                print('[launcher][Yaml_launcher_error] Remapping error,'
                      ' disable remapping in node[%s]' % node_def_name)
                continue
        else:
            print(
                '[launcher][Yaml_launcher_info] Empty remappings.yaml, disable all remappings'
            )
        print('[launcher][Yaml_launcher_info] Success check: node[%s]' %
              node_def_name)
        nodes_checked.append(
            Node(package=param['package'],
                 executable=param['executable'],
                 namespace=namespace,
                 parameters=[nodes_param] if 'load_nodes_param' in param
                 and param['load_nodes_param'] else None,
                 name=param['name']
                 if 'name' in param and param['name'] else None,
                 output='screen' if 'output_screen' in param
                 and param['output_screen'] else 'log',
                 prefix=[debug_param] if 'enable_debug' in param
                 and param['enable_debug'] else None,
                 respawn=True if 'respawn' in param
                 and param['respawn'] else False,
                 remappings=remappings_tuplelist))
    print('[launcher][Launch_Check_Info] Finish lanuch check, ', end='')
    if (checked_no_error['Base']):
        if (checked_no_error['Other']):
            print('all group nodes launch')
        else:
            print('without some nodes launch')
    else:
        print('launch error')
    for n in nodes_checked:
        ld.add_action(n)
    return ld
