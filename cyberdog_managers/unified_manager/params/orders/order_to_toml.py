#!/usr/bin/python3
#
# Copyright (c) 2021 Beijing Xiaomi Mobile Software Co., Ltd.
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

rules = {
    'locomotion': ['vel', 'omni', 'gait', 'duration'],
    'pose': ['foot_support', 'body_cmd', 'foot_cmd', 'ctrl_point', 'duration'],
    'swingleg': ['foot_support', 'body_cmd', 'foot_cmd', 'ctrl_point', 'duration'],
    'jump': ['contact_state', 'x_acc_cmd', 'w_acc_cmd', 'duration'],
    'torctrlposture': ['contact_state', 'body_cmd', 'foot_cmd', 'duration'],
    'transition': ['height', 'duration']
}


def get_var(t: str, num: int) -> str:
    max_len = 0
    L = rules[t]
    for i in L:
        if (len(i) > max_len):
            max_len = len(i)
    result = L[num] if num != -1 else 'type'
    for n in range(max_len + 1 - (len(result) if num != -1 else 4)):
        result += ' '
    result += '= ' if num != -1 else '= "%s"' % t
    return result


def group_number(T: str, check_num: int = -1) -> str:
    t = T.split(',')
    if(check_num != -1 and len(t) != check_num):
        print('[Warning] List length unmatched')
    result = '['
    for i in t:
        result += str(float(i)) + ', '
    return result[:-2]+']'


def order_to_toml(path: str):
    f = open(path, 'r')
    lines_old = f.readlines()
    f.close
    lines_new = []
    caser_type = None
    caser_line = 0
    for line in lines_old:
        if(line.find('#') != -1 or line == '\n' or line == ''):
            continue
        line = line[:-1]
        if(caser_type is None):
            caser_type = line
            caser_line = 0
            lines_new.append('\n[[step]]\n')
            lines_new.append(get_var(caser_type, -1)+'\n')
            continue
        lines_new.append(
            get_var(caser_type, caser_line) + group_number(line)+'\n')
        caser_line += 1
        if(caser_line == len(rules[caser_type])):
            caser_type = None
            continue
    f = open(path[:-3]+'toml', 'w')
    f.writelines(lines_new)
    f.close()


if __name__ == '__main__':
    for file in os.listdir():
        if (file[-3:] != 'txt'):
            continue
        try:
            order_to_toml(file)
            print('[ChangeSuccess] Change file[%s] to toml' % file)
        except IOError as ex:
            print(
                '[ChangeError] Exception [%s] happend. Cant change file[%s] to toml' % ex, file)
