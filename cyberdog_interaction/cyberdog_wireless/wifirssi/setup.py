from glob import glob

import os

from setuptools import setup

package_name = 'wifirssi'

setup(
    name=package_name,
    version='1.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name), glob('launch/*.launch.py')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='shenshuqiu',
    maintainer_email='shenshuqiu@xiaomi.com',
    description='ROS 2 package for Wi-Fi',
    license='Apache License, Version 2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
                'talker = wifirssi.publisher_wifi_rssi:main',
                'listener = wifirssi2.subscriber_wifi_rssi:main',
         ],
    },
)
