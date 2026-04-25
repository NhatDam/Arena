import os
from glob import glob
from setuptools import setup, find_packages

package_name = 'data_recorder'

setup(
    name=package_name,
    version='0.0.1',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    author='tom',
    author_email='tom@todo.todo',
    description='ROS2 node to record robot RGB frames, detection bboxes, and trajectory data',
    license='MIT',
    entry_points={
        'console_scripts': [
            'data_recorder_node = data_recorder.data_recorder_node:main',
        ],
    },
)
