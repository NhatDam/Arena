from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'arena_ai_integration'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', glob('launch/*.launch.py')),
        ('share/' + package_name + '/config', glob('config/*.yaml')),
        ('share/' + package_name + '/config/models', glob('config/models/*.yaml')),
        ('share/' + package_name + '/checkpoints', glob('checkpoints/*') + ['checkpoints/.gitkeep']),
        ('share/' + package_name + '/scripts', glob('scripts/*.sh')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Arena Team',
    maintainer_email='arena@example.com',
    description='Unified AI controller integration for Arena-Rosnav',
    license='Apache-2.0',
    tests_require=['pytest'],
    options={
        'develop': {
            'script_dir': '$base/lib/' + package_name,
        },
        'install': {
            'install_scripts': '$base/lib/' + package_name,
        },
    },
    entry_points={
        'console_scripts': [
            'ai_controller = arena_ai_integration.nodes.ai_controller_node:main',
            'human_states_bridge = arena_ai_integration.nodes.human_states_bridge:main',
            'semantic_laser_filter = arena_ai_integration.nodes.semantic_laser_filter:main',
            'aggregate_benchmark_metrics = arena_ai_integration.tools.aggregate_benchmark_metrics:main',
        ],
    },
)
