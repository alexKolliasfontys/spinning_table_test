from setuptools import find_packages, setup
import os
from glob import glob
from setuptools import setup

package_name = 'lite6_pick_predictor'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob(os.path.join('launch', '*launch.[pxy][yma]*'))),
    ],
    install_requires=['setuptools', 'numpy'],
    zip_safe=True,
    maintainer='linux',
    maintainer_email='linux@todo.todo',
    description='TODO: Package description',
    license='TODO: License declaration',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'ign_obj_bridge = lite6_pick_predictor.ign_obj_bridge:main',
            'kalman_predictor = lite6_pick_predictor.kalman_predictor:main',
            'moveit_bridge_action = lite6_pick_predictor.moveit_bridge_action:main',
        ],
    },
)
