from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'yolo_tracking'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', 'yolo_tracking', 'launch'), glob('launch/*.launch.py')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='huang',
    maintainer_email='12110313@mail.sustech.edu.cn',
    description='YOLO_ROS DEVELOPMENT',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'vision_node = yolo_tracking.vision_node:main',
            'rfid_sim_node = yolo_tracking.rfid_sim_node:main',
            'target_pub_node = yolo_tracking.target_pub_node:main',
            'depth_estimator_node = yolo_tracking.depth_estimator_node:main',  
            'tracking_controller_node = yolo_tracking.tracking_controller_node:main',
            'moving_object_node = yolo_tracking.moving_object_node:main',
            'fusion_tracking_node = yolo_tracking.fusion_tracking_node:main',
            'moving_ambulance_node = yolo_tracking.moving_ambulance_node:main',
            
        ],
    },
)
