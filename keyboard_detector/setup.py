from setuptools import find_packages, setup

package_name = 'keyboard_detector'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='krzysztof',
    maintainer_email='krzysiektkaczyk02@gmail.com',
    description='TODO: Package description',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'detection_node = keyboard_detector.detection_node:main',
            'publisher = keyboard_detector.__publish_image_node:main',
            'viewer = keyboard_detector.kd_results_viewer:main'
        ],
    },
)
