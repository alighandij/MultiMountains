from setuptools import find_packages, setup
setup(
    name='multi_mountains',
    packages=find_packages(),
    version='1.1.5',
    description='MultiMountains',
    author='Mahyar',
    license='MIT',
    install_requires=[
        "gym",
        "numpy",
        "scipy",
        "keyboard",
        "matplotlib",
    ]
)