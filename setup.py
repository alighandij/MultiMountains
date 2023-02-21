from setuptools import find_packages, setup

setup(
    name='sidechannels',
    packages=find_packages(),
    version='1.1.1',
    description='Sidechannels Util Libraries',
    author='Ali-Azam-Dana-Mahyar',
    license='MIT',
    install_requires=[
        ##! ONLY ADD ESSENTIAL PACKAGES !
        "gym",
        "scipy",
        "numpy",
        "matplotlib",
        "keyboard",
    ]
)