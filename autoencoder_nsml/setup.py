#nsml: floydhub/pytorch:0.4.0-gpu.cuda9cudnn7-py3.29

from distutils.core import setup
setup(
    name='nsml example',
    version='1.0',
    description='ns-ml',
    install_requires=[
        'opencv-python'
    ]
)