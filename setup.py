from setuptools import setup, find_packages

setup(
    name='libSimpleNnet',
    version='0.0.5a', 
    url='https://github.com/asparks1987/libsimplennet.git',
    author='Aryn M. Sparks',
    author_email='Aryn.sparks1987@gmail.com',
    description='A simple nnet using tensorflow',
    packages=find_packages(),    
    install_requires=['pandas','scikit-learn','tensorflow','matplotlib','PyQt5','numpy'],
)