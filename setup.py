from setuptools import setup, find_packages
from typing import List

def get_requirement(file_path:str)->List[str]:
    requirements = []
    with open(file_path) as file_obj:
        requirements = file_obj.readlines()
        requirements = [req.replace('\n','') for req in requirements]

        return requirements



setup(
    name='DiamondPricePrediction',
    version='0.0.1',
    author='Ravinder',
    author_email='rvindervrma@gmail.com',
    install_requires = get_requirement('requirements.txt'),
    packages=find_packages()
)