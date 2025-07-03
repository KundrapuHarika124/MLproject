from setuptools import setup, find_packages
from typing import List

def get_requirements(file_path:str)->list:
    """
    This function will return the list of requirements
    """
    requirements = []
    with open(file_path) as file_obj:
        requirements = file_obj.readlines()
        requirements = [req.replace('\n', '') for req in requirements]
        if '-e .' in requirements:
            requirements.remove('-e .')
    return requirements    

setup(

    name='end_to_end_project',
    version='0.0.1',
    author='kundrapu Harika',
    author_email='harikakundrapu216@gmail.com',
    packages=find_packages(),
    install_requires=get_requirements('requirements.txt')
)