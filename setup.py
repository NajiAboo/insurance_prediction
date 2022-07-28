from setuptools import setup,find_packages
from typing import List

REQUIREMENT_FILE_NAME = "requirements.txt"


def get_requirements_list() -> List[str]:
    """
    Description:
    This function going to return the list of requirements. 
    It removes "e .  " if its exist in the requirement file

    Args:
        None
    Returns:
        list of requirements
    Raises:
        None
    """

    rq_packages = []

    with open(REQUIREMENT_FILE_NAME) as requirement_file:
        rq_packages = requirement_file.readlines()
        
        rq_packages.remove('-e .')
        print(rq_packages)
    return rq_packages


PROJECT_NAME = "medicalcost-predictor"
VERSION = "0.0.3"
AUTHOR = "Mohamed Naji Aboo"
DESCRIPTION = "This is a sample template project"


setup(
    name=PROJECT_NAME,
    version=VERSION,
    author=AUTHOR,
    description=DESCRIPTION,
    packages= find_packages(),
    install_requires = get_requirements_list()
)