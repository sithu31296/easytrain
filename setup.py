import os
import pkg_resources
from setuptools import setup, find_packages


setup(
    name="easytrain",
    py_modules=['easytrain'],
    version="0.1",
    description="Resuable Trainer Tool",
    author="Sithu Aung",
    packages=find_packages(include="easytrain"),
    install_requires=[
        str(r)
        for r in pkg_resources.parse_requirements(
            open(os.path.join(os.path.dirname(__file__), "requirements.txt"))
        )
    ],
    include_package_data=True
)