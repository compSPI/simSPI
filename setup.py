"""Create instructions to build the simSPI package."""

import setuptools

requirements = []

setuptools.setup(
    name="simSPI",
    maintainer=[
        "Frédéric Poitevin",
        "Nina Miolane",
        "Geoffrey Woollard",
        "Harshit Gupta",
    ],
    version="0.0.3",
    maintainer_email=[
        "frederic.poitevin@stanford.edu",
        "ninamiolane@ucsb.edu",
        "geoffwoollard@gmail.com",
        "hgupta@slac.stanford.edu",
    ],
    description="Methods and tools for simulating single particle imaging data",
    long_description=open("README.md", encoding="utf8").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/compSPI/simSPI.git",
    packages=setuptools.find_packages(),
    install_requires=requirements,
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    zip_safe=False,
)
