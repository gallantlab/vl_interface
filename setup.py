from setuptools import setup, find_packages

requirements = [
    "numpy",
    "matplotlib",
    "scipy",
    "cottoncandy",
    "pycortex",
    "networkx"
]

if __name__ == '__main__':

    setup(name='vl_interface',
          version='0.1.0',
          description='Analysis of Visual-Linguistic Interface',
          author='Sara Popham',
          author_email='spopham@berkeley.edu',
          url='https://github.com/gallantlab/vl_interface/',
          packages=find_packages(),
          install_requires=requirements
    )
