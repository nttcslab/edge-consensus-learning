from setuptools import find_packages, setup

setup(
    name='edgecons',
    version='1.1.0',
    packages=find_packages(),
    include_package_data=True,
    zip_safe=False,
    url='https://github.com/nttcslab/edge-consensus-learning',
    entry_points={
        "console_scripts": [
            "eclmnist = eclsample.run_mnist:main",
            "eclcifar10 = eclsample.run_cifar10:main",
        ]
    },
    license='',
    author='',
    author_email='',
    description=''
)
