from setuptools import find_packages, setup

setup(
    name="MLVisualizer",
    version='0.0.2',
    packages=find_packages(),
    include_package_data=True,
    python_requires=">=3.8",
    install_requires=['seaborn', 'scikit-learn'],

)
