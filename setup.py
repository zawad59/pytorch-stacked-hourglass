from setuptools import setup, find_packages


setup(
    name='pytorch-stacked-hourglass',
    version='1.0.0a0',
    packages=find_packages('src'),
    package_dir={'': 'src'},
    include_package_data=True,
)
