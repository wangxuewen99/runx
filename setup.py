from setuptools import setup, find_packages

with open('README.md', 'r') as f:
    long_description = f.read()

with open('requirements.txt') as f:
    requirements = f.read().splitlines()

__version__ = '0.1.0'

setup(
    name='runx',
    version=__version__,
    description='Utils For Running A Deep Learning Experiment.',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://git.yy.com/wangxuewen/runx',
    author='Xuewen Wang',
    author_email='wangxuewen@yy.com',

    packages=find_packages(),
    python_requires='>=3.6',
    install_requires=requirements
)
