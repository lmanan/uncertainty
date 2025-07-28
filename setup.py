from setuptools import setup, find_packages

setup(
    name='uncertainty',
    version='0.0.1',
    packages=find_packages(),
    author='Manan Lalit',
    author_email='manan.lalit@gmail.com',
    description='Using TARFlow to model uncertainty of edges.',
    long_description=open('README').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/yourusername/my_package',  # optional
    classifiers=[
        'Programming Language :: Python :: 3',
    ],
    python_requires='>=3.6',
    install_requires=[
        "torch",
        "torchmetrics"
        #"motile_toolbox @ git+https://github.com/lmanan/motile_toolbox.git",
    ],
)

