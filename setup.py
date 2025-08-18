from setuptools import setup, find_packages

setup(
    name='beamseti',  
    version='0.1.0',
    description='Tools for comparing empirical and synthetic SETI surveys and plotting CMDs.',
    author='Matti Weiss',
    author_email='weissm@bxscience.edu',              
    url='https://github.com/matti656/beamseti',
    packages=find_packages(),
    python_requires='>=3.8',
    install_requires=[
        'numpy',
        'pandas',
        'matplotlib',
        'scipy',
        'astropy',
        'synthpop',
        'astroquery',
        's2sphere',
            ],
    classifiers=[
        'Programming Language :: Python :: 3',
        'Operating System :: OS Independent',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Astrophysics',
    ],
    include_package_data=True,
)
