from setuptools import find_packages
from setuptools import setup


__description__ = 'Application with NLP models'
__package_name__ = 'NLPer'
__version__ = '1.0.0'


with open('requirements.txt', 'r') as requirements:
    install_requires = requirements.read()


with open('requirements_test.txt', 'r') as test_requirements:
    test_requires = test_requirements.read()


setup(
    name=__package_name__,
    version=__version__,
    author='Wojciech Ozimek',
    author_email='ozimekwojciech@zoho.eu',
    description=__description__,
    packages=find_packages(),
    platforms='any',
    python_requires='>3.5.0',
    install_requires=install_requires,
    setup_requires=['flake8'],
    extras_require={
        'test': test_requires
    },
    entry_points={
        'console_scripts': [
            'clean-data = nlper.main:clean_data',
            'clean-text = nlper.main:clean_text',
            'predict = nlper.main:predict',
            'split-train-test = nlper.main:split_train_test',
            'train = nlper.main:train',
        ]
    },
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Environment :: Console',
        'Environment :: GPU :: NVIDIA CUDA',
        'Framework :: Flake8',
        'Framework :: Jupyter',
        'Framework :: Pytest',
        'Intended Audience :: Education',
        'Intended Audience :: Developers',
        'Intended Audience :: Information Technology',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: BSD License',
        'Operating System :: MacOS :: MacOS X',
        'Operating System :: POSIX',
        'Operating System :: Unix',
        'Programming Language :: Python :: 3',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Scientific/Engineering :: Information Analysis',
        'Topic :: Software Development :: Libraries :: Python Modules',
    ],
)
