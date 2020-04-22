import os
import sys

from setuptools import find_packages
from setuptools import setup


__description__ = 'Application with NLP models'
__package_name__ = 'NLPer'
__version__ = '0.1.0'


# ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
# SRC_DIR = os.path.join(ROOT_DIR, "nlper")
# sys.path += [ROOT_DIR, SRC_DIR]


dependencies = [
    'beautifulsoup4==4.8.0',
    'click==7.0',
    'numpy==1.18.1',
    'pandas==1.0.1',
    'scikit-learn==0.22.2.post1',
    'spacy==2.2.3',
    'tqdm==4.42.1',
]

dev_dependencies = [
    'flake8==3.7.9',
    'matplotlib==3.2.0',
]

test_dependencies = [
    'pytest==4.0.0',
    'pytest-cov==2.8.1',
    'pytest-flake8==1.0.4',
]


setup(
    name=__package_name__,
    version=__version__,
    author='Wojciech Ozimek',
    author_email='ozimekwojciech@zoho.eu',
    description=__description__,
    packages=find_packages(),
    platforms='any',
    python_requires='>3.5.0',
    install_requires=dependencies,
    tests_require=test_dependencies,
    extras_require={
        'dev': test_dependencies + dev_dependencies
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
        'Development Status :: 1 - Planning',
        'Environment :: Console',
        'Intended Audience :: Developers',
        'Operating System :: POSIX',
        'Operating System :: Unix',
        'Programming Language :: Python :: 3',
        'Topic :: Scientific/Engineering :: Artificial Intelligence'
        'Topic :: Software Development :: Libraries :: Python Modules',
    ],
)
