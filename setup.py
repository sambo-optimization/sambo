import os
import sys
from setuptools import setup, find_packages

if sys.version_info < (3, 10):
    sys.exit('ERROR: sambo requires Python 3.10+')


if __name__ == '__main__':
    setup(
        name="sambo",
        license="AGPL-3.0",
        description="Sequential And Model-Based Optimization",
        long_description=open(os.path.join(os.path.dirname(__file__), 'README.md'),
                              encoding='utf-8').read(),
        long_description_content_type='text/markdown',
        url="https://sambo-optimization.github.io",
        project_urls={
            'Documentation': 'https://sambo-optimization.github.io/doc/sambo/',
            'Source': 'https://github.com/sambo-optimization/sambo',
            'Tracker': 'https://github.com/sambo-optimization/sambo/issues',
        },
        classifiers=[
            "Topic :: Scientific/Engineering",
            "Topic :: Scientific/Engineering :: Mathematics",
            "Topic :: Scientific/Engineering :: Visualization",
            "License :: OSI Approved :: GNU Affero General Public License v3 or later (AGPLv3+)",
            "Development Status :: 5 - Production/Stable",
            "Environment :: Console",
            "Intended Audience :: Developers",
            "Intended Audience :: Education",
            "Intended Audience :: Financial and Insurance Industry",
            "Intended Audience :: Healthcare Industry",
            "Intended Audience :: Manufacturing",
            "Intended Audience :: Science/Research",
            "Operating System :: OS Independent",
            "Framework :: Jupyter",
            "Framework :: Matplotlib",
            'Programming Language :: Python :: 3 :: Only',
            "Typing :: Typed",
        ],
        entry_points={},
        packages=find_packages(),
        include_package_data=True,
        install_requires=[
            "numpy >= 1.10.0",
            "scipy >= 1.11.0",
        ],
        extras_require={
            'all': [
                "scikit-learn >= 1.1",
                "joblib",
                "matplotlib",
            ],
        },
        setup_requires=[
            'setuptools_git',
            'setuptools_scm',
        ],
        use_scm_version={
            'write_to': os.path.join('sambo', '_version.py'),
        },
        python_requires='>= 3.10',
    )
