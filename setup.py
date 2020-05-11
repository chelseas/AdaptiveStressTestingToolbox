import setuptools

# Required dependencies
REQUIRED = ['garage',
            'numpy==1.14.5',
            'gym',
            'python-dateutil',
            'dowel',
            'joblib',
            'cached_property',
            'akro',
            'glfw',
            'pyprind',
            'cma',
            'bsddb3',
            'fire',
            'depq',
            'compress_pickle',]

# Dependencies for optional features
EXTRAS = {}

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="ast-toolbox",
    version="06.1.2020.dev1",
    author="Stanford Intelligent Systems Laboratory",
    author_email="mkoren@stanford.edu",
    description="A toolbox for worst-case validation of autonomous policies",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/sisl/AdaptiveStressTestingToolbox",
    packages=setuptools.find_packages(where='ast_toolbox'),
    classifiers=[
        "License :: OSI Approved :: MIT License",

        "Operating System :: OS Independent",

        'Intended Audience :: Developers',
        'Intended Audience :: Education',
        'Intended Audience :: Science/Research',
        'Topic :: Software Development :: Libraries',
        'Topic :: Software Development :: Testing',
        'Topic :: Scientific/Engineering :: Mathematics',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',

        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",

        'Development Status :: 3 - Alpha'.
    ],
    python_requires='>=3.6',
    project_urls={
        'Source': 'https://github.com/sisl/AdaptiveStressTestingToolbox',
        'Documentation':'https://ast-toolbox.readthedocs.io/en/master/',
        'Tracker':'https://github.com/sisl/AdaptiveStressTestingToolbox/issues',
        'Status':'https://travis-ci.org/github/sisl/AdaptiveStressTestingToolbox',
        'Testing':'https://codecov.io/gh/sisl/AdaptiveStressTestingToolbox',
    },
    install_requires=REQUIRED,
    extras_require=EXTRAS,
)