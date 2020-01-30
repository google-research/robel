# Copyright 2019 The ROBEL Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Installs robel.

To install:
pip install -e .
"""

import fnmatch
import os

import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()


def get_requirements(file_name):
    """Returns requirements from the given file."""
    file_path = os.path.join(os.path.dirname(__file__), file_name)
    with open(file_path, 'r') as f:
        return [line.strip() for line in f]


def get_data_files(package_dir, patterns):
    """Returns filepaths matching the given pattern."""
    paths = set()
    for directory, _, filenames in os.walk(package_dir):
        for pattern in patterns:
            for filename in fnmatch.filter(filenames, pattern):
                base_path = os.path.relpath(directory, package_dir)
                paths.add(os.path.join(base_path, filename))
    return list(paths)


setuptools.setup(
    name="robel",
    version="0.1.2",
    license='Apache 2.0',
    description=('Robotics reinforcement learning benchmark tasks with '
                 'cost-effective robots.'),
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=setuptools.find_packages(),
    package_data={
        'robel': get_data_files('robel',
                                     ['*.xml', '*.pkl', '*.stl', '*.png']),
    },
    data_files=[
        ('', ['requirements.txt', 'requirements.dev.txt']),
    ],
    install_requires=get_requirements('requirements.txt'),
    extra_requires={
        'dev': get_requirements('requirements.dev.txt'),
    },
    tests_require=['absl-py'],
    python_requires='>=3.5.3',
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: Apache Software License',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Intended Audience :: Science/Research',
    ],
)
