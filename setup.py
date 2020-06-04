from setuptools import setup
from pathlib import Path


setup(name='convenience-collection',
      version='1.0.0',
      author="Liam Henrickson",
      author_email="liam.henrickson@gmail.com",
      license="MIT License",
      url="https://github.com/HLiam/ConvenienceCollection",
      long_description=Path('README.md').read_text(),
      long_description_content_type='text/markdown',
      py_modules=['convenience_collection'],
      install_requires=['colorama',
                        'win10toast ; platform_system=="Windows"'],
      classifiers=['Programming Language :: Python :: 3',
                   'License :: OSI Approved :: MIT License',
                   'Operating System :: OS Independent'],
      python_requires='>=3.5',
)
