#!/usr/bin/env python

from distutils.core import setup

setup(name='RuleFit',
      description='RuleFit algorithm',
      author='Christoph Molnar',
      author_email='christoph.molnar@gmail.com',
      url='',
      packages=['rulefit'],
      install_requires=['scikit-learn>=0.20.2',
                        'numpy>=1.16.1',
                        'pandas>=0.24.1',
                        'lightgbm>=3.0.0']
     )
