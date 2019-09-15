from setuptools import setup

setup(name='tdavis',
      version='0.0.1',
      description='Python 3 library a topological data analysis plotting library',
      author='Nathaniel Rodriguez',
      packages=['tdavis'],
      url='https://github.com/Nathaniel-Rodriguez/tdavis.git',
      install_requires=[
          'numpy',
          'scipy',
          'gudhi',
          'umap-learn',
          'scikit-learn',
          'networkx',
          'matplotlib',
          'linkcom==v0.0.1 @ https://github.com/Nathaniel-Rodriguez/linkcom/archive/v0.0.1.zip#egg=linkcom-v0.0.1'
      ],
      include_package_data=True)
