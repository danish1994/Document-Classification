from setuptools import setup

setup(name='Document Classification',
      version='0.1',
      description='Document Classification in Python',
      url='https://github.com/danish1994/Document-Classification',
      author='Danish',
      author_email='danish8802204230@gmail.com',
      license='MIT',
      packages=['src'],
      install_requires=[
          'nltk',
          'numpy',
          'scipy',
          'scikit-learn',
          'matplotlib',
          'setuptools',
          'python-dateutil',
          'pyparsing',
          'pytz',
          'cycler',
          'lda',
          'pypdf2',
          'tqdm'
      ])
