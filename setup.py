from setuptools import setup

setup(name='rise',
      version='0.1',
      description='RISE: Robust Individualized Decision Learning with Sensitive Variables',
      url='https://github.com/ellenxtan/rise',
      author='Xiaoqing Tan',
      author_email='xit31@pitt.edu',
      license='MIT',
      packages=['rise'],
      install_requires=[ #TODO
          'pandas','numpy','tensorflow','keras','sklearn','lifelines','scipy'
      ],
      zip_safe=False)
