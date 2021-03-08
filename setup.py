from setuptools import setup, find_packages

# setup(name='imagemp',
      # version='0.1',
      # url='',
      # license='MIT',
      # author='Max Nikitchenko',
      # author_email='nikitchmtech@gmail.com',
      # description='Process image sequences in multiple processes with shared memory areas',
      # classifiers=[
          # 'Development Status :: 3 - Alpha',
          # 'Intended Audience :: Developers',
          # 'Topic :: Software Development :: Libraries',
          # 'License :: OSI Approved :: MIT License',
          # 'Programming Language :: Python :: 2',
          # 'Programming Language :: Python :: 2.7',
          # 'Programming Language :: Python :: 3',
      # ],
      # packages=find_packages(exclude=['tests']),
      # long_description=open('README.md').read(),
      # zip_safe=False,
      # setup_requires=['pytest>=3.0'],
	  # test_suite='pytest')
	  

setup(name='imagemp',
      version='0.1.2',
      license='MIT',
      author='Max Nikitchenko',
      author_email='nikitchmtech@gmail.com',
      description='Process image sequences in multiple processes with shared memory areas',
	  packages=find_packages(),
	  zip_safe=False)