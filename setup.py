from setuptools import setup

setup(name='markcorr', 
      description='A package for computing galaxy two-point and marked correlation functions ',
      version='0.1',
      author='Unnikrishnan Sureshkumar',
      author_email='unnikrishnan.sureshkumar@wits.ac.za',
      packages=['markcorr'],
      include_package_data=True,
      install_requires=['numpy',
                        'matplotlib',
                        'scipy',
                        'astropy',
                        'treecorr',
                        'gundam',
                        'pandas'
                        ],
		scripts=[],
		url='https://github.com/ukp1513/markcorr', 
)
