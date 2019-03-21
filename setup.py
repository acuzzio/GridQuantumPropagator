'''
My setup !
'''

from setuptools import setup
from Cython.Build import cythonize
import numpy

setup(
    name='quantumpropagator',
    version='0.1.0',
    description='It propagates nuclear quantum wavepackets into electronic PES on a grid',
    license='',
    url='https://github.com/acuzzio/GridQuantumPropagator',
    author_email='',
    keywords='quantum propagation nuclear wavepacket',
    package_dir={'': 'src'},
    packages=["quantumpropagator"],
    classifiers=[
        'Intended Audience :: Science/Research',
        'Programming Language :: Python :: 3',
        'Development Status :: 3 - Alpha',
        'Topic :: Scientific/Engineering :: Chemistry'
    ],
    scripts=['Scripts/SinglePointPropagatorLauncher.py',
     'Scripts/multiGraphEneDipole.py',
     'Scripts/2dmultiGraphEneDipole.py',
     'Scripts/3dmultiGraphEneDipole.py',
     'Scripts/NorbornadieneGenerateGeometries.py',
     'Scripts/generateGeomsAroundConical.py',
     'Scripts/Report_Generator.py',
     'Scripts/ThreeDPropagator.py',
     'Scripts/CreateOutputFilesFromWF.py',
     'Scripts/PostProcessing3dData.py'],
    install_requires=['h5py', 'numpy', 'scipy', 'pandas', 'matplotlib', 'pyyaml','cython'],
    extras_require={'test': ['nose', 'coverage']},
    ext_modules=cythonize('src/quantumpropagator/CPropagator.pyx'),
    include_dirs=[numpy.get_include()]
)
