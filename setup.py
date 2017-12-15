'''
My setup !
'''

from setuptools import setup


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
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Chemistry'
    ],
    scripts=['Scripts/SinglePointPropagatorLauncher.py',
     'Scripts/OxiallylDensityWriter.py', 'Scripts/ReadCube.py',
     'Scripts/multiGraphEneDipole.py', 'Scripts/2dmultiGraphEneDipole.py',
     'Scripts/NorbornadieneGenerateGeometries.py'],
    install_requires=['h5py', 'numpy', 'scipy', 'pandas', 'matplotlib'],
    extras_require={'test': ['nose', 'coverage']}
)
