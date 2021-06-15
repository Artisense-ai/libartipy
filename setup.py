
from distutils.core import setup


setup(
    name='LibArtiPy',
    version='1.0',
    packages=[
        'libartipy',
        'libartipy.dataset',
        'libartipy.geometry',
        'libartipy.IO',
        'libartipy.pointcloud',
    ],
    package_data={'': ['IO/color_map.json', 'IO/semantic_class_mapping.json']},
    author='Dmytro Bobkov, Thomas Schmid, Dominik van Opdenbosch, Pavel Ermakov, Qing Cheng and others',
    description='Python library for the 4Seasons dataset',
    long_description=open('README.md').read()
)
