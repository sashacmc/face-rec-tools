from setuptools import setup
setup(name='face-rec-tools',
      version='1.0.0',
      description='Face Recognition Tools',
      author='Alexander Bushnev',
      author_email='Alexander@Bushnev.ru',
      license='GNU General Public License v3.0',
      packages=['face_rec_tools'],
      package_data={'face_rec_tools': ['cfg/*', 'web/*']},
      python_requires='>=3.6',
      install_requires=[
          'dlib',
          'numpy',
          'piexif',
          'pillow',
          'opencv-python',
          'face_alignment',
          'face_recognition'
      ],
      scripts=['face-rec-cli',
               'face-rec-server',
               'face-rec-patterns',
               'face-rec-plexsync'],
      zip_safe=False)
