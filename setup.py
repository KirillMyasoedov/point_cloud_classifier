from setuptools import setup
# from distutils.core import setup
from catkin_pkg.python_setup import generate_distutils_setup

settings = generate_distutils_setup(
    packages=[
        "src",
        "src.models",
        "src.utils",
        "src.criterions"
    ],
)

setup(install_requires=["numpy",
                        "scipy",
                        "torch",
                        "torchvision",
                        "scikit-image",
                        "pyrealsense2",
                        "pillow",
                        "sklearn"],
      **settings)
