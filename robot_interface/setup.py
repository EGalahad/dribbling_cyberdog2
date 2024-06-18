
import os
import re
import subprocess
import sys
from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext

class CMakeExtension(Extension):
    def __init__(self, name, sourcedir="."):
        Extension.__init__(self, name, sources=[])
        self.sourcedir = os.path.abspath(sourcedir)

class CMakeBuild(build_ext):
    def run(self):
        try:
            subprocess.check_output(["cmake", "--version"])
        except OSError:
            raise RuntimeError(
                "CMake must be installed to build the following extensions: "
                + ", ".join(e.name for e in self.extensions)
            )

        for ext in self.extensions:
            self.build_extension(ext)

    def build_extension(self, ext):
        extdir = os.path.abspath(os.path.dirname(self.get_ext_fullpath(ext.name)))
        cmake_args = [
            "-DCMAKE_LIBRARY_OUTPUT_DIRECTORY=" + extdir,
            "-DPYTHON_EXECUTABLE=" + sys.executable,
            "-DCMAKE_BUILD_TYPE=" + ("Debug" if self.debug else "Release"),
        ]

        build_args = ["-j6"]

        if not os.path.exists(self.build_temp):
            os.makedirs(self.build_temp)

        subprocess.check_call(
            ["cmake", ext.sourcedir] + cmake_args, cwd=self.build_temp
        )
        subprocess.check_call(
            ["cmake", "--build", ".", "--target", ext.name] + build_args,
            cwd=self.build_temp,
        )

setup(
    name="robot_interface",
    version="0.1",
    author="Haoyang Weng",
    author_email="elijahgalahad@gmail.com",
    description="A Python interface for cyberdog2 motor control based on pybind11.",
    long_description="",
    ext_modules=[CMakeExtension("robot_interface", sourcedir=".")],
    cmdclass={"build_ext": CMakeBuild},
    package_data={"": ["*.pyi"]},
    zip_safe=False,
)
