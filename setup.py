from pybind11.setup_helpers import Pybind11Extension, build_ext
from setuptools import setup

setup(
    name="quad_sim",
    ext_modules=[Pybind11Extension("quad_sim", ["cpp/quadcopter.cpp", "cpp/bindings.cpp"], cxx_std=17)],
    cmdclass={"build_ext": build_ext},
)