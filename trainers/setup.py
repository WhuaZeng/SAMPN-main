from setuptools import setup, Extension
from Cython.Build import cythonize
import os

setup(
    name='meta_trainer',
    ext_modules=cythonize([
        Extension(
            "meta_trainer",
            ["meta_trainer.py"],
            language="c"
        )
    ]),
    zip_safe=False,
)
# python setup.py build_ext --inplace