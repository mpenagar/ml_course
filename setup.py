from setuptools import setup

setup(
    name='ml_course',
    version='0.1.2',
    description='Educational wrappers for Machine Learning models',
    author='Your Name/Institution',
    # We specify the single Python module instead of a package/folder
    py_modules=['ml_course'],
    install_requires=[
        'scikit-learn',
        'numpy'
    ],
    python_requires='>=3.6',
)