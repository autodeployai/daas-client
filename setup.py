from setuptools import setup
import os

VERSION_PATH = os.path.join("daas_client", "version.py")
exec(open(VERSION_PATH).read())

VERSION = __version__ # noqa

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name="daas-client",
    version=VERSION,
    description="Python client library for DaaS",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=["daas_client"],
    # include_package_data=True,
    install_requires=[
        "requests", "pandas", "numpy", "pypmml", "onnx", "onnxruntime"
    ],
    url="https://github.com/autodeployai/daas-client",
    download_url="https://github.com/autodeployai/daas-client/archive/v" + VERSION + ".tar.gz",
    author="AutoDeploy AI",
    author_email="autodeploy.ai@gmail.com",
    license="Apache License 2.0",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Environment :: Console",
        "Environment :: Web Environment",
        "Intended Audience :: Developers",
        "Intended Audience :: System Administrators",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 2.7",
        "Programming Language :: Python :: 3.5",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Topic :: Scientific/Engineering :: Artificial Intelligence"
    ],
)
