from setuptools import setup, find_packages
import re
import os
import io


def _read(*parts, **kwargs):
    filepath = os.path.join(os.path.dirname(__file__), *parts)
    encoding = kwargs.pop("encoding", "utf-8")
    with io.open(filepath, encoding=encoding) as fh:
        text = fh.read()
    return text


def get_version():
    version = re.search(
        r'^__version__\s*=\s*[\'"]([^\'"]*)[\'"]',
        _read("chimaera", "__init__.py"),
        re.MULTILINE,
    ).group(1)
    return version


def get_requirements(path):
    content = _read(path)
    return [
        req
        for req in content.split("\n")
        if req != "" and not (req.startswith("#") or req.startswith("-"))
    ]

setup_requires = [
    'cython',
    'numpy',
]
install_requires = get_requirements("requirements.txt")

setup(
    name="chimaera",
    version=get_version(),
    author="ashkolikov",
    license="MIT",
    packages=find_packages(),
    include_package_data=True,
    setup_requires=setup_requires,
    install_requires=install_requires,
)
