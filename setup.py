from setuptools import setup

setup(
    name="mongoslabs",
    version="0.0.1",
    author="Sergey Plis",
    author_email="s.m.plis@gmail.com",
    packages=["mongoslabs"],
    url="http://pypi.python.org/pypi/mongoslabs/",
    license="MIT",
    description="Dataloader that serves MRI images from a mongodb",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    install_requires=["numpy", "scipy >= 1.7", "pymongo >= 4.0", "torch", ""],
)
