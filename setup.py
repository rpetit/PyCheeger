import setuptools

with open("../pycheeger/README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="pycheeger",
    version="0.0.1",
    author="Romain Petit",
    author_email="romain.petit@inria.fr",
    description="Cheeger sets computation package",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/rpetit/PyCheeger",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)