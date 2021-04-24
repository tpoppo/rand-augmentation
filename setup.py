import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="rand-augmentation",
    version="0.1.0",
    author="tpoppo",
    description="An augmentation function for tensorflow pipeline",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/tpoppo/rand-augmentation",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    packages=setuptools.find_packages(),
    python_requires=">=3.6",
    install_requires=["tensorflow", "tensorflow-addons"],
)