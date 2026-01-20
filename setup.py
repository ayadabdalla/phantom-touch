from setuptools import setup, find_packages

setup(
    name="phantomtouch",
    version="0.1.0",
    author="Abdalla Ayad",
    author_email="abdalla.ayad@utn.de",
    description="A repository for the Phantom Touch project.",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/ayadabdalla/phantom-touch",
    packages=find_packages(where="."),
    package_dir={"": "."},
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.7",
    install_requires=[
        # Add your dependencies here, e.g., "numpy>=1.21.0"
    ],
    extras_require={
        "dev": [
            "pytest>=6.0",
            "flake8",
            "black",
        ],
    },
    entry_points={
        "console_scripts": [
            "phantom-touch=main:main",  # Adjust this to your main entry point
        ],
    },
)