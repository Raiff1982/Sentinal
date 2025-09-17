from setuptools import setup, find_packages

setup(
    name="sentinal",
    version="0.1.0",
    description="A multi-timescale AI guardrails system with hoax/misinformation filter and CLI.",
    author="Raiff1982",
    packages=find_packages(),
    install_requires=[
        "nltk",
        "rapidfuzz",
        "filelock",
        "numpy"
    ],
    python_requires='>=3.8',
    entry_points={
        "console_scripts": [
            "hoax-scan=sentinal.hoax_scan:main"
        ]
    },
    include_package_data=True,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)