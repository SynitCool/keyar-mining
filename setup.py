from setuptools import setup, find_packages

setup(
    name="keyar-mining",
    packages=find_packages(),
    version="0.0.2",
    license="MIT",
    description="Package for machine learning algorithm and data mining",
    author="BASIS / SynitIsCool",
    author_email="zolarpixel3@gmail.com",
    url="https://github.com/SynitCool/keyar-mining",
    download_url="https://github.com/SynitCool/keyar-mining/archive/refs/tags/v0.0.2.tar.gz",
    keywords=["Data Mining", "Machine Learning"],
    include_package_data=True,
    python_requires=">= 3.5",
    extras_require={"docs": ["mkdocs"]},
    platforms="any",
    install_requires=["numpy", "pandas", "setuptools"],
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.5",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
    ],
)
