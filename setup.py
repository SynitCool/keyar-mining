from distutils.core import setup

setup(
    name="keyar-mining",
    packages=["krmining"],
    version="0.0.1",
    license="MIT",
    description="Package for machine learning algorithm and data mining",
    author="BASIS / SynitIsCool",
    author_email="zolarpixel3@gmail.com",
    url="https://github.com/SynitCool/keyar-mining",
    download_url="https://github.com/SynitCool/keyar-mining/archive/refs/tags/0.0.1.tar.gz",
    keywords=["Data Mining", "Machine Learning"],
    install_requires=[
        "numpy",
        "pandas",
    ],
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.5",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
    ],
)
