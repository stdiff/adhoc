from setuptools import setup

setup(
    name="adhoc",
    version="0.4",
    description="Module for ad hoc analysis",
    url="https://github.com/stdiff/adhoc",
    author="Hironori Sakai",
    author_email="crescent.lab@gmail.com",
    license="MIT",
    packages=["adhoc"],
    python_requires=">=3.7",
    install_requires = ["numpy>=1.18.0",
                        "pandas>=1.0.0",
                        "scipy>=1.2.3",
                        "matplotlib>=3.2.1",
                        "seaborn>=0.10.0",
                        "scikit-learn>=0.22.2",
                        "pydot>=1.4.1",
                        "graphviz>=0.13.2"
                        "openpyxl>=3.0.3",
                        "xlrd>=1.2.0"
                        ],
    zip_safe=False
)