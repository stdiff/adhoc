from setuptools import setup

setup(
    name="adhoc",
    version="0.2",
    description="Module for ad hoc analysis",
    url="https://github.com/stdiff/adhoc",
    author="Hironori Sakai",
    author_email="crescent.lab@gmail.com",
    license="MIT",
    packages=["adhoc"],
    python_requires=">=3.5",
    install_requires = ["numpy>=1.16.4",
                        "pandas>=0.23.4",
                        "scipy>=1.2.0",
                        "matplotlib>=3.0.2",
                        "seaborn>=0.9.0",
                        "scikit-learn>=0.21.3",
                        "pydot>=1.4.1",
                        "jupyter>=1.0.0",
                        "openpyxl>=2.6.3"
                        ],
    zip_safe=False
)