from setuptools import setup, find_packages

with open("README.md", mode="r", encoding="utf-8") as readme_file:
    readme = readme_file.read()

setup(
    name="LibVQ",
    version="0.1.0",
    author="MSRA-STCA",
    author_email="xxx@xxx.com",
    description="Vector Quantization For Efficiently Retrieval",
    long_description=readme,
    long_description_content_type="text/markdown",
    # license="Apache License 2.0",
    # url="https://github.com/",
    # download_url="https://github.com/",
    packages=find_packages(),
    install_requires=[
        'transformers>=4.6.0',
        'tqdm',
        'torch>=1.6.0',
        'numpy'
    ],
    keywords="Vector Quantization, Embedding based Retrieval, IVF"
)