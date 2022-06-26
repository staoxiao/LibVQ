from setuptools import setup, find_packages

with open("README.md", mode="r", encoding="utf-8") as readme_file:
    readme = readme_file.read()

setup(
    name="LibVQ",
    version="0.0.2",
    author="Shitao Xiao, Zheng Liu, Yingxia Shao",
    author_email="stxiao@bupt.edu.cn",
    description="A Library For Dense Retrieval Oriented Vector Quantization",
    long_description=readme,
    long_description_content_type="text/markdown",
    license="MIT License",
    url="https://github.com/staoxiao/LibVQ",
    packages=find_packages(),
    install_requires=[
        'transformers>=4.9.0',
        'faiss-gpu>=1.6.4.post2',
        'scann==1.2.3',
        'tqdm',
        'torch>=1.6.0',
        'numpy'
    ],
    keywords="Vector Quantization, ANN, IVF"
)