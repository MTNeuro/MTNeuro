from setuptools import setup, find_packages

import MTNeuro


def readme():
    with open('README.md', encoding='utf-8') as f:
        content = f.read()
    return content


README = readme()


setup(
    name='mtneuro',
    version=mtneuro.__version__,
    url=mtneuro.HOMEPAGE,
    license='MIT License',
    author='MTNeuro',
    author_email='evadyer@gatech.edu',
    python_requires=">=3.6.0",
    description='MTNeuro: A Benchmark for Evaluating Representations of Brain Structure Across Multiple Levels of Abstraction',
    long_description=readme(),
    long_description_content_type="text/markdown",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "matplotlib",
        "scikit-learn",
        "tqdm",
        "Pillow",
        "intern",
        "torch",
        "torchvision",
        "pretrainedmodels",
        "efficientnet-python",
        "timm"
    ],
    zip_safe=True,
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3 :: Only"
    ]
)
