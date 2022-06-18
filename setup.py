from setuptools import setup, find_packages

setup(
    name='mtneuro',
    version='1.0.0',
    url='https://mtneuro.github.io/',
    license='MIT License',
    author='MTNeuro',
    author_email='evadyer@gatech.edu',
    python_requires=">=3.6.0",
    description='MTNeuro: A Benchmark for Evaluating Representations of Brain Structure Across Multiple Levels of Abstraction',
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
        "efficientnet",
        "timm"
    ],
    zip_safe=True,
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3 :: Only"
    ]
    )
