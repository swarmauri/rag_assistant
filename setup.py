from setuptools import setup, find_packages
import rag_assistant

setup(
    name="rag_assistant",
    version=rag_assistant.__version__,
    author="Jacob Stewart",
    author_email="corporate@swarmauri.com",
    description=rag_assistant.__short_desc__,
    long_description=rag_assistant.__long_desc__,
    long_description_content_type="text/markdown",
    url="http://github.com/swarmauri/rag_assistant",
    license="MIT",
    packages=find_packages(include=["rag_assistant"]),
    entry_points={
        "console_scripts": [
            "rag_assistant = rag_assistant.main:main",
        ]
    },
    include_package_data=True,
    install_requires=["python =>=3.10,<4.0", "gradio==5.1.0", "swarmauri==0.5.0"],
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.10",
    ],
    python_requires=">=3.10",
    setup_requires=["wheel"],
)
