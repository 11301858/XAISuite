from setuptools import setup, find_namespace_packages

setup(name='XAISuite',
      version='0.6.9',
      description='XAISuite: Training and Explanation Generation Utilities for Machine Learning Models',
      long_description=open("README.md", "r", encoding="utf-8").read(),
      long_description_content_type="text/markdown",
      keywords="XAI Explainable AI Explanation Machine Learning Models",
      url="https://github.com/11301858/XAISuite",
      author = "Shreyan Mitra",
      install_requires=[
        "omnixai>=1.2"
        "dash"
        "dash_bootstrap_components"
      ],
       packages=["xaisuite", "demo"],
      )
