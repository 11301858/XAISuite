from setuptools import setup, find_namespace_packages

setup(name='XAISuite',
      version='2.9.2', 
      description='XAISuite: Training and Explanation Generation Utilities for Machine Learning Models',
      long_description=open("README.md", "r", encoding="utf-8").read(),
      long_description_content_type="text/markdown",
      keywords="XAI Explainable AI Explanation Machine Learning Models",
      url="https://github.com/11301858/XAISuite",
      author = "Shreyan Mitra",
      install_requires=[
        "omnixai==1.2.4",
        "dash>=2.7.1",
        "dash_bootstrap_components>=1.3.0",
        "skorch>=0.12.1", #For integration with pytorch
        "torch>=2.0.0",
        "scikeras>=0.10.0", #For integration with Tensorflow Keras
        "seaborn==0.12.2" #For visualization
      ],
      include_package_data=True,
      package_data={'': ['static/*']},
      packages=["xaisuite", "demo", "xaisuitegui", "models", "explainers"],
      )
