from setuptools import setup

setup(
    name="advfussion",
    py_modules=["advfussion", "advfussion"],
    install_requires=["blobfile>=1.0.5", "torch", "tqdm", "lpips", "robustbench", "tensorboardX",],
)
