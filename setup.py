from setuptools import setup

with open("VERSION") as f:
    version = f.read().strip()

setup(
    name="quantexa_text_classifier",
    version=version,
    packages=["quantexa_test_classifier"]
)
