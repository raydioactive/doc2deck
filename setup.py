from setuptools import setup, find_packages

setup(
    name="doc2deck",
    packages=["LLM_webapp"],
    include_package_data=True,
    package_data={
        "LLM_webapp": ["../templates/*.html"],
    },
)