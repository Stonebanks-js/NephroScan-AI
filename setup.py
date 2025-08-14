import setuptools

with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

with open("requirements.txt", "r", encoding="utf-8") as f:
    requirements = f.read().splitlines()

__version__ = "0.0.0"

REPO_NAME = "NephroScan-AI"
AUTHOR_USER_NAME = "Stonebanks-js"
SRC_REPO = "nephroscan_ai"  # package folder name, no spaces, if there will be spaces the control flow will break
AUTHOR_EMAIL = "aradhyachdry@outlook.com"

setuptools.setup(
    name=SRC_REPO,
    version=__version__,
    author=AUTHOR_USER_NAME,
    author_email=AUTHOR_EMAIL,
    description="Kidney tumor, disease, and infection detection system based on CNNs",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url=f"https://github.com/{AUTHOR_USER_NAME}/{REPO_NAME}",
    project_urls={
        "Bug Tracker": f"https://github.com/{AUTHOR_USER_NAME}/{REPO_NAME}/issues",
    },
    package_dir={"": "source"},
    packages=setuptools.find_packages(where="source"),
    install_requires=requirements,
    python_requires=">=3.8",
)
