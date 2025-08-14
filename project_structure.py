from pathlib import Path
import os
import logging

# Creating a logging stream here i.e. whatever logs we input, the terminal will keep a record automatically.
# INFO means we are creating only the information-related logs inside the terminal.
# Format parameter is specified to print the logs in a decided format.

logging.basicConfig(level=logging.INFO, format='[%(asctime)s]: %(message)s:')

project_name = 'NephroScan-AI'

# Created .github folder inside which workflows folder is kept because whenever we are committing our code in GitHub, and if it is empty, we keep a .gitkeep file. At the time of CI/CD pipeline creation, we will replace this .gitkeep
# with main.yaml file (a core file of CI/CD).
# Created a SOURCE folder to maintain a secure structure of code so that it follows the availability, scalability, maintainability principles.
# Created a configuration.yaml file where < YAML file is a human-readable text format used to store and organize data in a structured way.
# YAML is widely used for configurations, data serialization, and pipeline setups because it’s easy for both humans and machines to read.
# DVC (Data Version Control) is an open-source tool for managing datasets, machine learning models, and experiments — similar to Git but for data.
# Created a Jupyter notebook file for running temporary experiments.

list_of_files = [
    ".github/workflows/.gitkeep",
    f"source/{project_name}/__init__.py",
    f"source/{project_name}/components/__init__.py",
    f"source/{project_name}/utils/__init__.py",
    f"source/{project_name}/config/__init__.py",
    f"source/{project_name}/config/configuration.py",
    f"source/{project_name}/pipeline/__init__.py",
    f"source/{project_name}/entity/__init__.py",
    f"source/{project_name}/constants/__init__.py",
    "config/config.yaml",
    "dvc.yaml",
    "params.yaml",
    "requirements.txt",
    "setup.py",
    "research/trials.ipynb",
    "templates/index.html"
]

for filepath in list_of_files:
    filepath = Path(filepath)
    filedir, filename = os.path.split(filepath)

    if filedir != "":
        os.makedirs(filedir, exist_ok=True)
        logging.info(f"Creating directory: {filedir} for the file: {filename}")

    if (not os.path.exists(filepath)) or (os.path.getsize(filepath) == 0):
        with open(filepath, "w") as f:
            pass
        logging.info(f"Creating Empty File: {filepath}")
    else:
        logging.info(f"{filename} already exists")
