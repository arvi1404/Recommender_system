#!/bin/bash

# Define environment name
ENV_NAME="venv"

# Create a new conda environment with Python 3.10 (or change version as needed)
conda create -n $ENV_NAME python=3.10 -y

# Activate the environment
source activate $ENV_NAME

# Install required libraries
pip3 install flask pandas numpy scikit-learn, Flask-SQLAlchemy, mysql-connector

# Optional: Install additional libraries for development
# conda install jupyterlab matplotlib seaborn -c conda-forge -y

# Display message
echo "Environment '$ENV_NAME' created and libraries installed. Activate it using: conda activate $ENV_NAME"
