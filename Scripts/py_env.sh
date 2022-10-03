#!/bin/bash

# Add dead-snakes for older python versions
version=$1
environment=$2

# Install dead-snakes for older python versions
sudo add-apt-repository  -y ppa:deadsnakes/ppa

# Update
sudo apt update

# Install required version (default 3.8)
sudo apt install "$version"
sudo apt install "$version"-venv

# Create virtual environment
$("$version" -m venv "$environment")

# Activate virtual environment
source "$environment"/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install all requirements
pip install -r requirements.txt

