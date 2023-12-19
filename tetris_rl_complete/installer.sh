#!/bin/bash

REQUIREMENTS_FILE="requirements.txt"

pip install -r $REQUIREMENTS_FILE

INSTALLED_PACKAGES_FILE="installed_packages.txt"
pip list --format=freeze > $INSTALLED_PACKAGES_FILE

echo "List of installed packages by the installer script:"
grep -Fxf $REQUIREMENTS_FILE $INSTALLED_PACKAGES_FILE

rm $INSTALLED_PACKAGES_FILE
