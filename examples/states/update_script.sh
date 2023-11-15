#!/bin/bash

# Automates the redoing of local installation every time we need to update code within jaxrl5
cd ../..

pip uninstall jaxrl5

pip install ./

cd examples/states