#!/bin/bash
rm -rf venv
rm -rf __pycache__
rm -rf .pytest_cache
find . -type f -name '*.pyc' -delete
find . -type d -name '__pycache__' -delete
