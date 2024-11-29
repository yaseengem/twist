#!/bin/bash
source ./venv/bin/activate
pip install -r requirements.txt

uvicorn main:app --host 127.0.0.1 --port 8000 --reload

