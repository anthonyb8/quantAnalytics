#!/bin/bash
# shellcheck disable=SC1091

TYPE=$1
ENV=$2

VENV="venv"

if [ -f "$VENV/bin/activate" ]; then
	echo "Virtual environment found. Activating..."
	source "$VENV/bin/activate"
else
	echo "Error: Virtual environment not found at $VENV/bin/activate"
	exit 1
fi

if [ "$TYPE" = "integration" ]; then
	echo "Run backtest integration test..."
	python -m unittest discover tests.integration
else
	python -m unittest discover
fi
