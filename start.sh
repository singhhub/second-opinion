#!/bin/bash

# Second Opinion - Start script

echo "Starting Second Opinion backend on http://localhost:8001..."
echo "Open http://localhost:8001/ui once it's up."
echo ""

uv run python -m backend.main
