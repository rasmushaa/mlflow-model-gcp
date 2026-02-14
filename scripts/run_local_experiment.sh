#!/bin/bash

# Runs a local DEV experiment using the .env.dev configuration.
# Make sure to have the .env.dev file properly set up before running this script.
# This is intended for local testing and development purposes to avoid using production resources.

# Function to load environment variables from .env.dev file
load_env() {
    echo "Loading environment variables from .env.dev..."
    if [ -f .env.dev ]; then
        # Read the .env.dev file line by line
        while IFS= read -r line; do
            # Skip comments and empty lines
            [[ "$line" =~ ^#.*$ || -z "$line" ]] && continue
            # Export the variable
            export "$line"
            echo "$line" | sed 's/=.*/=***/'
        done < .env.dev
    else
        echo ".env.dev file not found."
        exit 1
    fi
}


load_env
uv run src/main.py
