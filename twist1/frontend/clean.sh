#!/bin/bash

# Exit immediately if a command exits with a non-zero status
set -e

# Print commands and their arguments as they are executed
set -x

# # Check if Verdaccio is installed
# if command -v verdaccio &> /dev/null; then
#     echo "Verdaccio is installed. Checking if it is running..."

#     # Ensure Verdaccio is running
#     if ! pgrep -f verdaccio > /dev/null; then
#         echo "Starting Verdaccio..."
#         nohup verdaccio &> verdaccio.log &
#         sleep 5 # Give Verdaccio some time to start
#     fi

#     # Set npm to use Verdaccio registry
#     npm set registry http://localhost:4873/
# else
#     echo "Verdaccio is not installed. Using the default npm registry."

#     # Set npm to use the default registry
#     npm set registry https://registry.npmjs.org/
# fi

echo "Cleaning the cache and reinstalling the node modules"
npm cache clean --force

# Check if package-lock.json exists and remove it
if [ -f package-lock.json ]; then
    rm package-lock.json
else
    echo 'package-lock.json does not exist'
fi


echo "Removing the node_modules and package-lock.json"
rm -rf node_modules

