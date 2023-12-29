#!/bin/bash

# Replace SOURCE_FILE_PATH with the path of the source file
SOURCE_FILE_PATH="/Kart.csv"

# Replace DESTINATION_PATH with the path of the destination directory
DESTINATION_PATH="../../data/"

# Move the file
mv "$SOURCE_FILE_PATH" "$DESTINATION_PATH"

# Optionally, print a message indicating the file has been moved
echo "File moved successfully from $SOURCE_FILE_PATH to $DESTINATION_PATH"