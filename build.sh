#!/bin/bash
# Build script for Blender add-in zip package
# Excludes /tests and files/folders in .gitignore

set -e

ZIP_NAME="prompt2blend.zip"

# Remove old zip if exists
rm -f "$ZIP_NAME"

# Gather .gitignore patterns and add /tests to exclude list
exclude_args=()
while IFS= read -r pattern; do
  # Skip comments and empty lines
  [[ -z "$pattern" || "$pattern" =~ ^# ]] && continue
  # Remove leading/trailing whitespace
  pattern="${pattern## }"
  pattern="${pattern%% }"
  # If pattern is a directory (no wildcard, no dot, not a file pattern)
  if [[ -d $pattern || ( ! "$pattern" =~ "*" && ! "$pattern" =~ "." ) ]]; then
    exclude_args+=("--exclude=$pattern/*")
  else
    exclude_args+=("--exclude=$pattern")
  fi
done < .gitignore

# Build the zip, excluding /tests and .gitignore patterns
zip -r "$ZIP_NAME" . \
  --exclude "*/tests/*" \
  "${exclude_args[@]}" \
  --exclude "$ZIP_NAME" \
  --exclude "./.git/*" \
  --exclude "./.gitignore" \
  --exclude "./build.sh"

echo "Created $ZIP_NAME excluding /tests and .gitignore patterns."
