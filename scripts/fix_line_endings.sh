#!/bin/bash
# Fix line endings for all scripts and Python files
# Run this if you get '\r' errors

echo "Fixing line endings for all scripts..."

# Fix shell scripts
find scripts -name "*.sh" -exec dos2unix {} \;
find local -name "*.sh" -exec dos2unix {} \;
find examples -name "*.sh" -exec dos2unix {} \;

# Fix Python scripts
find tools -name "*.py" -exec dos2unix {} \;
find local -name "*.py" -exec dos2unix {} \;
find cosyvoice -name "*.py" -exec dos2unix {} \;

# Fix path.sh files
find . -name "path.sh" -exec dos2unix {} \;

echo "✓ Line endings fixed!"
echo "You can now run: bash scripts/prepare_data.sh"
