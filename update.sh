#!/bin/bash

# --- CONFIGURATION ---
BRANCH="main"        # Change to 'master' if your repo uses master
VENV_DIR="venv"      # Your virtual environment folder name

# --- COLORS ---
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${YELLOW}---------------------------------------------${NC}"
echo -e "${YELLOW}      VISION ENGINE AUTO-UPDATER             ${NC}"
echo -e "${YELLOW}---------------------------------------------${NC}"

# 1. Check for Git Repository
if [ ! -d ".git" ]; then
    echo -e "${RED}[ERROR] This folder is not a git repository.${NC}"
    echo "Please clone the repository first or run this inside the project folder."
    exit 1
fi

# 2. Fetch Latest Info
echo -e "${GREEN}[INFO]${NC} Fetching latest updates from GitHub..."
git fetch --all
if [ $? -ne 0 ]; then
    echo -e "${RED}[ERROR] Could not fetch from remote. Check internet connection.${NC}"
    exit 1
fi

# 3. Status Check
LOCAL=$(git rev-parse @)
REMOTE=$(git rev-parse "origin/$BRANCH")

if [ "$LOCAL" = "$REMOTE" ]; then
    echo -e "${GREEN}[INFO] System is already up-to-date.${NC}"
    exit 0
fi

# 4. The Safe Update (Reset)
echo -e "${YELLOW}[ACTION] Updating files...${NC}"
echo "   (Your venv and models will be preserved)"

# git reset --hard moves the HEAD to the remote branch.
# It overwrites tracked files (vision-engine.py) but leaves untracked files (venv/, *.pt) alone.
git reset --hard "origin/$BRANCH"

if [ $? -eq 0 ]; then
    echo -e "${GREEN}[SUCCESS] Files updated to latest version.${NC}"
    
    # 5. Permissions Fix
    echo -e "${GREEN}[INFO]${NC} Restoring executable permissions..."
    chmod +x vision-engine.py
    chmod +x update.sh
    
    echo -e "${YELLOW}---------------------------------------------${NC}"
    echo -e "${GREEN} Update Complete.${NC}"
    echo -e " You can now restart the vision engine."
    echo -e "${YELLOW}---------------------------------------------${NC}"
else
    echo -e "${RED}[ERROR] Update failed.${NC}"
    exit 1
fi