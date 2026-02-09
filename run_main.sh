#!/usr/bin/env bash
# Wrapper to run the project under the project's Python venv.
set -euo pipefail
cd /Users/admin/Downloads/human_motion_detection

# Activate the venv if present
if [ -f ".venv311/bin/activate" ]; then
  # shellcheck disable=SC1091
  source .venv311/bin/activate
fi

# Ensure log directory exists and run, writing logs to a user-writable location
mkdir -p "$HOME/Library/Logs"
exec python main.py >> "$HOME/Library/Logs/com.human_motion_detection.log" 2>&1
