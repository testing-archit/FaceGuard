#!/bin/bash
# macOS: double-click to start FaceGuard. First run installs components (needs internet once).
cd "$(dirname "$0")" || exit 1
if ! command -v python3 >/dev/null 2>&1; then
  echo "Python 3.10+ is required: https://www.python.org/downloads/"
  read -r -p "Press Return to close." _
  exit 1
fi
python3 launcher.py
status=$?
[ $status -ne 0 ] && read -r -p "FaceGuard stopped with an error. Press Return to close." _
exit $status
