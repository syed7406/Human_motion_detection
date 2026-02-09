#!/usr/bin/env bash
# VM startup / deploy script
# Usage: sudo bash deploy_vm.sh <REPO_URL> [--user ubuntu] [--dir /home/ubuntu/human_motion_detection] [--gpu yes]

set -euo pipefail

REPO_URL=${1:-}
if [ -z "$REPO_URL" ]; then
  echo "Usage: sudo bash deploy_vm.sh <REPO_URL> [--user ubuntu] [--dir /home/ubuntu/human_motion_detection] [--gpu yes]"
  exit 1
fi

USER="ubuntu"
APP_DIR="/home/$USER/human_motion_detection"
GPU="no"

shift || true
while [[ $# -gt 0 ]]; do
  case "$1" in
    --user) USER="$2"; APP_DIR="/home/$USER/human_motion_detection"; shift 2;;
    --dir) APP_DIR="$2"; shift 2;;
    --gpu) GPU="$2"; shift 2;;
    *) shift ;;
  esac
done

echo "Deploying to $APP_DIR as user $USER (GPU=$GPU)"

# Install system deps (Ubuntu/Debian)
apt-get update
apt-get install -y python3.11 python3.11-venv python3.11-dev build-essential git curl

# Create application directory and clone
mkdir -p "$APP_DIR"
chown "$USER":"$USER" "$(dirname "$APP_DIR")" || true

sudo -u "$USER" git clone "$REPO_URL" "$APP_DIR" || {
  echo "Repository clone failed or already exists. If it already exists, pull updates instead.";
}

cd "$APP_DIR"

# Create venv
sudo -u "$USER" python3.11 -m venv .venv311
export PATH="$APP_DIR/.venv311/bin:$PATH"

pip install --upgrade pip setuptools wheel

# Install backend requirements
if [ -f backend/requirements.txt ]; then
  pip install -r backend/requirements.txt
else
  pip install -r requirements.txt
fi

if [ "$GPU" = "yes" ]; then
  echo "GPU requested — ensure the VM has NVIDIA drivers and CUDA installed. Install appropriate torch+cuda wheel for your CUDA version.";
  echo "Example (CUDA 11.8): pip install torch --index-url https://download.pytorch.org/whl/cu118";
fi

# Create systemd unit
SERVICE_PATH="/etc/systemd/system/hmd-backend.service"
cat > "$SERVICE_PATH" <<EOF
[Unit]
Description=Human Motion Detection Backend
After=network.target

[Service]
User=$USER
WorkingDirectory=$APP_DIR
ExecStart=$APP_DIR/.venv311/bin/uvicorn backend.api_server:app --host 0.0.0.0 --port 8000
Restart=always
RestartSec=5
Environment=PYTHONUNBUFFERED=1

[Install]
WantedBy=multi-user.target
EOF

systemctl daemon-reload
systemctl enable hmd-backend.service
systemctl start hmd-backend.service

echo "Deployed and started hmd-backend.service. Check status with: systemctl status hmd-backend.service"
echo "Logs: journalctl -u hmd-backend.service -f"
