#!/usr/bin/env bash
# Pull the latest code and redeploy. Run as your normal user (uses sudo).
set -euo pipefail
sudo -u oraculum -H git -C /srv/oraculum pull --ff-only
sudo -u oraculum -H bash -lc 'cd /srv/oraculum && uv sync --extra research'
sudo -u oraculum -H env PATH=/opt/node22/bin:/usr/bin:/bin bash -c \
  'cd /srv/oraculum/frontend && npm ci --no-audit --no-fund && npm run build'
sudo -u oraculum -H bash -lc 'cd /srv/oraculum && set -a && . /etc/oraculum/oraculum.env && set +a && .venv/bin/alembic upgrade head'
sudo systemctl restart oraculum
sleep 3
curl -fsS http://127.0.0.1:8092/health/ready && echo
