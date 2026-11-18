# Deploying Oraculum on a single Linux server

This guide puts the public demo on a server that **already runs other
services**. Oraculum gets its own user, folder, database and ports, and runs
at lower CPU priority with memory caps. It never edits another project's
files. Every command is meant to be pasted as your normal (sudo) user.

| Piece | Where | Port |
|---|---|---|
| Model server (llama.cpp + Qwen2.5-0.5B-Instruct, CPU) | `oraculum-llm.service`, `/opt/llama.cpp` | 127.0.0.1:8081 |
| API (FastAPI, PyTorch encoders) | `oraculum.service`, `/srv/oraculum` | 127.0.0.1:8092 |
| Frontend (static build) | `/srv/oraculum/frontend/dist`, served by Caddy | 443 via Caddy |
| Database | your existing PostgreSQL, new role + database `oraculum` | 5432 |

## 0. Preflight (changes nothing)

```bash
free -h; nproc; df -h /
ss -ltnp | grep -E ':8081|:8092' || echo "ports 8081 and 8092 are free"
grep -n "import" /etc/caddy/Caddyfile
```

The files are sized for a **shared 4 GB / 2 vCPU server**: the model server
is capped at 1 GB and the API at 1.4 GB, each with almost no swap, so if
Oraculum ever runs out of memory only Oraculum restarts. You need about
**1.6 GB of available RAM** and **5 GB of free disk**. With 8 GB of RAM or
more, use the 1.5B model instead (see step 4). If 8081 or 8092 is taken,
choose other ports and change them in both `.service` files, `oraculum.caddy`
and `update.sh`.

## 1. User, folder and code

```bash
sudo useradd --system --create-home --home-dir /srv/oraculum --shell /usr/sbin/nologin oraculum
sudo -u oraculum git clone https://github.com/safo-124/oraculum.git /tmp/oraculum-src
sudo -u oraculum bash -c 'shopt -s dotglob && mv /tmp/oraculum-src/* /srv/oraculum/ && rmdir /tmp/oraculum-src'
sudo -u oraculum mkdir -p /srv/oraculum/models /srv/oraculum/data
# Caddy (its own user) must pass through to frontend/dist; data/ stays private.
sudo chmod 711 /srv/oraculum
sudo chmod 700 /srv/oraculum/data
```

## 2. Python environment (uv)

```bash
command -v uv || curl -LsSf https://astral.sh/uv/install.sh | sudo env UV_INSTALL_DIR=/usr/local/bin sh
sudo -u oraculum -H bash -lc 'cd /srv/oraculum && uv sync --extra research && uv cache clean'
```

This installs **CPU-only PyTorch** (about 200 MB) from the PyTorch wheel
index set in `pyproject.toml`. uv downloads Python 3.13 itself if the server
doesn't have it.

## 3. Frontend build (Node 22 in /opt, the system stays untouched)

```bash
NODE_TAR=$(curl -fsSL https://nodejs.org/dist/latest-v22.x/SHASUMS256.txt | grep -o 'node-v22[^ ]*-linux-x64.tar.xz')
curl -fsSL "https://nodejs.org/dist/latest-v22.x/$NODE_TAR" | sudo tar -xJ -C /opt
sudo ln -sfn "/opt/${NODE_TAR%.tar.xz}" /opt/node22
sudo -u oraculum -H env PATH=/opt/node22/bin:/usr/bin:/bin bash -c \
  'cd /srv/oraculum/frontend && npm ci --no-audit --no-fund && npm run build && npm cache clean --force'
```

## 4. Model server (llama.cpp + Qwen2.5)

```bash
command -v unzip || sudo apt-get install -y unzip
# Stops at the first problem and works only inside its own temporary folder.
# llama.cpp's "latest" release is a placeholder, so search recent releases.
bash -euo pipefail <<'EOF'
work=$(mktemp -d)
curl -fsSL "https://api.github.com/repos/ggml-org/llama.cpp/releases?per_page=30" -o "$work/releases.json"
url=$(grep -o 'https://[^"]*-bin-\(ubuntu\|linux\)-x64\.\(zip\|tar\.gz\)' "$work/releases.json" | head -1 || true)
[ -n "$url" ] || { echo "STOP: no Linux x64 CPU build found"; rm -rf "$work"; exit 1; }
echo "Downloading $url"
curl -fL -o "$work/pkg" "$url"
mkdir "$work/x"
case "$url" in *.zip) unzip -q "$work/pkg" -d "$work/x" ;; *.tar.gz) tar -xzf "$work/pkg" -C "$work/x" ;; esac
bin=$(find "$work/x" -name llama-server -type f | head -1)
[ -n "$bin" ] || { echo "STOP: llama-server not in the package"; rm -rf "$work"; exit 1; }
src=$(dirname "$bin")
case "$src" in "$work"/x*) ;; *) echo "STOP: unexpected path $src"; exit 1 ;; esac
sudo mkdir -p /opt/llama.cpp
sudo cp -a "$src"/. /opt/llama.cpp/
rm -rf "$work"
LD_LIBRARY_PATH=/opt/llama.cpp /opt/llama.cpp/llama-server --version
EOF

sudo -u oraculum curl -fL -o /srv/oraculum/models/qwen2.5-0.5b-instruct-q4_k_m.gguf \
  https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-GGUF/resolve/main/qwen2.5-0.5b-instruct-q4_k_m.gguf

sudo cp /srv/oraculum/deploy/oraculum-llm.service /etc/systemd/system/
sudo systemctl daemon-reload && sudo systemctl enable --now oraculum-llm
sleep 10 && curl -s http://127.0.0.1:8081/health && echo
```

Check that grammar-constrained JSON and token log-probabilities both work.
The reply should contain `"content":"{...}"` and a `logprobs` block:

```bash
curl -s http://127.0.0.1:8081/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "messages":[{"role":"user","content":"Is the sky blue? Reply as JSON."}],
  "response_format":{"type":"json_schema","json_schema":{"schema":{"type":"object","properties":{"answer":{"type":"string"}},"required":["answer"]}}},
  "logprobs":true,"top_logprobs":3,"max_tokens":30}' | head -c 600; echo
```

With 8 GB of RAM or more, the 1.5B model gives clearly better answers.
Download `qwen2.5-1.5b-instruct-q4_k_m.gguf` from `Qwen/Qwen2.5-1.5B-Instruct-GGUF`,
change `--model`/`--alias` and `MemoryMax=2G` in the unit, and `CHAT_MODEL` in
the env file. Then run the study again, because the calibrator belongs to the
model it was fitted on.

## 5. Database (a new role and database; existing ones are not touched)

```bash
DBPASS=$(openssl rand -hex 24)
sudo -u postgres psql -v ON_ERROR_STOP=1 \
  -c "CREATE ROLE oraculum LOGIN PASSWORD '$DBPASS'" \
  -c "CREATE DATABASE oraculum OWNER oraculum"
sudo mkdir -p /etc/oraculum
sed "s/CHANGE_ME/$DBPASS/" /srv/oraculum/deploy/oraculum.env.example | sudo tee /etc/oraculum/oraculum.env >/dev/null
sudo chown root:oraculum /etc/oraculum/oraculum.env && sudo chmod 640 /etc/oraculum/oraculum.env
unset DBPASS
sudo -u oraculum -H bash -lc 'cd /srv/oraculum && set -a && . /etc/oraculum/oraculum.env && set +a && .venv/bin/alembic upgrade head'
```

## 6. Admin account, then the demo documents (in this order)

`create_admin.py` takes ownership of every shared document that already
exists, and guests only see shared documents. So create the admin first,
then load the demo corpus:

```bash
sudo -u oraculum -H bash -lc 'cd /srv/oraculum && set -a && . /etc/oraculum/oraculum.env && set +a && .venv/bin/python scripts/create_admin.py you@example.com'
sudo -u oraculum -H bash -lc 'cd /srv/oraculum && set -a && . /etc/oraculum/oraculum.env && set +a && .venv/bin/python scripts/load_demo_corpus.py samples'
```

The first run downloads the three encoders from Hugging Face (about 250 MB
into `/srv/oraculum/.cache`).

## 7. API service

```bash
sudo cp /srv/oraculum/deploy/oraculum.service /etc/systemd/system/
sudo systemctl daemon-reload && sudo systemctl enable --now oraculum
sleep 20 && curl -s http://127.0.0.1:8092/health/ready && echo
```

`model_server` should be `ok`. `calibrator` stays `missing` until step 9, and
until then answers are shown without a confidence gate.

## 8. Caddy site (added beside the existing sites)

Import the Oraculum site **by its exact file name**, never with a wildcard
(other files in `sites/` may be inactive on purpose). If another project's
deploy installs `/etc/caddy/Caddyfile`, add the `import` line to that
project's copy too, or its next deploy will drop Oraculum.

Use the server's public IP in the hostname (nip.io resolves it, and Caddy
gets the HTTPS certificate):

```bash
IP=$(curl -s https://ifconfig.me); echo "$IP"
sudo mkdir -p /etc/caddy/sites
sed "s/YOUR-SERVER-IP/$IP/" /srv/oraculum/deploy/oraculum.caddy | sudo tee /etc/caddy/sites/oraculum.caddy >/dev/null
```

The Caddyfile must import that folder. If the preflight `grep` showed no
`import /etc/caddy/sites/*` line, append one (this only adds a line):

```bash
grep -q 'import /etc/caddy/sites/\*' /etc/caddy/Caddyfile || echo 'import /etc/caddy/sites/*' | sudo tee -a /etc/caddy/Caddyfile
sudo caddy validate --config /etc/caddy/Caddyfile && sudo systemctl reload caddy
```

If your Caddyfile lives in another project's repository (FliipTech), add the
same `import` line there too, so that project's next deploy keeps it.

Open `https://oraculum.<IP>.nip.io` and use **Try the demo**.

## 9. Calibration study (about 3–4 hours on 2 vCPUs, runs in the background)

On a 4 GB server the study needs the API's memory, so the demo is offline
while it runs. Start it in the evening:

```bash
sudo systemctl stop oraculum
sudo systemd-run --unit=oraculum-study --uid=oraculum --gid=oraculum \
  --working-directory=/srv/oraculum --setenv=HOME=/srv/oraculum --setenv=PYTHONUNBUFFERED=1 \
  -p EnvironmentFile=/etc/oraculum/oraculum.env -p Nice=15 -p CPUWeight=20 \
  -p MemoryMax=1400M -p MemorySwapMax=128M -p OOMScoreAdjust=500 \
  /bin/bash /srv/oraculum/deploy/run_study.sh
sudo journalctl -u oraculum-study -f -a   # Ctrl+C stops watching, not the run
sudo -u oraculum wc -l /srv/oraculum/evals/runs/squad.jsonl   # progress
```

The study builds the SQuAD 2.0 and in-domain benchmarks, runs every question
through the live pipeline, then writes `evals/study/REPORT.md` and
`models/calibrator.json`. It can be resumed: if it stops, start it again and
finished questions are skipped. When it finishes:

```bash
sudo systemctl start oraculum
sleep 20 && curl -s http://127.0.0.1:8092/health/ready && echo     # calibrator: loaded
```

## Operating it

```bash
systemctl status oraculum oraculum-llm --no-pager
journalctl -u oraculum -n 100 --no-pager
/srv/oraculum/deploy/update.sh       # pull, rebuild, migrate, restart
```

To remove everything: stop and disable both units, delete the unit files,
`/etc/oraculum`, `/etc/caddy/sites/oraculum.caddy`, `/opt/llama.cpp`,
`/opt/node22`, `/srv/oraculum`, then the `oraculum` database, role and user.
