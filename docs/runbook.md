# Runbook

> **Status:** Phase 5 stub. Full daily-ops content (log review, dead-letter
> triage, disk monitoring, ingestion resumption, snapshot/restore, etc.) is
> added in Phase 11. This file currently contains only the Tantivy sidecar
> systemd unit template required by Phase 5.

## Hosting the Tantivy BM25 sidecar with systemd

The Tantivy sidecar runs on the **host**, not inside docker-compose, because
the Phase 4 bulk ingester writes to the same on-disk index directory. The
Open WebUI container reaches it via `http://host.docker.internal:8765`.

For production deployment, install as a systemd unit:

```ini
# /etc/systemd/system/tantivy-sidecar.service
[Unit]
Description=Tantivy BM25 search sidecar (transcript-rag)
After=network-online.target
Wants=network-online.target

[Service]
Type=exec
User=transcript-rag
Group=transcript-rag
WorkingDirectory=/opt/transcript-rag
Environment=TANTIVY_DIR=/opt/transcript-rag/data/tantivy
# Source the project venv before exec to pick up uvicorn, tantivy, fastapi.
ExecStart=/opt/transcript-rag/.venv/bin/uvicorn \
    services.tantivy_server.tantivy_server:app \
    --host 0.0.0.0 --port 8765 \
    --workers 1 --log-level info
Restart=on-failure
RestartSec=5
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
```

A copy of this unit lives at
[`services/tantivy_server/tantivy-sidecar.service`](../services/tantivy_server/tantivy-sidecar.service)
so it can be `cp`'d into `/etc/systemd/system/` and enabled with:

```bash
sudo cp services/tantivy_server/tantivy-sidecar.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now tantivy-sidecar
sudo systemctl status tantivy-sidecar
journalctl -u tantivy-sidecar -f
```

After bulk ingestion adds new documents to the index dir, force the sidecar
to pick them up:

```bash
curl -X POST http://localhost:8765/reload
```
