#!/bin/sh
# Ensure /data is owned by the app user (volume is initially owned by root)
mkdir -p /data
chown -R 1000:1000 /data

# Drop privileges and exec the app
exec gosu 1000:1000 uvicorn main:app --host 0.0.0.0 --port 8000
