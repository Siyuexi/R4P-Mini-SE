
### Modify "MODEL_PATH" in model_server.py and "SERVER_URL" in model_client.py

# sudo apt-get install net-tools
# export BYTED_RAY_SERVE_RUN_HOST="::"

pip install -e .

ray stop --force
ray start --head \
  --node-ip-address=127.0.0.1 \
  --dashboard-host=127.0.0.1 \
  --disable-usage-stats
python verl_utils/reward/model_server.py

echo "Server is running. Press Ctrl+C to stop."
while true; do
    sleep 3600
    echo "[Heartbeat] Server is still running at $(date)"
done
# serve shutdown # for exit app