YOUR_SERVER_IP=your_ip
ray stop --force
ray start --head \
  --node-ip-address=$YOUR_SERVER_IP \
  --dashboard-host=127.0.0.1 \
  --disable-usage-stats
python verl_utils/reward/model_server.py
# serve run verl_utils.reward.model_server:app_handle
# serve shutdown # for exit app