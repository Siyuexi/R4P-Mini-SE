
export BYTED_RAY_SERVE_RUN_HOST="::"
ray stop --force
# ray start --head --node-ip-address=2605:340:cd51:4900:2568:d735:e417:7adf --port=6379 --dashboard-host=0.0.0.0 --disable-usage-stats
ray start --head \
  --node-ip-address=[2605:340:cd51:4900:d40a:ace0:770b:51a2] \
  --dashboard-host=127.0.0.1 \
  --disable-usage-stats
python verl_utils/reward/model_server.py
# serve run verl_utils.reward.model_server:app_handle
# serve shutdown # for exit app