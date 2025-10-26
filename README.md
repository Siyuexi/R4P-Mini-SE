# R4P-Mini-SE

This repository contains the official implementation of the paper: "Scalable Supervising Software Agents with Patch Reasoner".

In this paper, we explore a reasoning-based patch verification strategy to provide scalable supervision for software engineering agents. This approach (1) mitigates data scarcity caused by test quality requirements in open-source codebases, (2) removes the need for environment setup and makes data expansion costless, and (3) greatly reduces computational overhead compared to heavy test execution. We aim to leverage such imperfect yet easily scalable supervision to enhance model capability even after high-quality test data is exhausted.

## Setup

Our framework is based on [Verl](https://github.com/volcengine/verl). To install our environment, please refer to the Verl repo.

## Data

You can find our training and testing data [here](https://drive.google.com/drive/folders/1vgPpnxO0WfAXGPl8VOpKUQk5fDDha4mT?usp=drive_link). Please create the `datasets` directory and save them to `datasets`.

- `info_xxx.parquet`: Original data without prompt.
- `data_xxx.parquet`: Data with prompt. You can process a `info` parquet to `data` parquet by using `verl_utils/data/data_proc.py`

## Run

You can find our scripts in `verl_utils/scripts`:

- `r4p.sh`: Training and testing scripts of R4P.
- `minise.sh`: Training and testing scripts of Mini-SE.
- `eval.sh`: Evaluation scripts of R4P individually.
- `tts.sh`: Evaluation scripts of Mini-SE (test-time patch selection).
- `setup.sh`: Script for serving R4P. You may need to adjust `MODEL_PATH` in `verl_utils/model_server.py` and `SERVER_URL` in `verl_utils/reward/model_client.py`.

Before training Mini-SE, please use `verl_utils/data/env_init.py` to checkout the repositories first.

## Citation

TODO.
