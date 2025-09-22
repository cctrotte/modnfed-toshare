#!/bin/bash

while true; do
    rsync -a --exclude "run.lock" wandb/ ~/wandb_off/
    sleep 3600  # sync every hour
done
