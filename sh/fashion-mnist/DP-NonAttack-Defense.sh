#!/usr/bin/env bash

# Accept the defense name as $1 (default to “Deepsight” if not provided)
DEFENSE=${1:-Deepsight}
IID=${2:-qty}
GPU=${3:-0}
EPS=${4:-5}
NA=${5:-0}
# Common args
FRAC=0.25
#NA=0
EPOCHS=100
BS=500
LOCAL_EP=5
LR=0.05
CLIP=10.0
#EPS=5
DELTA=1e-5
PDR=0


for ATTACK in None; do
  SESSION="DP_A_${ATTACK}_D_${DEFENSE}_F=${FRAC}_NA=${NA}_iid_${IID}_e_${EPS}_c_${CLIP}_lr_${LR}_PDR_${PDR}_m"
  CMD="python BD_Attack_Collusion.py \
    --dataset fashion-mnist \
    --lr ${LR} \
    --dp_mechanism MA \
    --dp_epsilon ${EPS} \
    --dp_delta ${DELTA} \
    --dp_clip ${CLIP} \
    --dp_sample 1 \
    --attack_type ${ATTACK} \
    --frac ${FRAC} \
    --num_attacker ${NA} \
    --epochs ${EPOCHS} \
    --bs ${BS} \
    --local_ep ${LOCAL_EP} \
    --defense ${DEFENSE} \
    --gpu ${GPU} \
    --PDR ${PDR} \
    --iid ${IID} \
    --re_weight"
  # detach a screen with that session name and run the command
  screen -dmS "${SESSION}" bash -c "${CMD}"
done
