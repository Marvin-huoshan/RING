#!/usr/bin/env bash
IID=${1:-qty}
GPU=${2:-0}
EPS=${3:-1}
NA=${4:-6}
# Common args
FRAC=0.25
#NA=4
EPOCHS=100
BS=500
LOCAL_EP=5
LR=0.05
CLIP=10.0
#EPS=5
DELTA=1e-5
PDR=0.5
DEFENSE=None


for ATTACK in Input Output Collusion; do
#for ATTACK in C_2 C_3; do
  SESSION="DP_A_${ATTACK}_D_${DEFENSE}_F=${FRAC}_NA=${NA}_iid_${IID}_e_${EPS}_c_${CLIP}_lr_${LR}_PDR_${PDR}_m"
  CMD="python BD_Attack_Collusion.py \
    --dataset mnist \
    --lr ${LR} \
    --dp_mechanism MA \
    --dp_epsilon ${EPS} \
    --dp_delta ${DELTA} \
    --dp_clip ${CLIP} \
    --dp_sample 1 \
    --attack_type ${ATTACK} \
    --frac ${FRAC} \
    --num_attacker ${NA} \
    --attack \
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
