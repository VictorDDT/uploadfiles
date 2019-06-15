#!/bin/bash

MODEL_NAME=faster_rcnn_inception_resnet_v2_atrous_oid_2018_01_28

PIPELINE_CONFIG_PATH=/storage/Gray_Whales/models/${MODEL_NAME}/pipeline.config
MODEL_DIR=/storage/Gray_Whales/models/${MODEL_NAME}/
NUM_TRAIN_STEPS=50000
SAMPLE_1_OF_N_EVAL_EXAMPLES=1

echo ""
echo "PIPELINE_CONFIG_PATH: "${PIPELINE_CONFIG_PATH}
echo "MODEL_DIR: "${MODEL_DIR}
echo ""

# python3 /opt/develop/virtualenv-tf/lib/python3.6/site-packages/tensorflow/models/research/object_detection/model_main.py \
#     --pipeline_config_path=${PIPELINE_CONFIG_PATH} \
#     --model_dir=${MODEL_DIR} \
#     --num_train_steps=${NUM_TRAIN_STEPS} \
#     --sample_1_of_n_eval_examples=$SAMPLE_1_OF_N_EVAL_EXAMPLES \
#     --alsologtostderr

python3 /opt/develop/virtualenv-tf/lib/python3.6/site-packages/tensorflow/models/research/object_detection/legacy/train.py \
    --pipeline_config_path=${PIPELINE_CONFIG_PATH} \
    --train_dir=${MODEL_DIR} \
    --num_train_steps=${NUM_TRAIN_STEPS} \
    --sample_1_of_n_eval_examples=$SAMPLE_1_OF_N_EVAL_EXAMPLES \
    --alsologtostderr
