#!/bin/bash
# compiled tensorflow_model_server home path
TF_SERVING_HOME=/home/CV_Libs/tf-serving-r1.12/bazel-bin
# converted saved_model path
SERVING_MODEL_HOME=/home/tf_serving_tutorials/saved_model/

# boot your model serving
$TF_SERVING_HOME/tensorflow_serving/model_servers/tensorflow_model_server --port=18592 \
        --model_name=edge \
        --model_base_path=$SERVING_MODEL_HOME
