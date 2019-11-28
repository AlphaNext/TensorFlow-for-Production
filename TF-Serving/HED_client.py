# Copyright 2016 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Send JPEG image to tensorflow_model_server loaded with ResNet model.

"""

from __future__ import print_function

# This is a placeholder for a Google-internal import.

import grpc
import requests
import tensorflow as tf
from time import time
from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2_grpc
import pdb
# scipy lib maybe error in python3.6.x
import scipy.misc as spm
import imageio
import numpy as np
from tensorflow.core.framework import types_pb2
# The image URL is the location of the image we should send to the server
IMAGE_URL = 'https://tensorflow.org/images/blogs/serving/cat.jpg'

# change address and port 
tf.app.flags.DEFINE_string('server', 'localhost:12549',
                           'PredictionService host:port')
tf.app.flags.DEFINE_string('image', 'Lenna.png', 'path to image in JPEG format')
FLAGS = tf.app.flags.FLAGS


def main(_):
  if FLAGS.image:
    with open(FLAGS.image, 'rb') as f:
      print(FLAGS.image)
      data = f.read()
  else:
    # Download the image since we weren't given one
    dl_request = requests.get(IMAGE_URL, stream=True)
    dl_request.raise_for_status()
    data = dl_request.content

  channel = grpc.insecure_channel(FLAGS.server)
  stub = prediction_service_pb2_grpc.PredictionServiceStub(channel)
  # Send request
  # See prediction_service.proto for gRPC request/response details.
  start = time()
  request = predict_pb2.PredictRequest()
  request.model_spec.name = 'edge'
  request.model_spec.signature_name = 'serving_default'
  request.inputs['input'].CopyFrom(
      tf.contrib.util.make_tensor_proto(data, shape=[1]))
  response = stub.Predict(request, 100.0)  # 10 secs timeout
  results = {}
  for key in response.outputs:
    tensor_proto = response.outputs[key]
    nd_array = tf.contrib.util.make_ndarray(tensor_proto)
    results[key] = nd_array
  # print(key, nd_array)
  print(key, nd_array.shape)
  # pdb.set_trace()
  reshape_out = np.reshape(results['output'], (480,480))
  binary_out = np.where(reshape_out>0.5, 255, 0)
  np.savetxt('debug.txt', binary_out);
  stop = time()
  print( str(1000*(stop-start)) + " ms")
  imageio.imwrite('edge.jpg', binary_out)
  # maybe error image high scipy verion in python 3.6.x
  # spm.imsave("edge.jpg", binary_out)
  print("####### all output info ########")
  # print(response)


if __name__ == '__main__':
  tf.app.run()
