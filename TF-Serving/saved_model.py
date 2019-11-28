import tensorflow as tf
import pdb
import os
import tensorflow.contrib.slim as slim
from tensorflow.python.util import compat

## VGG mean pixel
R_MEAN = 123.68
G_MEAN = 116.779
B_MEAN = 103.939

# Subtracts the given means from each image channel.
def mean_image_subtraction(image, means):
    if image.get_shape().ndims != 3:
        raise ValueError('Input Tensor image must be of shape [height, width, 3]')

    num_channels = image.get_shape().as_list()[-1]
    if num_channels != 3:
        raise ValueError('Input Tensor image must have 3 channels')

    if len(means) != num_channels:
        raise ValueError('len(means) must match the number of channels')

    rgb_channels = tf.split(image, num_channels, 2)
    for i in range(num_channels):
        # print('____________, i={}, aftrer split, channels shape is: {}'.format(i, rgb_channels[i].get_shape()))
        rgb_channels[i] -= means[i]

    return tf.concat(rgb_channels, 2)

# ERROR Link: preprocessing layer on keras model to accept input string image for tensorflow serving
# https://github.com/tensorflow/serving/issues/878 (not recommand)
def preprocess_image(image_buffer):
    image = tf.image.decode_image(image_buffer, channels=3)
    image_shape = tf.shape(image)
    image.set_shape([None, None, 3])
    image = tf.image.resize_images(image, [480, 480])
    img_float = tf.cast(image, dtype=tf.float32)
#    img_float = mean_image_subtraction(img_float, [R_MEAN, G_MEAN, B_MEAN])  # v1
    img_float = mean_image_subtraction(img_float, [B_MEAN, G_MEAN, R_MEAN])   # v2
#   image = tf.reshape(imgs, (-1, 480, 480, 3))
#   img_float = tf.cast(image, dtype=tf.float32) / 255
    return img_float

# https://zhuanlan.zhihu.com/p/60069860 error
# https://stackoverflow.com/questions/44329185/convert-a-graph-proto-pb-pbtxt-to-a-savedmodel-for-use-in-tensorflow-serving-o
def load_merge_graph(frozen_graph_filename):
    # We load the protobuf file from the disk and parse it to retrieve the
    # unserialized graph_def
    with tf.gfile.GFile(frozen_graph_filename, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())

    # Then, we import the graph_def into a new Graph and returns it
    with tf.Graph().as_default() as graph:
        # The name var will prefix every op/nodes in your graph
        # Since we load everything in a new graph, this is not needed
        serialized_tf_example = tf.placeholder(tf.string, name='tf_example')
        feature_configs = {
            'image/encoded': tf.FixedLenFeature(
            shape=[], dtype=tf.string),
        }
        tf_example = tf.parse_example(serialized_tf_example, feature_configs)
        jpegs = tf_example['image/encoded']
        images = tf.map_fn(preprocess_image, jpegs, dtype=tf.float32)
#        tf.identity(images, "input")
        tf.import_graph_def(graph_def, input_map={'input:0': images},
                                          return_elements=["ofuse/Sigmoid:0"],
                                          name="edge")
#        tf.identity(predictions, "prediction")
    # merged_model.pb uses byte stream as input
    tf.train.write_graph(graph, './', 'merged_model.pb', as_text=False)

def export_savedmodel(pb_path, checkpoint_path):
  model_path = "saved_model"
  model_version = 1
  with tf.gfile.GFile(pb_path, 'rb') as f:
       graph_def = tf.GraphDef()
       graph_def.ParseFromString(f.read())
       graph = tf.Graph()
  with graph.as_default():
      sess = tf.Session()
      tf.import_graph_def(graph_def, name='import')
      g = tf.get_default_graph()
      for op in g.get_operations():
          print(op.name)
      inp = g.get_tensor_by_name("import/ParseExample/ParseExample:0")
      out = g.get_tensor_by_name("import/edge/ofuse/Sigmoid:0")
      saver = tf.train.import_meta_graph(tf.train.latest_checkpoint(checkpoint_path) + '.meta')
      saver.restore(sess, tf.train.latest_checkpoint(checkpoint_path))

      model_input_tensor_info = tf.saved_model.utils.build_tensor_info(inp)
      model_output_tensor_info = tf.saved_model.utils.build_tensor_info(out)
      model_signature = tf.saved_model.signature_def_utils.build_signature_def(
          inputs={'input': model_input_tensor_info}, outputs={'output': model_output_tensor_info},
          method_name=tf.saved_model.signature_constants.
                        PREDICT_METHOD_NAME)
      export_path = os.path.join(compat.as_bytes(model_path), compat.as_bytes(str(model_version)))
      builder = tf.saved_model.builder.SavedModelBuilder(export_path)
      builder.add_meta_graph_and_variables(
          sess=sess,
          tags=[tf.saved_model.tag_constants.SERVING],
          signature_def_map={
              tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY:
              model_signature
          })
      builder.save()

if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"]= '0'
    input_model_path = './tf_model/hed.pb'
    load_merge_graph(input_model_path)
    export_savedmodel('merged_model.pb', './tf_model')
