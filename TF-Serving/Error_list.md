## tf serving error list
* (1) run docker
```
<_Rendezvous of RPC that terminated with:
status = StatusCode.UNAVAILABLE
details = “Trying to connect an http1.x server”
debug_error_string = “{“created”:”@1551864894.132760000",“description”:“Error received from peer”,“file”:“src/core/lib/surface/call.cc”,“file_line”:1039,“grpc_message”:“Trying to connect an http1.x server”,“grpc_status”:14}"
```

* (2) input preprocess part ( input tensor position )
```
example_parsing_ops.cc:144 : Invalid argument: Could not parse example input
```

* (3) numpy error ( numpy version )
```
AttributeError: 'numpy.ufunc' object has no attribute '__module__'

>>> import numpy
>>> print numpy.__version__
1.15.4

//先卸载numpy，多卸载几次
pip uninstall numpy
//安装限定numpy版本
sudo pip install numpy==1.15.4

link: https://blog.csdn.net/qq_30460905/article/details/88956627
```

* (4) when [ ==pip install tensorflow-serving-api== ], ERROR: Cannot uninstall 'wrapt' 
```
pip install -U --ignore-installed wrapt（新wrapt包）
```
* (5) saver = tf.train.Saver(), ValueError: No variables to save
```
saver =tf.train.import_meta_graph("Model/model.ckpt.meta")
# https://blog.csdn.net/marsjhao/article/details/72829635

# https://www.cnblogs.com/sandy-t/p/8081807.html
```

* (6) error code from pb to savedmodel
```
# part 1, will not have not variables folders
# https://stackoverflow.com/questions/44329185/convert-a-graph-proto-pb-pbtxt-to-a-savedmodel-for-use-in-tensorflow-serving-o
import tensorflow as tf
from tensorflow.python.saved_model import signature_constants
from tensorflow.python.saved_model import tag_constants

export_dir = './saved'
graph_pb = 'my_quant_graph.pb'

builder = tf.saved_model.builder.SavedModelBuilder(export_dir)

with tf.gfile.GFile(graph_pb, "rb") as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())

sigs = {}

with tf.Session(graph=tf.Graph()) as sess:
    # name="" is important to ensure we don't get spurious prefixing
    tf.import_graph_def(graph_def, name="")
    g = tf.get_default_graph()
    inp = g.get_tensor_by_name("real_A_and_B_images:0")
    out = g.get_tensor_by_name("generator/Tanh:0")

    sigs[signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY] = \
        tf.saved_model.signature_def_utils.predict_signature_def(
            {"in": inp}, {"out": out})

    builder.add_meta_graph_and_variables(sess,
                                         [tag_constants.SERVING],
                                         signature_def_map=sigs)

builder.save()

# part 2, No variables to save
# https://zhuanlan.zhihu.com/p/60069860
def build_and_saved_wdl():
#1）导入pb文件，构造graph
  model_path = './model.pb'
  with tf.gfile.GFile(model_path, 'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
    graph = tf.Graph()   
  with graph.as_default():
    sess = tf.Session()
    tf.import_graph_def(graph_def, name='import')
    #恢复指定的tensor
    input_image = graph.get_tensor_by_name('import/image_tensor:0')
    output_ops = graph.get_tensor_by_name('import/num_boxes:0')
    variables_to_restore = slim.get_variables_to_restore()
  
    saver = tf.train.Saver(variables_to_restore)
    saver.restore(sess, tf.train.latest_checkpoint(checkpoint_path))
  
  
    print('Exporting trained model to', export_path)
    #构造定义一个builder，并制定模型输出路径
    builder = tf.saved_model.builder.SavedModelBuilder(export_path)
    #声明模型的input和output
    tensor_info_input = tf.saved_model.utils.build_tensor_info(input_images)
    tensor_info_output = tf.saved_model.utils.build_tensor_info(output_result)
    #定义签名
    prediction_signature = (
  	  tf.saved_model.signature_def_utils.build_signature_def(
      inputs={'images': tensor_info_input},
      outputs={'result': tensor_info_output},
      method_name=tf.saved_model.signature_constants.PREDICT_METHOD_NAME))
    builder.add_meta_graph_and_variables(sess, [tf.saved_model.tag_constants.SERVING],signature_def_map={'predict_images': prediction_signature})
    builder.save()
    print('Done exporting!')
```
* (7) useless in preprocess

```
# preprocessing layer on keras model to accept input string image for tensorflow serving
# https://github.com/tensorflow/serving/issues/878
    string_inp = tf.placeholder(tf.string, shape=(None,))
    imgs_map = tf.map_fn(
        tf.image.decode_image,
        string_inp,
        dtype=tf.uint8
    )
    imgs_map.set_shape((None, None, None, 3))
    imgs = tf.image.resize_images(imgs_map, [120, 160])
    imgs = tf.reshape(imgs, (-1, 120, 160, 3))
    img_float = tf.cast(imgs, dtype=tf.float32) / 255 - 0.5
# Then I loaded keras model and defined output tensor by calling model object:

    from keras.models import load_model
    model = load_model("...")
    output = model(img_float)
# Finally I used string_inp and output to define the prediction signature.
```
* (8) Tensorflow Serving Transform data from input function

```
# https://stackoverflow.com/questions/55271842/tensorflow-serving-transform-data-from-input-function
# https://github.com/tensorflow/transform/blob/master/examples/sentiment_example.py

```
* (9) Python: Convert Image to String, Convert String to Image

```
import base64
# Convert Image to String 
with open("t.png", "rb") as imageFile:
    str = base64.b64encode(imageFile.read())
#    str = base64.b64encode(imageFile.read()).encode('utf-8')
    print str
    
# Convert String to Image    
fh = open("imageToSave.png", "wb")
fh.write(str.decode('base64'))
fh.close()
```

* (10) multi pb model to merge

```
# error: input_map must be a dictionary mapping strings to Tensor objects.
# solve: input_map={'image_tensor:0': input_image}
# not input_map={'image_tensor:0', input_image} (XXXXXXXX)



with tf.Graph().as_default() as g_combined:
	with tf.Session(graph=g_combined) as sess:
		graph_def_detect = load_def(detect_pb_path)
		graph_def_seg= load_def(seg_pb_path)
		input_image = tf.placeholder(dtype=tf.uint8,shape=[1,None,None,3], name="image")#定义新的网络输入
		input_image1 = tf.placeholder(dtype=tf.float32,shape=[1,None,None,3], name="image1")
		#将原始网络的输入映射到input_image(节点为：新的输入节点“image”)
		detection = tf.import_graph_def(graph_def_detect, input_map={'image_tensor:0': input_image},return_elements=['detection_boxes:0', 'detection_scores:0','detection_classes:0','num_detections:0' ])
                #新的输出节点为“detect”
		tf.identity(detection, 'detect')
		# second graph load
		seg_predict = tf.import_graph_def(graph_def_seg, input_map={"create_inputs/batch:0": input_image1}, return_elements=["conv6/out_1:0"])
		tf.identity(seg_predict, "seg_predict")
 
		# freeze combined graph
		g_combined_def = graph_util.convert_variables_to_constants(sess, sess.graph_def, ["seg_predict","detect"])
                #合成大图，生成新的pb
		tf.train.write_graph(g_combined_def, out_pb_path, 'merge_model.pb', as_text=False)
# ———————————————— 
#版权声明：本文为CSDN博主「hustwayne」的原创文章，遵循CC 4.0 #by-sa版权协议，转载请附上原文出处链接及本声明。
#原文链接：https://blog.csdn.net/hustwayne/article/details/89482873

# and https://blog.csdn.net/qq26983255/article/details/85797707
```
* (11)  [tf serving signature constant lists](https://www.tensorflow.org/api_docs/python/tf/saved_model/signature_constants)

```
Other Members
CLASSIFY_INPUTS = 'inputs'
CLASSIFY_METHOD_NAME = 'tensorflow/serving/classify'
CLASSIFY_OUTPUT_CLASSES = 'classes'
CLASSIFY_OUTPUT_SCORES = 'scores'
DEFAULT_SERVING_SIGNATURE_DEF_KEY = 'serving_default'
PREDICT_INPUTS = 'inputs'
PREDICT_METHOD_NAME = 'tensorflow/serving/predict'
PREDICT_OUTPUTS = 'outputs'
REGRESS_INPUTS = 'inputs'
REGRESS_METHOD_NAME = 'tensorflow/serving/regress'
REGRESS_OUTPUTS = 'outputs'
```

* (12) docker: Error response from daemon: driver failed programming external connectivity on endpoint dazzling_mendel

```
端口被占用导致的
```
* (13) StatusCode.INVALID_ARGUMENT details = "NodeDef mentions attr 'unit' not in Op, saved_model used tensorflow version and client tensorflow version are different, should use one tensorflow version, ref [Executor failed to create kernel. Invalid argument: NodeDef mentions attr 'index_type' not in Op](https://github.com/tensorflow/serving/issues/831)

```
tatus = StatusCode.INVALID_ARGUMENT
	details = "NodeDef mentions attr 'unit' not in Op<name=Substr; signature=input:string, pos:T, len:T -> output:string; attr=T:type,allowed=[DT_INT32, DT_INT64]>; NodeDef: {{node import/map/while/decode_image/Substr}} = Substr[T=DT_INT32, _output_shapes=[<unknown>], unit="BYTE", _device="/job:localhost/replica:0/task:0/device:CPU:0"](import/map/while/TensorArrayReadV3, import/map/while/decode_image/Substr/pos, import/map/while/decode_image/Substr/len). (Check whether your GraphDef-interpreting binary is up to date with your GraphDef-generating binary.).
	 [[{{node import/map/while/decode_image/Substr}} = Substr[T=DT_INT32, _output_shapes=[<unknown>], unit="BYTE", _device="/job:localhost/replica:0/task:0/device:CPU:0"](import/map/while/TensorArrayReadV3, import/map/while/decode_image/Substr/pos, import/map/while/decode_image/Substr/len)]]"
	debug_error_string = "{"created":"@1574932547.518516747","description":"Error received from peer","file":"src/core/lib/surface/call.cc","file_line":1039,"grpc_message":"NodeDef mentions attr 'unit' not in Op<name=Substr; signature=input:string, pos:T, len:T -> output:string; attr=T:type,allowed=[DT_INT32, DT_INT64]>; NodeDef: {{node import/map/while/decode_image/Substr}} = Substr[T=DT_INT32, _output_shapes=[<unknown>], unit="BYTE", _device="/job:localhost/replica:0/task:0/device:CPU:0"](import/map/while/TensorArrayReadV3, import/map/while/decode_image/Substr/pos, import/map/while/decode_image/Substr/len). (Check whether your GraphDef-interpreting binary is up to date with your GraphDef-generating binary.).\n\t [[{{node import/map/while/decode_image/Substr}} = Substr[T=DT_INT32, _output_shapes=[<unknown>], unit="BYTE", _device="/job:localhost/replica:0/task:0/device:CPU:0"](import/map/while/TensorArrayReadV3, import/map/while/decode_image/Substr/pos, import/map/while/decode_image/Substr/len)]]","grpc_status":3}"
```

* (14) **request port error** example

```
grpc._channel._Rendezvous: <_Rendezvous of RPC that terminated with:
	status = StatusCode.UNAVAILABLE
	details = "failed to connect to all addresses"
	debug_error_string = "{"created":"@1574942452.038365732","description":"Failed to pick subchannel","file":"src/core/ext/filters/client_channel/client_channel.cc","file_line":3818,"referenced_errors":[{"created":"@1574942452.038357275","description":"failed to connect to all addresses","file":"src/core/ext/filters/client_channel/lb_policy/pick_first/pick_first.cc","file_line":395,"grpc_status":14}]}"
```

* (15) **url writing form error** example: should replace https://xx.xxx.xxx.xxx:5001 (:frowning_face:)  by xx.xxx.xxx.xxx:5001 (:grinning:)
```
grpc._channel._Rendezvous: <_Rendezvous of RPC that terminated with:
        status = StatusCode.UNAVAILABLE
        details = "Name resolution failure"
        debug_error_string = "{"created":"@1577698071.277316602","description":"Failed to create subchannel","file":"src/core/ext/filters/client_channel/client_channel.cc","file_line":2267,"referenced_errors":[{"created":"@1577698071.277310755","description":"Name resolution failure","file":"src/core/ext/filters/client_channel/request_routing.cc","file_line":166,"grpc_status":14}]}"
```
