# Compile TensorFlow CPP API from Source Code

## 0. Prerequisites
* protobuf (select proper version: see  ==tensorflow/tensorflow/workspace.bzl== in your cloned tensorflow repo)
* bazel (version reference [this link](https://www.tensorflow.org/install/source#tested_build_configurations))

## 1. Compilation and Installation
```
# select a tensorflow version to download
git clone https://github.com/tensorflow/tensorflow.git

cd tensorflow

./configure
chmod +x tensorflow/contrib/makefile/download_dependencies.sh
./tensorflow/contrib/makefile/download_dependencies.sh

bazel --output_user_root=/path/to/bazel_out //tensorflow:libtensorflow_cc.so
bazel --output_user_root=/path/to/bazel_out //tensorflow:libtensorflow_framework.so

# copy some files and folders
# set your save path to srcDir

cp -a bazel-genfiles/. 				$srcDir
cp -a tensorflow/cc 					$srcDir/tensorflow
cp -a tensorflow/core 				$srcDir/tensorflow
cp -a google/protobuf/src/google		$srcDir
cp -a third_party					$srcDir

# copy this eigen or next in third_party
cp -r tensorflow/contrib/makefile/downloads/eigen/unsupported $srcDir
cp -r tensorflow/contrib/makefile/downloads/eigen/Eigen $srcDir

# or next 
cp -r third_party/eigen3/unsupported $srcDir
cp -r third_party/eigen3/Eigen $srcDir

cp -r /usr/local/protobuf/include/google $srcDir
cp -r tensorflow/contrib/makefile/downloads/absl $srcDir

'''
.
├── absl
├── Eigen
├── external
├── google
├── tensorflow
├── third_party
└── unsupported
'''
```

## 3. Reference
* [https://github.com/node-tensorflow/node-tensorflow/blob/master/tools/install.sh](https://github.com/node-tensorflow/node-tensorflow/blob/master/tools/install.sh)
* [How to build and use Google TensorFlow C++ api](https://stackoverflow.com/questions/33620794/how-to-build-and-use-google-tensorflow-c-api)
* [Building an inference module using TensorFlow C++ API](https://medium.com/@dibyajyoti_20397/building-an-inference-module-in-tensorflow-c-api-5cac2096c0ec)
* [Tensorflow C++ Examples](https://github.com/rockzhuang/tensorflow)
* [Tensorflow 编译及应用C++ 动态库](https://ce39906.github.io/2018/09/10/Tensorflow-%E7%BC%96%E8%AF%91%E5%8F%8A%E5%BA%94%E7%94%A8C-%E5%8A%A8%E6%80%81%E5%BA%93/)    
* [用Tensorflow C++ API训练模型](https://spockwangs.github.io/blog/2018/01/13/train-using-tensorflow-c-plus-plus-api/)

* [Tensorflow c++ 实践及各种坑](https://cloud.tencent.com/developer/article/1006107)

* [Tensorflow C++ API调用训练好的模型](http://www.bearoom.xyz/2018/08/29/tensorflow-c-api%E8%B0%83%E7%94%A8%E8%AE%AD%E7%BB%83%E5%A5%BD%E7%9A%84%E6%A8%A1%E5%9E%8B/)