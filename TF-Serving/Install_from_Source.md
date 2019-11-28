# tf serving install and compile

## build from source without docker
### 1) Prerequisites   

```
# gRPC, bazel
# and the following

sudo yum install -y \
        automake \
        build-essential \
        curl \
        libcurl3-dev \
        git \
        libtool \
        libfreetype6-dev \
        libpng12-dev \
        libzmq3-dev \
        pkg-config \
        python-dev \
        python-numpy \
        python-pip \
        software-properties-common \
        swig \
        zip \
        zlib1g-dev

# install nccl, make PREFIX=/usr install, PREFIX to set install path
git clone https://github.com/NVIDIA/nccl.git
cd nccl
make -j12 src.build CUDA_HOME=/usr/local/cuda
make PREFIX=/usr/local/nccl2 install

```
### 2) Build tf serving   
```
# --recurse-submodules option is very important
git clone --recurse-submodules -b r1.12 https://github.com/tensorflow/serving 
cd serving

# set the following version according to your environment
export TF_NEED_CUDA=1
export TF_CUDA_VERSION=9.0
export TF_CUDNN_VERSION=7
export TF_NCCL_VERSION='2.4.8'  
export NCCL_HDR_PATH=/usr/local/nccl2/include
export NCCL_INSTALL_PATH=/usr/local/nccl2/lib

# ref link https://github.com/tensorflow/tensor2tensor/issues/687
bazel query 'kind(rule, @local_config_cuda//...)'  # key for serve model with gpu

# set output log position, you could use --output_user_root or export TEST_TMPDIR=/path/to/directory
# such as: bazel --output_user_root=/path/to/directory build //foo:bar       and ref the next link
# https://stackoverflow.com/questions/40775920/how-can-i-make-bazel-use-external-storage-when-building

bazel --output_user_root=/home/dev_CV/tf-serving-r1.12/tf_bazel_out build --config=cuda --copt="-fPIC" -c opt --copt=-mavx --copt=-msse4.2 tensorflow_serving/model_servers:tensorflow_model_server

# for run python client code
pip install tensorflow-serving-api
```

### 3) More useful reference links   

* [Installing ModelServer from Source](https://github.com/tensorflow/serving/blob/master/tensorflow_serving/g3doc/setup.md)
* [TensorFlow serving 安装教程与使用（1）](https://blog.csdn.net/sthp888/article/details/82499060)
* [TensorFlow serving 安装教程与使用（2）](https://blog.csdn.net/sthp888/article/details/82500456)
* [tensorflow server gpu 编译安装总结](https://zhuanlan.zhihu.com/p/64826067)
* [Fork me on GitHub
Tensorflow Serving CentOS 7源码编译（CPU + GPU版）](https://www.dearcodes.com/index.php/archives/25/)
* [How to install TensorFlow Serving, load a Saved TF Model and connect it to a REST API in Ubuntu 16.04](https://medium.com/@noone7791/how-to-install-tensorflow-serving-load-a-saved-tf-model-and-connect-it-to-a-rest-api-in-ubuntu-48e2a27b8c2a)
* [4.5 Tensorflow Serving | Tensorflow Serving](https://bookdown.org/leovan/TensorFlow-Learning-Notes/4-5-deploy-tensorflow-serving.html)
* [TensorFlow Serving Note](https://notes-by-yangjinjie.readthedocs.io/zh_CN/latest/python/12-%E6%9C%BA%E5%99%A8%E5%AD%A6%E4%B9%A0/01-TensorFlow/04-TensorFlow_Serving/README.html)
* [Build TensorFlow Serving From Source without Docker](https://yinguobing.com/build-tensorflow-serving-from-source-without-docker/)


