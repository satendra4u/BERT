# BERT Question Answering Inference with Mixed Precision

1. Overview
Bidirectional Embedding Representations from Transformers (BERT), is a method of pre-training language representations which obtains state-of-the-art results on a wide array of Natural Language Processing (NLP) tasks.

This version of BERT 19.10 is an optimized version of Google's official implementation, leveraging mixed precision arithmetic and tensor cores on NVidia Tesla V100 GPUS for faster training times while maintaining target accuracy.

</br>1.a Learning objectives

</br>This notebook demonstrates:
</br>Inference on QA task with BERT Large model
</br>The use/download of fine-tuned NVIDIA BERT models
</br>Use of Mixed Precision for Inference

2. Requirements:


</br> a) GPU -  if you are running this example in collab, before running this notebook, please set the Colab runtime environment to GPU via the menu Runtime => Change runtime type => GPU.

This demo will basically work on any NVIDIA GPU with CUDA cores, though for improved FP16 inference, a Volta, Turing or newer generation GPU with Tensor cores is desired. 

On Google Colab, this normally means a T4 GPU. If you are assigned an older K80 GPU, another trial at another time might give you a T4 GPU.

</br>

#Select lower version of tensroflow on Google Colab
%tensorflow_version 1.x
import tensorflow
print(tensorflow.__version__)

!nvidia-smi

+-----------------------------------------------------------------------------+
| NVIDIA-SMI 460.32.03    Driver Version: 460.32.03    CUDA Version: 11.2     |
|-------------------------------+----------------------+----------------------+
| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|                               |                      |               MIG M. |
|===============================+======================+======================|
|   0  Tesla T4            Off  | 00000000:00:04.0 Off |                    0 |
| N/A   38C    P8     9W /  70W |      0MiB / 15109MiB |      0%      Default |
|                               |                      |                  N/A |
+-------------------------------+----------------------+----------------------+
                                                                               
+-----------------------------------------------------------------------------+
| Processes:                                                                  |
|  GPU   GI   CI        PID   Type   Process name                  GPU Memory |
|        ID   ID                                                   Usage      |
|=============================================================================|

