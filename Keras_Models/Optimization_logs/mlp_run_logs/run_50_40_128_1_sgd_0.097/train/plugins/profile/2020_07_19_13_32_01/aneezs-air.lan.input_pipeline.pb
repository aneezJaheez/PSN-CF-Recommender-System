	
ףp=:I@
ףp=:I@!
ףp=:I@	6�fAJ��?6�fAJ��?!6�fAJ��?"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$
ףp=:I@�~j�t��?Au�VI@Y�Zd;�?*	     X�@2X
!Iterator::Model::ParallelMap::Zip��Q���?!Ɍ�!?�V@)u�V�?1y��
�HT@:Preprocessing2F
Iterator::Model��~j�t�?!�����"@)�~j�t��?1�H�t��@:Preprocessing2j
3Iterator::Model::ParallelMap::Zip[1]::ForeverRepeat9��v���?!��1���@)Zd;�O��?1�%�Z#�@:Preprocessing2S
Iterator::Model::ParallelMapy�&1��?!՗�ƞ@)y�&1��?1՗�ƞ@:Preprocessing2t
=Iterator::Model::ParallelMap::Zip[0]::FlatMap[0]::Concatenate{�G�z�?!ļ���@)y�&1��?1՗�ƞ�?:Preprocessing2�
MIterator::Model::ParallelMap::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice�~j�t�x?!�H�t���?)�~j�t�x?1�H�t���?:Preprocessing2v
?Iterator::Model::ParallelMap::Zip[1]::ForeverRepeat::FromTensor�~j�t�x?!�H�t���?)�~j�t�x?1�H�t���?:Preprocessing2d
-Iterator::Model::ParallelMap::Zip[0]::FlatMap�������?!�k��1�@){�G�zt?1ļ����?:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
device�Your program is NOT input-bound because only 0.2% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.no*no#You may skip the rest of this page.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	�~j�t��?�~j�t��?!�~j�t��?      ��!       "      ��!       *      ��!       2	u�VI@u�VI@!u�VI@:      ��!       B      ��!       J	�Zd;�?�Zd;�?!�Zd;�?R      ��!       Z	�Zd;�?�Zd;�?!�Zd;�?JCPU_ONLY