	�$���X@�$���X@!�$���X@	�#P���?�#P���?!�#P���?"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$�$���X@q=
ףp�?Am����X@Y���(\��?*	      g@2j
3Iterator::Model::ParallelMap::Zip[1]::ForeverRepeaty�&1��?!�7��Mo>@)�&1��?1ԛ����;@:Preprocessing2F
Iterator::Model��~j�t�?!zӛ���D@)
ףp=
�?1ozӛ�t8@:Preprocessing2S
Iterator::Model::ParallelMapX9��v��?!�,d!�0@)X9��v��?1�,d!�0@:Preprocessing2t
=Iterator::Model::ParallelMap::Zip[0]::FlatMap[0]::ConcatenateL7�A`�?!�7��M�1@)/�$��?1��Moz�&@:Preprocessing2X
!Iterator::Model::ParallelMap::Zip���S㥻?!�,d!YM@)�~j�t��?1!Y�B@:Preprocessing2�
MIterator::Model::ParallelMap::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice�~j�t��?!!Y�B@)�~j�t��?1!Y�B@:Preprocessing2d
-Iterator::Model::ParallelMap::Zip[0]::FlatMap{�G�z�?!����7�5@)y�&1�|?1�7��Mo@:Preprocessing2v
?Iterator::Model::ParallelMap::Zip[1]::ForeverRepeat::FromTensor{�G�zt?!����7�@){�G�zt?1����7�@:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
device�Your program is NOT input-bound because only 0.1% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.no*no#You may skip the rest of this page.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	q=
ףp�?q=
ףp�?!q=
ףp�?      ��!       "      ��!       *      ��!       2	m����X@m����X@!m����X@:      ��!       B      ��!       J	���(\��?���(\��?!���(\��?R      ��!       Z	���(\��?���(\��?!���(\��?JCPU_ONLY