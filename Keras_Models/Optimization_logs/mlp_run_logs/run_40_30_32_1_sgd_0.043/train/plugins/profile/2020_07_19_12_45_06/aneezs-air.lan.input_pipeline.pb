	����sF@����sF@!����sF@	��Q�a�?��Q�a�?!��Q�a�?"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$����sF@Zd;�O��?AL7�A`EF@Y�� �rh�?*	     `o@2t
=Iterator::Model::ParallelMap::Zip[0]::FlatMap[0]::ConcatenateR���Q�?!r����B@)j�t��?1�+oI�!A@:Preprocessing2F
Iterator::Model�E���Ը?!w�V�RC@){�G�z�?1�Q�\�?@:Preprocessing2j
3Iterator::Model::ParallelMap::Zip[1]::ForeverRepeatj�t��?!�+oI�!1@)+�����?1�Ov�`/@:Preprocessing2S
Iterator::Model::ParallelMap�� �rh�?!ZEtJu@)�� �rh�?1ZEtJu@:Preprocessing2X
!Iterator::Model::ParallelMap::Zip��ʡE��?!�N��b�N@)�I+��?1�,<?��@:Preprocessing2�
MIterator::Model::ParallelMap::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice;�O��n�?!pI�!m�@);�O��n�?1pI�!m�@:Preprocessing2d
-Iterator::Model::ParallelMap::Zip[0]::FlatMap�������?!�2
��C@){�G�zt?1�Q�\��?:Preprocessing2v
?Iterator::Model::ParallelMap::Zip[1]::ForeverRepeat::FromTensor����Mbp?!FA@s}�?)����Mbp?1FA@s}�?:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
device�Your program is NOT input-bound because only 0.6% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.no*no#You may skip the rest of this page.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	Zd;�O��?Zd;�O��?!Zd;�O��?      ��!       "      ��!       *      ��!       2	L7�A`EF@L7�A`EF@!L7�A`EF@:      ��!       B      ��!       J	�� �rh�?�� �rh�?!�� �rh�?R      ��!       Z	�� �rh�?�� �rh�?!�� �rh�?JCPU_ONLY