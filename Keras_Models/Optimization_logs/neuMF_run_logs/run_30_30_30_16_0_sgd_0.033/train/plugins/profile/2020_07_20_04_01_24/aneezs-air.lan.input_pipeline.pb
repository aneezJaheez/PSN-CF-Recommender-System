	���Mb0M@���Mb0M@!���Mb0M@	,��@,��@!,��@"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$���Mb0M@y�&1��?A��x�&�K@Yu�V@*	     ڧ@2S
Iterator::Model::ParallelMap��C�l�@!���qW@)��C�l�@1���qW@:Preprocessing2F
Iterator::Model��|?5^@!���I�W@)V-��?1��ue�?:Preprocessing2j
3Iterator::Model::ParallelMap::Zip[1]::ForeverRepeatX9��v��?!Ė��? @)�v��/�?17�8L��?:Preprocessing2t
=Iterator::Model::ParallelMap::Zip[0]::FlatMap[0]::ConcatenateZd;�O��?!��d��?)���Q��?1�#	�q�?:Preprocessing2X
!Iterator::Model::ParallelMap::ZipL7�A`��?!~��bK@);�O��n�?1�8L���?:Preprocessing2�
MIterator::Model::ParallelMap::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice����Mb�?!���C8��?)����Mb�?1���C8��?:Preprocessing2d
-Iterator::Model::ParallelMap::Zip[0]::FlatMap9��v���?!��n{@�?)�~j�t�x?1q�e�'�?:Preprocessing2v
?Iterator::Model::ParallelMap::Zip[1]::ForeverRepeat::FromTensor{�G�zt?!�°T���?){�G�zt?1�°T���?:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is MODERATELY input-bound because 5.2% of the total step time sampled is waiting for input. Therefore, you would need to reduce both the input time and other time.no*no>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	y�&1��?y�&1��?!y�&1��?      ��!       "      ��!       *      ��!       2	��x�&�K@��x�&�K@!��x�&�K@:      ��!       B      ��!       J	u�V@u�V@!u�V@R      ��!       Z	u�V@u�V@!u�V@JCPU_ONLY