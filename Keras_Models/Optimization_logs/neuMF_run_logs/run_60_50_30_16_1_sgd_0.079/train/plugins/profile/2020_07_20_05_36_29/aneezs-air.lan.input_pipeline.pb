	}?5^�5`@}?5^�5`@!}?5^�5`@	�U��ߙ�?�U��ߙ�?!�U��ߙ�?"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$}?5^�5`@�Q����?A�n��`@Ysh��|?�?*	     ��@2S
Iterator::Model::ParallelMap��ʡE�?![�[�U@)��ʡE�?1[�[�U@:Preprocessing2j
3Iterator::Model::ParallelMap::Zip[1]::ForeverRepeat���Q��?!VUUUUU@))\���(�?1�8��8�@:Preprocessing2F
Iterator::Modelףp=
��?!�q�qV@)J+��?1�l�l@:Preprocessing2t
=Iterator::Model::ParallelMap::Zip[0]::FlatMap[0]::Concatenate�������?!r�q�@)L7�A`�?1wwwwww@:Preprocessing2�
MIterator::Model::ParallelMap::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice�� �rh�?!؂-؂-�?)�� �rh�?1؂-؂-�?:Preprocessing2X
!Iterator::Model::ParallelMap::Zip�p=
ף�?!�q�q'@)���Q��?1VUUUUU�?:Preprocessing2d
-Iterator::Model::ParallelMap::Zip[0]::FlatMap)\���(�?!�8��8�@){�G�zt?1�q�q�?:Preprocessing2v
?Iterator::Model::ParallelMap::Zip[1]::ForeverRepeat::FromTensor{�G�zt?!�q�q�?){�G�zt?1�q�q�?:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
device�Your program is NOT input-bound because only 0.8% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.no*no#You may skip the rest of this page.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	�Q����?�Q����?!�Q����?      ��!       "      ��!       *      ��!       2	�n��`@�n��`@!�n��`@:      ��!       B      ��!       J	sh��|?�?sh��|?�?!sh��|?�?R      ��!       Z	sh��|?�?sh��|?�?!sh��|?�?JCPU_ONLY