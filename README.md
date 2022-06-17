# RIDL
# The Representation of Imprecision in Deep Learning techniques(RIDL)
This is the supplementary of RIDL for the paper ["Representation of Imprecision in Deep Neural Networks for Image Classification: Active Learning via Belief Functions"].


`Detailed_examples_of_manually_report` folder: We reported the label correction of training images on different dataset in detail. NI represent the number of initial uncertainty samples that Identified by the first step of RIDL. NF represent the number of final uncertainty samples that manually corrected by human annotators. NM represent the number of manually corrected samples. Here are the detailed examples.

Table 3 The report of the samples need to be manually corrected by human annotators on different dataset.
RIDL	Imagewoof-5
(4687)	Flowers Recognition (2592)	Intel (11034)	CIFAR-10(50000)
networks	NI	NF	NM	NI	NF	NM	NI	NF	NM	NI	NF	NM
GoogLeNet	739	349	390	444	207	237	1231	695	536	2072	925	1147
MobileNetV2	437	155	282	222	66	156	890	361	529	4911	847	4064
DenseNet169	336	141	195	594	229	365	708	285	423	1577	430	1147
VGG16	350	138	212	343	98	245	1048	499	549	1070	312	758
ResNet101	393	141	252	856	253	603	953	374	579	1809	616	1193
EfficientNetB0	413	154	259	216	58	158	760	267	493	3192	932	2260
ShuffleNetV2	862	308	554	377	121	256	1234	659	575	4490	757	3733


`per_class_confusion_matrix.xlsx`: We counted the confusion matrices for all results and provide a detailed per-class analysis data.

We sincerely hope that our revised manuscript is now suitable for publication and that it will interest the readers of IEEE Transactions on Image Processing.
