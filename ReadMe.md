Community detection becomes an important problem with the booming of social networks. The Medoid-Shift algorithm preserves the benefits of Mean-Shift and can be applied to problems based on distance matrix, such as community detection. One drawback of the Medoid-Shift algorithm is that there may be no data points within the neighborhood region defined by a distance parameter.
To deal with the problem, a new algorithm called Revised Medoid-Shift (RMS) is proposed. During the process of finding the next medoid, the RMS algorithm is based on a neighborhood defined by KNN, while the original Medoid-Shift is based on a neighborhood defined by a distance parameter. Since the neighborhood defined by KNN is more stable than the one defined by the distance parameter
in terms of the number of data points within the neighborhood, the RMS algorithm may converge more smoothly. The RMS algorithm is tested on two kinds of datasets including community datasets with known ground truth partition and community datasets without ground truth partition respectively. The experiment results show that the proposed RMS algorithm generally produces better results
than Medoid-Shift and some state-of-the-art together with most classic community detection algorithms on different kinds of community detection datasets.<br>
<br> <br> <br>
Since there are many codes written for different other algorithms, I've condensed most of them to a .py file which is a pipeline for comparing our method to other algorithms. FOR THE CODES in the py file, you can find our method in "class MeanShift_knn", and the file is a combination of evaluating the performance for other different methods. <br> <br> 
The arxiv version of this method can be found at: https://arxiv.org/ftp/arxiv/papers/2304/2304.09512.pdf <br>
The published version of this method can be found at: https://link.springer.com/chapter/10.1007/978-981-99-4752-2_29 <br>
