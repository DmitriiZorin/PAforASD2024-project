# PAforASD2024 project
k-means clustering algorithm parallelization project for ITMO Parallel algorithms for the analysis and synthesis of data course.

## Installation
1. Clone repo
2. Make python venv and activate it
3. pip install freeze.txt

## General moments
In this project sequentive and parallel version of k-means clustering algorithm was coded. 
Idea of parallelization is to divide orginal array of data into small chunks and calculate their centroids in the same time.

## Use-case 
Execute kmeans_benchmark.py with parameters: dataset size, num of features, num of clusters. 
According to these params dataset will be generated, test made. 

Results of time measurments and speedup will be shown in console.
In addition to that, graph will be generated in file "speedup_graph_{ds_size}\_{ds_features}\_{ds_clust}.png".

To get pictures with results mentioned in next chapter, execute run script (it will take little less time than 5-7 minutes).

## Small remarks about results
In repo there is a couple of pictures.
As can be seen, there is NO performance improvement on small dataset (1000). 
Even with higher number of features speedup falls down. 
CPU is taking its time to schedule thread workers, and on small dataset it takes more time than if we just calculate stright in one thread.

But once dataset have not so small size (100000), there is a positive speedup.