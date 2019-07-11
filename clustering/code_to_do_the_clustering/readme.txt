To implement the clustering:
1. Extract the features from all the 100K unlabeled images using "extracting_features_005"
2. Search for the clusters with the aglomerative clustering in the "exploring_agglomClustering_001" file.
3. Cluster the childresn previously found with "clustering_children_006_less_collisions" or "clustering_children_005_not_iterative_retrieval"
4. Merge the 3 folders generated (clusters from single images, large clusters, and small clusters)
5. Compute the edges
6. Then use a variation of "75_..." to generate the transformations  of the images.
