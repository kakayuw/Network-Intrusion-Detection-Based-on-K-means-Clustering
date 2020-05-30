# Network-Intrusion-Detection-Based-on-K-means-Clustering
Applying K-means algorithm improved by Mahalanobis distance to the KDD 99 network intrusion data.

This project implements an K-Means classification algorithm to detect anormal network invasion, then experiments various parameter combination on various size of resampling data. 

The train/test data comes from [KDD 99 data](http://kdd.ics.uci.edu/databases/kddcup99/).

Inspired by [KMeans-Multidistance](https://github.com/dperede/KMeans-Multidistance)

## Project Structure
- dataset.cfg:
Configuration file of Dataset, marking file path and key attributes

- DataTidying.py:
Import data, reduce columns, encode polynomial categories and normalize numeric column.

- Kmenas.py:
Train unsupervised model utilizing euclidean/mahalanobis distance, and plotting loss.

- main.py:
Entry of program, evaluating model in different resampling size.

