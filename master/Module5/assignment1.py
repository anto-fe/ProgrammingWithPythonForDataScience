#
# TOOD: Import whatever needs to be imported to make this work
#
# .. your code here ..
import pandas as pd

from scipy import misc
from mpl_toolkits.mplot3d import Axes3D
import matplotlib
import matplotlib.pyplot as plt
import os

import numpy as np
import scipy.io
import random, math
from mpl_toolkits.mplot3d import Axes3D


matplotlib.style.use('ggplot') # Look Pretty


#
# TODO: To procure the dataset, follow these steps:
# 1. Navigate to: https://data.cityofchicago.org/Public-Safety/Crimes-2001-to-present/ijzp-q8t2
# 2. In the 'Primary Type' column, click on the 'Menu' button next to the info button,
#    and select 'Filter This Column'. It might take a second for the filter option to
#    show up, since it has to load the entire list first.
# 3. Scroll down to 'GAMBLING'
# 4. Click the light blue 'Export' button next to the 'Filter' button, and select 'Download As CSV'


def doKMeans(df):
  #
  # INFO: Plot your data with a '.' marker, with 0.3 alpha at the Longitude,
  # and Latitude locations in your dataset. Longitude = x, Latitude = y
  fig = plt.figure()
  ax = fig.add_subplot(111)
  ax.scatter(df.Longitude, df.Latitude, marker='.', alpha=0.3)

  #
  # TODO: Filter df so that you're only looking at Longitude and Latitude,
  # since the remaining columns aren't really applicable for this purpose.
  #
  # .. your code here ..
  df = df[['Latitude', 'Longitude' ]]
  print(df)
  #
  # TODO: Use K-Means to try and find seven cluster centers in this df.
  #
  # .. your code here ..
  from sklearn.cluster import KMeans
  kmeans_model = KMeans(n_clusters=7, random_state=123)
  kmeans_model = kmeans_model.fit(df)

  #
  # INFO: Print and plot the centroids...
  centroids = kmeans_model.cluster_centers_
  ax.scatter(centroids[:,1], centroids[:,0], marker='x', c='red', alpha=0.5, linewidths=3, s=169)
  print(centroids)



#
# TODO: Load your dataset after importing Pandas
#
# .. your code here ..
df = pd.read_csv("../Module5/crimes.csv")

#
# TODO: Drop any ROWs with nans in them
#
# .. your code here ..
df = df.dropna(axis=0)

#
# TODO: Print out the dtypes of your dset
#
# .. your code here ..


#
# Coerce the 'Date' feature (which is currently a string object) into real date,
# and confirm by re-printing the dtypes. NOTE: This is a slow process...
#
# .. your code here ..
df.Date = pd.to_datetime(df.Date, errors='coerce')
df = df.dropna(axis=0)

# INFO: Print & Plot your data
doKMeans(df)


#
# TODO: Filter out the data so that it only contains samples that have
# a Date > '2011-01-01', using indexing. Then, in a new figure, plot the
# crime incidents, as well as a new K-Means run's centroids.
#
# .. your code here ..
df = df[df.Date > '2011-01-01']



# INFO: Print & Plot your data
doKMeans(df)
plt.show()


