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
import matplotlib.pyplot as plt

# Look pretty...
matplotlib.style.use('ggplot')


#
# TODO: Start by creating a regular old, plain, "vanilla"
# python list. You can call it 'samples'.
#
# .. your code here .. 
samples = []
colors = []
#
# TODO: Write a for-loop that iterates over the images in the
# Module4/Datasets/ALOI/32/ folder, appending each of them to
# your list. Each .PNG image should first be loaded into a
# temporary NDArray, just as shown in the Feature
# Representation reading.
#
for filename in os.listdir('../Module4/Datasets/ALOI/32/'):
    # print(filename)
    filename = "../Module4/Datasets/ALOI/32/" + filename
    img = misc.imread(filename)
    img = img.reshape(-1)
    samples.extend([img])
    colors.extend(['b'])
for filename in os.listdir('../Module4/Datasets/ALOI/32i/'):
    #print(filename)
    filename = "../Module4/Datasets/ALOI/32i/" + filename
    img = misc.imread(filename)
    img = img.reshape(-1)
    samples.extend([img])
    colors.extend(['r'])

#print(samples)
# Optional: Resample the image down by a factor of two if you
# have a slower computer. You can also convert the image from
# 0-255  to  0.0-1.0  if you'd like, but that will have no
# effect on the algorithm's results.
#
# .. your code here .. 


#
# TODO: Once you're done answering the first three questions,
# right before you converted your list to a dataframe, add in
# additional code which also appends to your list the images
# in the Module4/Datasets/ALOI/32_i directory. Re-run your
# assignment and answer the final question below.
#
# .. your code here .. 
df = pd.DataFrame.from_records(samples)#

# TODO: Convert the list to a dataframe
#
# .. your code here .. 



#
# TODO: Implement Isomap here. Reduce the dataframe df down
# to three components, using K=6 for your neighborhood size
#
# .. your code here .. 
from sklearn import manifold
iso = manifold.Isomap(n_neighbors=6 , n_components=3)
#iso = manifold.Isomap(n_components=3)
iso.fit(df)
T = iso.transform(df)
print(T.shape)
print(T)



#
# TODO: Create a 2D Scatter plot to graph your manifold. You
# can use either 'o' or '.' as your marker. Graph the first two
# isomap components
#
# .. your code here .. 
def Plot2D(T, title, x, y, num_to_plot=100):
  # This method picks a bunch of random samples (images in your case)
  # to plot onto the chart:
  fig = plt.figure()
  ax = fig.add_subplot(111)
  ax.set_title(title)
  ax.set_xlabel('Component: {0}'.format(x))
  ax.set_ylabel('Component: {0}'.format(y))
  x_size = (max(T[:,x]) - min(T[:,x])) * 0.1
  y_size = (max(T[:,y]) - min(T[:,y])) * 0.1
  for i in range(num_to_plot):
    img_num = int(random.random() * num_images)
    x0, y0 = T[img_num,x]-x_size/2., T[img_num,y]-y_size/2.
    x1, y1 = T[img_num,x]+x_size/2., T[img_num,y]+y_size/2.
    img = df.iloc[img_num,:].reshape(144, 192)
    ax.imshow(img, aspect='auto', cmap=plt.cm.gray, interpolation='nearest', zorder=100000, extent=(x0, x1, y0, y1))

  # It also plots the full scatter:
  ax.scatter(T[:,x],T[:,y], marker='.',alpha=0.7, color=colors)

num_images, num_pixels = df.shape
num_pixels = int(math.sqrt(num_pixels))
#
#for i in range(num_images):
#  df.loc[i,:] = df.loc[i,:].reshape(192, 144).T.reshape(-1)

Plot2D(T, "Isomap", 0, 2, num_to_plot=0)

#
# TODO: Create a 3D Scatter plot to graph your manifold. You
# can use either 'o' or '.' as your marker:
#
# .. your code here .. 
def Plot3D(T, title, x, y, z, num_to_plot=40):
  # This method picks a bunch of random samples (images in your case)
  # to plot onto the chart:
  fig = plt.figure()
  ax = fig.add_subplot(111, projection='3d')
  ax.set_title(title)
  ax.set_xlabel('Component: {0}'.format(x))
  ax.set_ylabel('Component: {0}'.format(y))
  ax.set_zlabel('Component: {0}'.format(z))
  x_size = (max(T[:,x]) - min(T[:,x])) * 0.1
  y_size = (max(T[:,y]) - min(T[:,y])) * 0.1
  z_size = (max(T[:,z]) - min(T[:,z])) * 0.1
  for i in range(num_to_plot):
    img_num = int(random.random() * num_images)
    x0, y0, z0 = T[img_num,x]-x_size/2., T[img_num,y]-y_size/2., T[img_num,z]-z_size/2.
    x1, y1, z1 = T[img_num,x]+x_size/2., T[img_num,y]+y_size/2., T[img_num,z]-z_size/2
    img = df.iloc[img_num,:].reshape(144, 192)
    ax.imshow(img, aspect='auto', cmap=plt.cm.gray, interpolation='nearest', zorder=100000, extent=(x0, x1, y0, y1))

  # It also plots the full scatter:
  ax.scatter(T[:,x],T[:,y],T[:,z], marker='.',alpha=0.7, color=colors)

Plot3D(T, "Isomap", 0, 1, 2, num_to_plot=40)
  
plt.show()

