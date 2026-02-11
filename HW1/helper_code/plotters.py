"""
general file for some helper methods to plot using 
matplotlib.
"""

import numpy as np
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt


# can plot 1+ classes with colored decision regions
def plot_2_params(
    X, 
    y, 
    classifiers, # list of classifiers
    titles, # list of titles
    x_axis_titles, # list of x axis titles
    y_axis_titles, # list of y axis titles
    resolution=0.02, 
    show=True, 
    num_plots=1
    ):

  # setup marker generator and color map
  markers = ('o', 's', '^', 'v', '<')
  colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
  
  for p in range(0, num_plots):  
    plt.subplot(nrows=1, ncols=num_plots, index=p+1) # choose which plot in fig
    cmap = ListedColormap(colors[:len(np.unique(y))])
    # plot the decision surface
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    # THIS IS NOT MALEABLE TO MORE THAN 2 PARAMS
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution), np.arange(x2_min, x2_max, resolution))
    lab = classifiers[p].predict(np.array([xx1.ravel(), xx2.ravel()]).T) # T -> transpose
    lab = lab.reshape(xx1.shape)

    plt.contourf(xx1, xx2, lab, alpha=0.3, cmap=cmap)
    plt.xlim(xmin=xx1.min(), xmax=xx1.max())
    plt.ylim(ymin=xx2.min(), ymax=xx2.max())
    # plot class examples
    for idx, cl in enumerate(np.unique(y)):
      plt.scatter(x=X[y == cl, 0],
        y=X[y == cl, 1],
        alpha=0.8,
        c=colors[idx],
        marker=markers[idx],
        label=f'Class {cl}',
        edgecolor='black')
    plt.legend()
    plt.title(titles[p])
    plt.xlabel(x_axis_titles[p])
    plt.ylabel(y_axis_titles[p])

  if show:
    plt.show()


# old plotter that plots only one
# def plot_2_params(
#     X, 
#     y, 
#     classifier, 
#     resolution=0.02, 
#     show=True, 
#     title="",
#     x_axis_title="",
#     y_axis_title=""
#     ):

#   # setup marker generator and color map
#   markers = ('o', 's', '^', 'v', '<')
#   colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
#   cmap = ListedColormap(colors[:len(np.unique(y))])
#   # plot the decision surface
#   x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
#   x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
#   # THIS IS NOT MALEABLE TO MORE THAN 2 PARAMS
#   xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution), np.arange(x2_min, x2_max, resolution))
#   lab = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T) # T -> transpose
#   lab = lab.reshape(xx1.shape)
#   plt.contourf(xx1, xx2, lab, alpha=0.3, cmap=cmap)
#   plt.xlim(xx1.min(), xx1.max())
#   plt.ylim(xx2.min(), xx2.max())
#   # plot class examples
#   for idx, cl in enumerate(np.unique(y)):
#     # print(idx, cl)
#     plt.scatter(x=X[y == cl, 0],
#       y=X[y == cl, 1],
#       alpha=0.8,
#       c=colors[idx],
#       marker=markers[idx],
#       label=f'Class {cl}',
#       edgecolor='black')
#   plt.legend()
#   plt.title(title)
#   plt.xlabel(x_axis_title)
#   plt.ylabel(y_axis_title)
#   if show:
#     plt.show()
 


def plot_loss_ada_v_log(ada, log, show=True):
  fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(10, 4))

  ax[0].plot(range(1, len(ada.losses_) + 1), ada.losses_, marker='o')
  ax[0].set_xlabel(f'{ada.n_iter} Epochs')
  ax[0].set_ylabel('Loss Function')
  ax[0].set_title(f"Adaline - Learning rate {ada.eta}")
  
  ax[1].plot(range(1, len(log.losses_) + 1), log.losses_, marker='o')
  ax[1].set_xlabel(f'{log.n_iter} Epochs')
  ax[1].set_ylabel('Loss Function')
  ax[1].set_title(f"Logistic Regression - Learning rate {log.eta}")
  if show:
    plt.show()

