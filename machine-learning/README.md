# Machine learning

## Feature engineering

## Supervised learning

### K-nearest neighbors
KNN is a non parametric classifier. In order to classify a point, it looking in a labeled training set for the points that have features most similar to the new data point, and will chose the most common label among these nearest neigbors.

Here is an example of classification with [KNN on the *Breast Cancer Wisconsin* and *Haberman's Survival* datasets](./ROB311_k-Nearest_Neighbors.ipynb).

One drawback of the method is that classifiying points is quite expensive. While there is absolutely no training time (this is a non-parametric classifier, so we don't have to seek the parameters), one needs to seek the nearest neighbors of a point: this may involve computing at most as many distances as there are points in the training set.

KD-trees and ball-trees are data structures for storing the training set, which speed-up the search for neighbors. KD-trees work well for low dimensonal spaces (few features) while ball-trees work best on high dimensional spaces.

### SVM
Support Vector Machines are linear classifiers that seek an hyperplan that will separate two sets of points in <img src="https://render.githubusercontent.com/render/math?math=\mathbb{R}^d"> with the highest possible margin. This is of course possible only for linearly separable sets. In the generic case, SVM uses a relaxed version of the problem where some data points are allowed to lie on the wrong side of the margin, provided the error &xi; is not too big. The problem is formulated as:  
<img src="https://render.githubusercontent.com/render/math?math=(\mathbf{w}^*,w_0^*,\xi^*)\in \underset{\mathbf{w}\in \mathcal{H},w_0\in \mathbb{R}, \xi \in \mathbb{R}^n}{\arg \min}\left(\frac{1}{2}\Vert\mathbf{w}\Vert^2 + C\sum_{i=1}^{n}{\xi_i}\right)">  
subject to constaints <img src="https://render.githubusercontent.com/render/math?math=\xi_i\ge 0"> and <img src="https://render.githubusercontent.com/render/math?math=y_i(\langle\mathbf{w},\Phi(\mathbf{x_i})\rangle+w_0)\ge 1-\xi_i">

The beauty of it is that vector **w** and bias w<sub>0</sub> that define the decision boundary can be expressed as a linear combination of data points that lie around the decision boundary, the **support vectors**, and they will usually represent only a small fraction of the data points. The decision function used to classify new points is a weighted sum of the scalar product of the new vector with these support vectors.

Now, the whole process can be made non-linear thanks to the *kernel trick*: replacing the scalar product with a kernel function such as a polynomial kernel or a radial basis function (the ubiquitous RBF kernel), the method still works.

Here is an example of using [SVM in the MNIST digit classification problem](./ROB311-SVM_Digit_Recognition.ipynb).

### Trees and ensemble methods

## Clustering

Here is an example of using **k-means** to [split images into clusters representing the same digit](./ROB311_Clustering_Digits.ipynb).
