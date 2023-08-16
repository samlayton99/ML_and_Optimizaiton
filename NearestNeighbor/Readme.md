# K Nearest Neighbor for Handwriting Recognition

K-dimensional trees (KDTrees) are a pivotal data structure in computer science, aiding in the efficient solving of the nearest neighbor search problem. This code provides an implementation of KDTrees and a K-nearest neighbors classifier. It also applies this classifier to train a KDTRee to identify handwritten digits with 93% accuracy.

## Mathematical Background and Overview of Functions

The basis for KDTrees and the nearest neighbors classifier lies in several foundational mathematical concepts. An understanding of these principles ensures a more profound comprehension of the algorithms and their significance.

1. **Euclidean Distance**
    - Defined as the true straight line distance between two points in Euclidean space. For two points, `p` and `q` in `n`-dimensional space, it's mathematically expressed as:
    \[d(p, q) = \sqrt{\sum_{i=1}^{n} (q_i - p_i)^2}\]

2. **K-Dimensional Tree (KDTree)**
    - A unique space-partitioning data structure, KDTrees play a pivotal role in dividing space into distinct regions. Their main utility comes in the form of organizing points in a k-dimensional space. Such structuring is essential for numerous applications, particularly those that deal with multi-dimensional search keys.

3. **Nearest Neighbor**
    - Within a specific set of points, the nearest neighbor of any point is the one closest to it. This closeness is determined based on a specific distance metric, with the Euclidean distance being the most common.

### Functions and Classes Overview

1. **Exhaustive Search**
    - `exhaustive_search(X, z)`: A thorough search function that identifies the nearest neighbor in the dataset `X` for the point `z`. The function then returns the closest point and its respective Euclidean distance from `z`.

2. **KDTree Node Class**
    - `KDTNode`: Represents the core node in the KDTree, equipped with left and right child nodes. Each node embodies a k-dimensional value along with a pivot that dictates the dimension used during comparisons.

3. **KDTree Class**
    - `KDT`: Symbolizes the KDTree itself, vital for addressing the nearest neighbor issue. It's furnished with methods that facilitate data insertion, data finding, and tree querying to identify the nearest neighbor.

4. **K-Nearest Neighbors Classifier Class**
    - `KNeighborsClassifier`: A manifestation of the renowned K-nearest neighbors classification algorithm. It utilizes SciPy's KDTree to ascertain the nearest neighbors of a point swiftly. This class is replete with methods designed for fitting the classifier on training datasets and predicting labels for novel data.

5. **Dataset Classification Function**
    - `prob6(n_neighbors, filename="mnist_subset.npz")`: This function is tasked with data loading from a file, training the KNeighborsClassifier on the available training data, and subsequently calculating the accuracy on the test data.

## Project Flow

1. **Initialization**: Kick off the process by defining the fundamental KDTree Node through the `KDTNode` class.
2. **Constructing the KDTree**: The `KDT` class is instrumental in crafting the KDTree, enabling functions like insertion, data search, and nearest neighbor queries.
3. **K-Nearest Neighbors Algorithm**: With the KDTree at your disposal, you can then employ the `KNeighborsClassifier` class to enact the K-nearest neighbors classification.
4. **Dataset Classification**: Using the `prob6` function, you can showcase the classifier's prowess on the MNIST dataset subset and discern its accuracy.

## Emphasized Applications

- **Image Recognition**: Leveraging the nearest neighbor algorithms on datasets like MNIST facilitates the classification and identification of images.
- **Recommendation Systems**: The nearest neighbor search proves invaluable in identifying similar users or items, paving the way for nuanced recommendations rooted in inherent similarities.
- **Geospatial Data**: Pinpoint locations in proximity to a designated point, such as identifying the closest eateries on a map.
- **Anomaly Detection**: Observing outliers in datasets becomes a breeze by spotting points lacking close neighbors.

## Dependencies

```python 
import numpy as np
from scipy.spatial import KDTree
import scipy.stats
from matplotlib import pyplot as plt
```


## Additional Information

A supplementary function, `digitcheck` offers a method to visualize the digits within the MNIST dataset. It also predicts their labels using the classifier.
```