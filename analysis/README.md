### Methods
The implemented methods so far: 

### Entropy
Let $\hat{P}(X) =\underset{c \in {\{0,1 \}}}{\max} P(Y=c | X)$ denote the confidence of the model and $\hat{y}_{X} = \underset{c \in {\{0,1 \}}}{\argmax} P(Y=c | X)$  denote the model's prediction. Entropy is simply $$H_{\hat{P}}(X) = \hat{P}(X)(1-\hat{P}(X)).$$

### kNN Confidence
Let the indices of $k$-nearest neighbors of $X$ be denoted by the set $\mathcal{D}^{(k)}_{X}$. kNN Confidence is defined as $$ \frac{1}{|\mathcal{D}^{(k)}_{X}|} \sum_{i=1}^{\mathcal{D}^{(k)}_{X}} \mathbf{1}[y_i = \hat{y}_x].$$ 

### Trust Score
This score is from "To Trust Or Not To Trust A Classifier" by Jiang et al. 2018. It is based on the kNN distances. Namely, Trust Score is the ratio of the distance to the closest class to the distance to the second closest class. Distance to a class can be defined as the distance to the closest point in that class, or the average to the 5-closest points, or similar extensions.

For a more concrete definition, see Algorithms 1/2 in the [paper](https://arxiv.org/pdf/1805.11783.pdf). 

### Normalized Margin
TBD


### Weighted-Entropy
TBD

### Simple Density Estimation (GMMs or KDE)
TBD

