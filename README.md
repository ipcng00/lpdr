# DRAL+ in Keras
An implementation of LPDR-based active learning algorithm referred to as Disagree Ratio for Active Learning Plus (DRAL+). The LPDR is a measure of the sample's sensitiveness to the perturbation of the decision boundary, and it is based on the disagree metric between the decision boundary and its perturbation.

## Abstract
Active learning strategy to query samples closest to the decision boundary can be an effective strategy for sampling the most uncertain and thus informative samples. This strategy is valid only when the sample's "closeness" to the decision boundary can be estimated. As a measure for evaluating closeness to a given decision boundary of a given sample, this paper considers the least probable disagreement region (LPDR) which is a measure of the smallest perturbation on the decision boundary leading to altered prediction of the sample. Experimental results show that the proposed LPDR-based active learning algorithm consistently outperforms other high performing active learning algorithms and leads to state-of-the-art performance on various datasets and deep networks.

### Prerequisites:
- Linux
- Python 3.7
- NVIDIA GPU + CUDA 10.0, CuDNN 7.6

### Installation
The required Python3 packages can be installed using
```
pip3 install -r requirements.txt
```

