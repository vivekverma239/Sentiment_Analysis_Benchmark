# IMDB Sentiment Analysis Benchmarking
The Aim of this repository is to establish a benchmark for IMDB sentiment analysis dataset. In future it can be extended to other datasets/

The following models are implemented:
  - TF-IDF with Naive Bayes Classifier
  - TF-IDF with Linear SVM
  - TF-IDF with Non-Linear SVM
  - TF-IDF with NBSVM

Future Tasks:
  - Deep Learning Models


## Software Requirements
All the required packages can be installed by running the following commands

`pip install -r reqirements.txt`


## Experiments
First, using TF-IDF to extract features from the sentence and then classifying using Naive Bayse, Linear SVM, Non-Linear SVM (using rbf kernel and sigmoid). GridSearch was used to tune the hyperparameters.



## Results:
Here are the results:

| Tables                             | Accuracy |
| -------------                      |:---------:|
| TF-IDF with Naive Bayes Classifier |   87.96%    |
| TF-IDF with Linear SVM             |   90.94%    |
| TF-IDF with Non-Linear SVM         |   89.20%    |
| TF-IDF with NBSVM                  |   92.01%   |         


### Running the code
To run all the TF-IDF based classification models

+ `python tf_idf_classifier.py`

This generates a classification metrics of test data.

To run all the TF-IDF based classification models

## References
