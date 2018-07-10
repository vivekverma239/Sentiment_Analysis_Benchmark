* [Deep pyramid convolutional neural networks for text categorization](http://aclweb.org/anthology/P/P17/P17-1052.pdf).  Rie Johnson and Tong Zhang.  ACL 2017.
* [Effective use of word order for text categorization with convolutional neural networks](https://aclweb.org/anthology/N/N15/N15-1011.pdf).   Rie Johnson and Tong Zhang.  NAACL HLT 2015.    
* [Semi-supervised convolutional neural networks for text categorization via region embedding](https://papers.nips.cc/paper/5849-semi-supervised-convolutional-neural-networks-for-text-categorization-via-region-embedding).  Rie Johnson and Tong Zhang.  NIPS 2015.  
* [Supervised and semi-supervised text categorization using LSTM for region embeddings](http://proceedings.mlr.press/v48/johnson16.pdf).  Rie Johnson and Tong Zhang.  ICML 2016.   

* [Convolutional Neural Networks for Sentence Classification](https://arxiv.org/pdf/1408.5882.pdf) Yoon Kim 2014

* [A Sensitivity Analysis of (and Practitioners Guide to) Convolutional Neural Networks for Sentence Classification](https://arxiv.org/pdf/1510.03820.pdf)

* [Recurrent Convolutional Neural Networks for Text Classification]()

* [Character-level Convolutional Networks for Text Classification](http://papers.nips.cc/paper/5782-character-level-convolutional-networks-for-text-classification.pdf)

* [S. I. Wang and C. D. Manning. Baselines and bigrams: Simple, good sentiment and topic
classification. In ACL, 2012.](https://www.aclweb.org/anthology/P12-2018)

* [Semi-supervised Sequence Learning](https://arxiv.org/pdf/1511.01432.pdf)

* [Convolutional Neural Networks for Sentence Classification](https://arxiv.org/pdf/1408.5882.pdf)

Datasets:

• MR: Movie reviews with one sentence per review. Classification involves detecting positive/negative reviews (Pang and Lee, 2005).3
• SST-1: Stanford Sentiment Treebank—an extension of MR but with train/dev/test splits provided and fine-grained labels (very positive,positive, neutral, negative, very negative),re-labeled by Socher et al. (2013)
• SST-2: Same as SST-1 but with neutral reviews removed and binary labels.
• Subj: Subjectivity dataset where the task is to classify a sentence as being subjective or objective (Pang and Lee, 2004).
• TREC: TREC question dataset—task involves classifying a question into 6 question types (whether the question is about person,location, numeric information, etc.) (Li and Roth, 2002).
• CR: Customer reviews of various products (cameras, MP3s etc.). Task is to predict positive/negative reviews (Hu and Liu, 2004)
•MPQA: Opinion polarity detection subtask of the MPQA dataset (Wiebe et al., 2005).
•Irony (Wallace et al., 2014): this contains 16,006 sentences from reddit labeled as ironic (or not). The dataset is imbalanced (relatively few sentences are ironic). Thus before training, we under-sampled negative instances to make classes sizes equal.3 For this dataset we report the Area Under Curve (AUC), rather than accuracy, because it is imbalanced
