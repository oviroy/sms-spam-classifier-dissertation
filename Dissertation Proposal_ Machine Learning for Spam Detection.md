# **DISSERTATION PROPOSAL**

# **Comparative Analysis of Classical Machine Learning Classifiers for Spam Detection**

**Student Name:** Shuvo Roy Ovi  
ID: 25004376

# **INTRODUCTION**

Spam and unsolicited bulk messages remain a persistent global challenge across email, SMS, and social media platforms. The economic and security costs of spam are substantial, making automated detection a critical application of machine learning. While deep learning and transformer-based models have achieved prominence in natural language processing, classical machine learning approaches—when combined with thoughtful feature engineering—remain highly effective, computationally efficient, and accessible to practitioners without specialized hardware.

This dissertation investigates the comparative performance of classical text classification pipelines for spam detection. Using the publicly available UCI SMS Spam Collection Dataset, the study systematically evaluates four machine learning classifiers across multiple text representation strategies and preprocessing configurations. The research is designed to run entirely on CPU within standard Python environments or Google Colab, requiring no GPU acceleration. The work aims to provide evidence-based recommendations for model and feature selection in resource-constrained settings, and to release fully reproducible notebooks suitable for educational use.

# **AIM**

To conduct a rigorous, reproducible empirical comparison of classical machine learning classifiers for short-text spam detection, evaluating the interaction between model choice, text representation, and preprocessing strategies on a publicly available dataset.

# **OBJECTIVES**

1. To implement and optimize four classical machine learning classifiers for spam detection, spanning three model families: Multinomial Naive Bayes (probabilistic), Logistic Regression and Linear Support Vector Machine (SVM) (linear), and XGBoost (tree-based ensemble).  
2. To evaluate three text representation strategies: bag-of-words (Count Vectorizer), TF-IDF with unigrams, and TF-IDF with n-grams (unigrams \+ bigrams, unigrams \+ bigrams \+ trigrams).  
3. To conduct a focused preprocessing ablation study on the two best-performing classifiers, testing a small set of key configurations (a full-preprocessing baseline versus targeted variants that toggle stopword removal, stemming, and punctuation removal) rather than exhaustively evaluating every permutation across all models.  
4. To design and evaluate a hybrid feature set combining lexical features (TF-IDF) with simple statistical features (message length, digit density, punctuation frequency, uppercase ratio, and URL count).  
5. To evaluate all model configurations using multiple metrics: accuracy, precision, recall, F1-score, AUC-ROC, and false positive rate.  
6. To perform statistical significance testing (McNemar's test, paired t-tests across cross-validation folds) to determine whether performance differences between top-performing models are meaningful.  
7. To analyze model interpretability for the best-performing linear model and the tree-based ensemble (XGBoost), using model coefficients and feature-importance scores, with SHAP values applied only to these two representative models to identify the linguistic and statistical patterns most indicative of spam.  
8. To measure and compare computational efficiency: training time, inference latency, and memory footprint across all models.  
9. To release fully reproducible Jupyter notebooks with fixed random seeds, version-pinned dependencies, and inline documentation.

# **OUTCOMES**

## **Expected Contributions**

* A comprehensive benchmark comparing four classical classifiers, spanning probabilistic, linear, and tree-ensemble families, across multiple feature representations and preprocessing strategies, controlling for experimental conditions often overlooked in prior spam detection literature.  
* Evidence-based recommendations on whether stopword removal, stemming, and lowercasing improve or harm spam detection performance for specific model types.  
* Quantification of the value added by simple statistical features (message length, digit density, punctuation) beyond standard TF-IDF lexical features.  
* Practical trade-off analysis between accuracy, training time, and model size, enabling informed decisions for deployment on low-resource systems.  
* Fully reproducible, well-documented notebooks suitable for teaching text classification, feature engineering, and model evaluation in introductory machine learning courses.  
* Evaluation of class imbalance handling techniques specifically for spam detection, emphasizing the critical importance of false positive rate in real-world filtering systems.

## **Deliverables**

* Reproducible notebooks covering data exploration, preprocessing, model training, evaluation, and visualization.  
* A 6,000-word dissertation report comprising: Introduction, Literature Review, Methodology, Results and Analysis, Discussion, and Conclusion.  
* A presentation summarizing methodology, key findings, and recommendations.

# **REFERENCES**

Almeida, T. A., Hidalgo, J. M. G., & Yamakami, A. (2011). Contributions to the study of SMS spam filtering: New collection and results. Proceedings of the 11th ACM Symposium on Document Engineering, 259–262.

Androutsopoulos, I., Koutsias, J., Chandrinos, K. V., & Spyropoulos, C. D. (2000). An experimental comparison of naive Bayesian and keyword-based anti-spam filtering with personal e-mail messages. Proceedings of the 23rd Annual International ACM SIGIR Conference on Research and Development in Information Retrieval, 160–167.

Chen, T., & Guestrin, C. (2016). XGBoost: A scalable tree boosting system. Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining, 785–794.

Drucker, H., Wu, D., & Vapnik, V. N. (1999). Support vector machines for spam categorization. IEEE Transactions on Neural Networks, 10(5), 1048–1054.

Forman, G. (2003). An extensive empirical study of feature selection metrics for text classification. Journal of Machine Learning Research, 3, 1289–1305.

Genkin, A., Lewis, D. D., & Madigan, D. (2007). Large-scale Bayesian logistic regression for text categorization. Technometrics, 49(3), 291–304.

Ke, G., Meng, Q., Finley, T., Wang, T., Chen, W., Ma, W., ... & Liu, T. Y. (2017). LightGBM: A highly efficient gradient boosting decision tree. Advances in Neural Information Processing Systems, 30\.

Lundberg, S. M., & Lee, S. I. (2017). A unified approach to interpreting model predictions. Advances in Neural Information Processing Systems, 30\.