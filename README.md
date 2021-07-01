# NLP Classification of Subreddit Posts

In this project, I scraped text data from two subreddits then built a classification model that predicted which subreddit a post originated from with over 95% accuracy.

## Introduction & Problem Statement

In this project, we will be exploring the NLP task of making predictions based on the text contents of a corpus. A major source of unstructured natural language data are social media websites such as Reddit, where users create posts in communities known as subreddits. Specifically, we will be working with the contents of posts from two different subreddits - `r/AskScience` and `r/AskSocialScience`.

**Problem Statement:
How can we best develop a classification model using NLP to classify posts belonging to two different subreddits?**

Both subreddits serve as forums for answering questions about their respective fields - Science & Social Science. The majority of posts come in the form of questions about something related to the field, which would then be answered by other users in the comments. In this project, we will only be focusing on the text contents of the original post itself, including the title and the self-text of the post.

At first glance, both subreddits seem to be quite similar in that their functions are identical and the fields are not too different from each other. Through various NLP techniques, we will attempt to train a model that will be able to identify the differences between posts from each subreddit and classify new posts accordingly. While the actual model may not have any practical use in real life (since Reddit posts are by default already assigned to whichever subreddit they are posted in), the insights gained from our analysis and processing may be useful to the moderators of the respective subreddits.

## Executive Summary

2042 rows of data from `r/AskScience` and `r/AskSocialScience` were scraped using the Reddit API and combined into a single dataframe. After several preprocessing steps including data cleaning and stemming, the text was then vectorized to form a Bag-of-Words model.

A pipeline was formed to test various models' performance in conjunction with different vectorizers. 3 different vectorizers were used together with 10 different classification models and were evaluated on multiple metrics.

**Vectorizers & Models**

| Vectorizer         | Classifier              |
|--------------------|-------------------------|
| Count Vectorizer   | Logistic Regression     |
| Tfidf Vectorizer   | K Nearest Neighbors     |
| Hashing Vectorizer | Multinomial Naïve Bayes |
|                    | Decision Tree           |
|                    | Bagging                 |
|                    | Random Forest           |
|                    | Extra Trees             |
|                    | Ada Boost               |
|                    | Gradient Boost          |
|                    | Support Vector Machine  |

After gridsearching over each combination, the MultinomialNB model (with Tfidf Vectorizer) was found to be the best performing model, achieving a test accuracy score of 95.8%. Further evaluation of the feature importance and misclassification analysis was then carried out to understand more about the dataset.

## Data Dictionary

| File                 | Column         | Type   | Description                                                         |
|----------------------|----------------|--------|---------------------------------------------------------------------|
| `combined_clean.csv` | clean_text     | string | Self-text of the reddit post                                        |
| `combined_clean.csv` | subreddit      | string | Subreddit the post originates from                                  |
| `model_results.csv`  | Vectorizer     | string | Vectorizer used to form bag-of-words                                |
| `model_results.csv`  | Classifier     | string | Classifier used in the model                                        |
| `model_results.csv`  | Train Accuracy | float  | Accuracy of the model against the training data                     |
| `model_results.csv`  | Test Accuracy  | float  | Accuracy of the model against the test data                         |
| `model_results.csv`  | F1             | float  | F1 score of the model (`r/AskScience` as the positive class)        |
| `model_results.csv`  | Precision      | float  | Precision score of the model (`r/AskScience` as the positive class) |
| `model_results.csv`  | Recall         | float  | Recall score of the model (`r/AskScience` as the positive class)    |
| `model_results.csv`  | ROC-AUC        | float  | ROC-AUC of the model (`r/AskScience` as the positive class)         |
