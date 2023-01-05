# Toxic Comment Classification with NLP

## Business Understanding
Toxic comment classification can be useful for businesses in a number of ways. For example, if a business operates a social media platform or an online forum, it may be important to identify and remove toxic comments in order to create a positive and welcoming environment for users. This can help to reduce the risk of users becoming distressed or offended by toxic content, which can lead to negative brand associations and a decrease in user engagement and retention. Additionally, identifying and removing toxic comments can help to prevent the spread of misinformation or hateful ideologies, which can have serious consequences for both the business and society as a whole. In general, toxic comment classification can help businesses to foster a more positive and safe online community, which can ultimately lead to improved brand reputation and customer loyalty.

## Business Problem
Company XYZ is operating a website with an online discussion forum. Due to that they are relatively new to the market and thus far don’t have much experience in content moderation. Their resources for content moderation are also limited, and with a high volume of comments  being posted in multiple forums at the same time. As a result they have trouble flagging all the toxic comments. Therefore there is a risk of users turning away from their site due to harassment and a toxic environment.
They would like to build a machine learning model which can handle large amounts of data and flag these types of comments in a timely manner, to prevent the spreading of these issues and take the correct disciplinary action against the violators.

## Data

Source:
https://www.kaggle.com/datasets/julian3833/jigsaw-toxic-comment-classification-challenge

The dataset contains wikipedia user comments in various areas. I only used the train csv, that had been provided in this competition.
The dataset contains 159571 rows and 8 columns, including: id, comment_text, toxic, sever_toxic, obscene, threat, insult, identity_hate. I will be using the comment_text and toxic columns to create a binary classification.
There is a high imbalance 90.5% - 9.5% in the classes, this will be addressed in the modeling section by applying SMOTE and RandomOverSampler techniques.

## Preprocessing
There were no missing values, however the comments contained http links, foreign characters which had to be taken out.
After removing those the dataset was tokenized, stopwords and punctuations were removed.

Top 20 tokens in the whole corpus
<p align='center'>
  <img src="data/top20_tokens_all.png">
</p>

Top 20 toxic tokens
<p align='center'>
  <img src="data/top20_toxic.png">
</p>
In the above wordcloud of most frequent toxic comments clearly shows that some users use abusive language to harass others, creating a toxic and unwelcoming environment envirement, which woulds lead to more user either stop expressing themself on the online platform or leave it altogether.


## Modelling
I used Machine learning models, namely: Multinomial Naive Bayes, Random Forest and XGBoost in my classification project.
These models all work well with large dataset. MultinomialNB is simple and efficient technique for classifying text. Part of the reason I choose this because it's easy to implement and can be trained on relatively large dataset quickly. It is relatively insensitive to the scaling of the input features, which is convenient due to the high dimensionality after the vectorization (140k+ cols).

Random Forest can also handle high-dimensional data well. It is robust to noise in the data and resistant to overfitting. It's easy to implement and tune. It can also perform well without fine-tuning. And lastly it fast to train and predict, which makes it an optimal choise for large datasets.

### Best Model

Top 20 toxic tokens
<p align='center'>
  <img src="data/ensemble_auc.png">
</p>
This last ensemble model performed the best. This model combines the prediction of the MultinomialNB and the XGBoost model with best params. This Stacking model uses the tfidf vectorizer and SMOTE. The resulting model has a high recall score of 84% and an AUC of 0.97 on the test set, with relatively low numbers of false negatives (492) and false positives (1111).
<p align='center'>
  <img src="data/matrix.png">
</p>
Part of the reason I choose this model is becaue this model had the lowest FN - FP pair. It is important to minimize the number of false negatives, because users wouldn't be flagged for writting toxic comments, when actually they are.

## Conclusions
This model is likely performed the well because the combination of MultinomialNB and XGBoost are able to capture a wide range of patters in the data, allowing it to make more accurate predictions than the previous models. In addition the use of tfidf vectorizer may be able to capture important information about the frequency and importance of different words in the data. SMOTE likely helped to improve the model's performance by better representing the features of the data and addressing class imbalances respectively.

### Limitations
Due to the high imbalance the model might not had enough toxic data to train on. 

## Repository Structure
```
├── data
├── images
├── .gitignore
├── README.md
├── Toxic_Comment_Classification-toxic-stem.ipynb
├── Toxic_Comment_Classification-toxic-lemmatization.ipynb
└── presentation.pdf
```
