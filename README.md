# Toxic Comment Classification with NLP

## Business Case
Toxic comment classification can be useful for businesses in a number of ways. For example, if a business operates a social media platform or an online forum, it may be important to identify and remove toxic comments in order to create a positive and welcoming environment for users. This can help to reduce the risk of users becoming distressed or offended by toxic content, which can lead to negative brand associations and a decrease in user engagement and retention. Additionally, identifying and removing toxic comments can help to prevent the spread of misinformation or hateful ideologies, which can have serious consequences for both the business and society as a whole. In general, toxic comment classification can help businesses to foster a more positive and safe online community, which can ultimately lead to improved brand reputation and customer loyalty.

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





### Conclusions

### Limitations
Due to the high imbalance the model might not had enough toxic data to train on. 

## Repository Structure

├── data
├── images
├── .gitignore
├── README.md
├── model.py
├── model2.pkl
├── presentation.pdf
└── Toxic_Comment_Classification-toxic-stem.ipynb
