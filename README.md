# 🏥 Sentiment Analysis on Healthcare Dataset using NLP  

This project uses *Natural Language Processing (NLP)* techniques to analyze patient feedback from a healthcare dataset.  
The model classifies sentiments into *positive, **neutral, and **negative* categories to help healthcare providers gain insights from patient opinions.


## 📌 Overview
The project demonstrates text preprocessing, feature extraction, and model building for sentiment classification.  
Both *classical ML algorithms* and an *optional deep learning model (LSTM)* are applied to improve accuracy.


## 🗂️ Dataset
 Source: healthcare_reviews.csv (contains patient reviews and sentiment labels)  
 Columns:  
  - Review → Text of patient feedback  
  - Sentiment → Target label (Positive, Neutral, Negative)  

🔹 A sample dataset is included for demonstration.  
🔹 Full dataset is not shared due to privacy concerns.


## 🛠️ Technologies Used
Python Libraries: Pandas, NumPy, Matplotlib, Seaborn  
NLP Tools: NLTK, Scikit-learn, TF-IDF / Word2Vec  
Models: Logistic Regression, Naive Bayes, Random Forest, LSTM


## 🔍 Methodology
1. *Data Preprocessing*
   - Lowercasing, tokenization, stopword removal, lemmatization  
   - Feature extraction using TF-IDF vectorization  

2. *Model Training*
   - ML Models (Logistic Regression, Naive Bayes, Random Forest)  
   - Deep Learning Model (LSTM with word embeddings)  

3. *Evaluation*
   - Metrics: Accuracy, Precision, Recall, F1-score
  

## ✅ Results
| Model                 | Accuracy |
|-----------------------|----------|
| Logistic Regression   | 85%      |
| Naive Bayes           | 83%      |
| Random Forest         | 87%      |
| LSTM                  | 90%+     |
