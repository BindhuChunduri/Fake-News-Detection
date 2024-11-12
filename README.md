Fake News Detection Using Machine Learning

Project Overview

This project focuses on building a machine learning model to detect fake news, leveraging Natural Language Processing (NLP) and various classification algorithms. With the rise of misinformation on social media, automated fake news detection can help curb the spread of false information and provide more reliable content to users.

Problem Statement

The rapid spread of fake news on social media and online platforms threatens the integrity of public information. This project aims to develop an automated system that accurately identifies and classifies fake news to improve information quality and protect public trust.

Dataset

We used a dataset from Kaggle, specifically the COVID-19 Fake News Dataset. The dataset includes:
	•	Training Set: 6420 entries
	•	Validation Set: 2140 entries
	•	Test Set: 2140 entries
	•	Features:
	•	TF-IDF: 5000 features
	•	Statistical: 5 features
	•	Word2Vec: 100-dimensional embeddings

Class Distribution: Real (52.34%), Fake (47.66%)

Methodology

The project follows these main steps:
	1.	Data Preprocessing: Cleaning text by removing URLs, special characters, tokenization, stop word removal, and lemmatization.
	2.	Feature Extraction:
	•	TF-IDF for text representation (5000 features)
	•	Statistical features (5 features)
	•	Word2Vec embeddings (100 dimensions)
	3.	Model Training: Testing multiple classifiers, including:
	•	Logistic Regression
	•	Linear Support Vector Classifier (SVC)
	•	Passive Aggressive Classifier
	•	Decision Tree
	4.	Hyperparameter Tuning: Using GridSearchCV with 5-fold cross-validation for optimal parameter settings.
	5.	Model Evaluation: Using precision, recall, F1 score, and confusion matrix to evaluate performance.

Installation

To run this project locally, ensure you have the following dependencies installed:
	•	Python 3.x
	•	Scikit-learn
	•	NLTK
	•	Gensim
	•	Pandas
	•	NumPy
	•	Matplotlib (for visualization)

Install dependencies with:

pip install -r requirements.txt

Usage

	1.	Clone the Repository:

git clone https://github.com/yourusername/fake-news-detection.git
cd fake-news-detection


	2.	Preprocess Data: Run the data preprocessing script to clean and tokenize text.
	3.	Train Models: Execute the training script to train and validate multiple classifiers.
	4.	Evaluate: Check model performance on test data and review the precision-recall curves, F1 scores, and confusion matrix.

Results and Evaluation

	•	Best Model: Linear SVC achieved the highest accuracy at approximately 93%.
	•	F1 Scores: Training (0.988), Validation (0.987), and Test (0.930), indicating consistent performance.
	•	Common Errors: Misclassification was more frequent in health-related news, where specific wording patterns might be similar in both fake and real news.

Project Structure

fake-news-detection/
│
├── data/                  # Dataset files
├── notebooks/             # Jupyter notebooks for exploratory analysis
├── src/                   # Source code for preprocessing, model training, and evaluation
│   ├── preprocess.py      # Data preprocessing pipeline
│   ├── feature_extraction.py # Feature extraction with TF-IDF, statistical, and Word2Vec
│   ├── train.py           # Model training and cross-validation
│   ├── evaluate.py        # Model evaluation and performance metrics
│
├── requirements.txt       # List of dependencies
├── README.md              # Project description and setup instructions
└── results/               # Evaluation results and visualizations

Contributors

	•	Himabindu Chunduri - Version control and repository management
	•	Vijayalakshmi Pepala - CI/CD pipeline and automated testing
	•	Jaswanth Nalluri - Sprint planning and feature management
	•	Sandeep Chowdary Ari - Code reviews and documentation

References

	•	Shu, K., et al. (2017). Fake News Detection on Social Media: A Data Mining Perspective.
	•	Devlin, J., et al. (2019). BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding.
	•	Scikit-learn Library - Pedregosa et al., JMLR 12, 2011.
