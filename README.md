# Fake News Detection Using Machine Learning

## Project Overview
This project focuses on building a machine learning system to detect **fake news** using Natural Language Processing (NLP) and classical machine learning algorithms. With the rapid spread of misinformation on social media—especially during events like the COVID-19 pandemic—automated fake news detection systems are essential to protect public trust and information integrity.

The system leverages **text preprocessing, multi-feature extraction, model comparison, and hyperparameter optimization** to accurately classify news content as **Fake** or **Real**.

## Problem Statement
The rapid spread of fake news on social media platforms poses significant risks to public information integrity and decision-making.  
This project aims to develop an **automated machine learning-based system** that accurately identifies fake news content with high precision and recall.

## Dataset
The project uses the **COVID-19 Fake News Dataset** (Kaggle).

### Dataset Split
- **Training Set:** 6420 samples  
- **Validation Set:** 2140 samples  
- **Test Set:** 2140 samples  

### Dataset Fields
- `id` – unique identifier  
- `tweet` – text content  
- `label` – `fake` or `real`  

### Class Distribution
- **Real:** 52.34%  
- **Fake:** 47.66%  

## Methodology

### 1. Data Preprocessing
- Lowercasing text
- URL removal
- Removing user mentions and hashtags
- Removing punctuation and numbers
- Tokenization
- Stopword removal
- Lemmatization (NLTK WordNet)

### 2. Feature Engineering
A multi-feature approach is used:

| Feature Type |     Description    |    Dimensions  |
|--------------|--------------------|----------------|
| TF-IDF       | Text vectorization |      5000      |
| Statistical  | Word/character-level metrics |  5   |
| Word2Vec     | Semantic embeddings|       100      |
| **Total Features** | Combined | **5105** |

### 3. Models Evaluated
- Logistic Regression  
- Linear Support Vector Classifier (SVC)  
- Passive Aggressive Classifier  
- Decision Tree Classifier  

### 4. Hyperparameter Tuning
- GridSearchCV with **5-fold cross-validation**
- Model selection based on validation accuracy and CV score

### 5. Evaluation Metrics
- Accuracy
- Precision
- Recall
- F1-Score
- Confusion Matrix
- Precision-Recall Curves
- Error Analysis (False Positives & False Negatives)

## Results and Evaluation

### Best Performing Model
**Linear SVC**

### Test Performance
- **Accuracy:** 92.7%  
- **Precision:** ~0.93  
- **Recall:** ~0.93  
- **F1-Score:** ~0.93  

### Observations
- Linear SVC and Logistic Regression performed best
- Decision Tree showed overfitting
- Passive Aggressive performed weakest
- Misclassifications were more common in **health-related and statistical news**, where wording overlaps between fake and real content

## Contributors
- **Himabindu Chunduri** – Project lead and core ML contributor; designed the end-to-end machine learning pipeline including data preprocessing, feature engineering (TF-IDF, statistical features, Word2Vec), model training, hyperparameter tuning, and evaluation. Managed Git version control, repository structure, and final integration of results.

- **Jaswanth Nalluri** – Technical coordination and sprint lead; contributed to model experimentation and comparison, assisted with feature engineering workflows, managed sprint planning and task breakdown, and ensured timely integration of model improvements and evaluation results.

- **Vijayalakshmi Pepala** – CI/CD pipeline setup lead and automated testing; responsible for configuring continuous integration workflows, validating model runs, and maintaining build consistency.

- **Sandeep Chowdary Ari** – Code reviews and documentation lead; conducted systematic code reviews, improved documentation clarity, and ensured adherence to coding and reporting standards.

## References
	•	Shu, K., et al. (2017). Fake News Detection on Social Media: A Data Mining Perspective.
	•	Devlin, J., et al. (2019). BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding.
	•	Scikit-learn Library - Pedregosa et al., JMLR 12, 2011.
	
## Project Structure
```text
fake-news-detection/
│
├── data/                  # Dataset files (or samples / links)
├── notebooks/             # Jupyter notebooks for EDA and experiments
├── src/                   # Source code
│   ├── preprocess.py
│   ├── feature_extraction.py
│   ├── train.py
│   └── evaluate.py
│
├── results/               # Evaluation results and visualizations
├── requirements.txt       # Project dependencies
└── README.md              # Project documentation
