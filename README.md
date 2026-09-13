# NLP Sentiment Analysis using BoW, TF-IDF & Word2Vec

An NLP-based sentiment classification project using Amazon Kindle book reviews. The project compares three different text representation techniques — **Bag of Words (BoW), TF-IDF, and Word2Vec** — and applies machine learning algorithms to classify reviews into positive and negative sentiment classes.

## Project Overview

The goal of this project is to build a complete Natural Language Processing pipeline for sentiment classification.

The project covers:

* Data loading and exploration
* Text cleaning and preprocessing
* Sentiment label creation
* Train-test splitting
* Bag of Words feature extraction
* TF-IDF feature extraction
* Word2Vec word embeddings
* Machine learning model training
* Model evaluation and comparison

## Dataset

The project uses an Amazon Kindle review dataset containing **12,000 reviews**.

The main columns used are:

* `reviewText` — Text of the customer review
* `rating` — Rating given by the customer

The original ratings range from **1 to 5**.

### Sentiment Labeling

The ratings were converted into binary sentiment labels:

| Original Rating | Sentiment Label |
| --------------- | --------------- |
| 1, 2            | 0 — Negative    |
| 3, 4, 5         | 1 — Positive    |

After conversion:

* **8,000 Positive reviews**
* **4,000 Negative reviews**

The dataset contains no missing values in the selected columns.

## Technologies Used

* Python
* Pandas
* NumPy
* NLTK
* Scikit-learn
* Gensim
* Word2Vec
* Matplotlib / data analysis tools used in the notebook

## Project Workflow

```text
Amazon Kindle Reviews
        ↓
Data Loading
        ↓
Data Selection
        ↓
Sentiment Label Creation
        ↓
Text Preprocessing
        ↓
Train-Test Split
        ↓
 ┌───────────────┬────────────────┬──────────────────┐
 ↓               ↓                ↓
BoW            TF-IDF          Word2Vec
 ↓               ↓                ↓
Naive Bayes   Logistic        Random Forest
              Regression
 └───────────────┴────────────────┴──────────────────┘
                    ↓
              Model Evaluation
                    ↓
             Model Comparison
```

## 1. Data Loading

The dataset was loaded using Pandas and the required columns were selected:

```python
data = df[['reviewText', 'rating']]
```

The resulting dataset contains **12,000 rows and 2 columns**.

## 2. Text Preprocessing

A custom preprocessing function was created using regular expressions, NLTK stopwords, and WordNet Lemmatizer.

The preprocessing pipeline includes:

1. Convert text to string
2. Convert text to lowercase
3. Remove HTML tags
4. Remove URLs
5. Remove email addresses
6. Remove numbers
7. Remove punctuation
8. Remove extra spaces
9. Tokenize using `split()`
10. Remove English stopwords
11. Lemmatize words

The cleaned tokens are then joined back into a processed text string.

Example transformation:

```text
Original:
"Great short read! I didn't want to put it down."

Processed:
"great short read didnt want put read one sitting..."
```

## 3. Train-Test Split

The processed review text and sentiment labels were divided into training and testing sets using an **80/20 split** with `random_state=42`.

```python
train_test_split(
    data['reviewText'],
    data['rating'],
    test_size=0.20,
    random_state=42
)
```

This resulted in:

* **9,600 training samples**
* **2,400 testing samples**

## 4. Bag of Words (BoW)

The Bag of Words approach was implemented using Scikit-learn's `CountVectorizer`.

The configuration used:

```python
CountVectorizer(
    ngram_range=(1, 2),
    min_df=2,
    max_features=30000
)
```

This represents the text using word frequencies and includes both:

* Unigrams — single words
* Bigrams — pairs of consecutive words

The vocabulary was limited to a maximum of **30,000 features**.

### Model

A **Multinomial Naive Bayes** classifier was trained on the BoW features:

```python
MultinomialNB(alpha=0.1)
```

## 5. TF-IDF

TF-IDF (Term Frequency-Inverse Document Frequency) was used as another way to represent the review text numerically.

```python
TfidfVectorizer()
```

The TF-IDF representation was then used with **Logistic Regression**.

The Logistic Regression model was configured with:

```python
LogisticRegression(
    class_weight='balanced',
    max_iter=1000
)
```

## 6. Word2Vec

Word2Vec was used to create dense vector representations of words.

The model was trained using:

```python
Word2Vec(
    sentences=words,
    window=5,
    vector_size=200
)
```

Therefore, each word is represented using a **200-dimensional vector** with a context window of 5 words.

### Document Representation

Since Word2Vec generates vectors for individual words, the vectors of the words in each review were averaged to create a single vector representing the complete review.

Each review was therefore represented by a **200-dimensional feature vector**.

## 7. Machine Learning Model

A **Random Forest Classifier** was trained on the averaged Word2Vec representations.

```python
RandomForestClassifier(
    class_weight='balanced',
    random_state=42
)
```

The classifier used **100 decision trees by default** and balanced class weights during training.

## Model Performance

The different text representation techniques and classifiers were evaluated using accuracy, precision, recall, F1-score, and classification reports.

### Accuracy Comparison

| Feature Representation | Model                   |   Accuracy |
| ---------------------- | ----------------------- | ---------: |
| Bag of Words           | Multinomial Naive Bayes | **85.08%** |
| TF-IDF                 | Logistic Regression     | **84.96%** |
| Word2Vec               | Random Forest           | **76.08%** |

The best-performing approach in this notebook was **Bag of Words + Multinomial Naive Bayes**, achieving an accuracy of approximately **85.08%** on the test set.

## Detailed Results

### Bag of Words + Multinomial Naive Bayes

```text
Accuracy: 85.08%

              precision    recall  f1-score   support

           0       0.77      0.78      0.78       803
           1       0.89      0.88      0.89      1597

    accuracy                           0.85      2400
   macro avg       0.83      0.83      0.83      2400
weighted avg       0.85      0.85      0.85      2400
```

### TF-IDF + Logistic Regression

```text
Accuracy: 84.96%

              precision    recall  f1-score   support

           0       0.75      0.82      0.78       803
           1       0.90      0.87      0.88      1597

    accuracy                           0.85      2400
   macro avg       0.83      0.84      0.83      2400
weighted avg       0.85      0.85      0.85      2400
```

### Word2Vec + Random Forest

```text
Accuracy: 76.08%

              precision    recall  f1-score   support

           0       0.64      0.67      0.65       803
           1       0.83      0.81      0.82      1597

    accuracy                           0.76      2400
   macro avg       0.73      0.74      0.73      2400
weighted avg       0.76      0.76      0.76      2400
```

The Word2Vec representation produced 200 features per review, which were then used as input to the Random Forest classifier.

## Key Observations

* **BoW + Multinomial Naive Bayes** achieved the highest test accuracy of **85.08%**.
* **TF-IDF + Logistic Regression** produced a very similar accuracy of **84.96%**.
* **Word2Vec + Random Forest** achieved **76.08%** accuracy in this implementation.
* The BoW model performed particularly well despite using a relatively simple representation.
* Comparing multiple text representation techniques helped evaluate how different approaches affect sentiment classification performance.

## Why Compare Multiple NLP Techniques?

Different text representation techniques capture information in different ways:

### Bag of Words

Represents text based on word occurrence/frequency.

**Advantages:**

* Simple
* Easy to understand
* Effective for many text classification problems

### TF-IDF

Assigns higher importance to words that are important within a document but less common across the complete collection.

**Advantages:**

* Reduces the importance of very common words
* Useful for document classification
* Works well with linear models

### Word2Vec

Represents words as dense numerical vectors based on their contextual relationships.

**Advantages:**

* Captures semantic relationships between words
* Produces dense vector representations
* Can represent words in a continuous vector space

## Evaluation Metrics

The models were evaluated using:

* Accuracy
* Precision
* Recall
* F1-score
* Classification report

The evaluation was performed on the **2,400-sample test set**.

## Skills Demonstrated

This project demonstrates practical experience with:

* Natural Language Processing
* Text preprocessing
* Regular expressions
* Tokenization
* Stopword removal
* Lemmatization
* Feature engineering
* Bag of Words
* N-grams
* TF-IDF
* Word embeddings
* Word2Vec
* Naive Bayes
* Logistic Regression
* Random Forest
* Model evaluation
* Classification metrics
* Comparing machine learning approaches

## Future Improvements

Possible improvements to the project include:

* Hyperparameter tuning for the machine learning models
* Experimenting with different Word2Vec configurations
* Using pretrained word embeddings
* Trying more advanced NLP models
* Handling sentiment labels using alternative rating thresholds
* Experimenting with class balancing techniques
* Comparing with deep learning approaches such as LSTM, GRU, or Transformers
* Building an inference pipeline for classifying new reviews

## Project Structure

```text
NLP-sentiment-analysis/
│
├── kindle_sentiment_analysis.ipynb
│
└── README.md
```

## Notebook

The complete implementation and experiments are available in:

`kindle_sentiment_analysis.ipynb`

## Conclusion

This project demonstrates an end-to-end NLP sentiment classification workflow, starting from raw Amazon Kindle reviews and progressing through text preprocessing, feature extraction, machine learning, and model evaluation.

Three different approaches were compared:

```text
BoW       → Multinomial Naive Bayes → 85.08%
TF-IDF    → Logistic Regression     → 84.96%
Word2Vec  → Random Forest           → 76.08%
```

Among the approaches implemented in this notebook, **Bag of Words with Multinomial Naive Bayes achieved the best test accuracy of 85.08%**.

---

## Author

**Aditya Janagal**

GitHub: [AdityaJanagal](https://github.com/AdityaJanagal)
