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
    cla
```
