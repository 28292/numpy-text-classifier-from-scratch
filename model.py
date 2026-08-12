"""
NumPy Text Classifier from Scratch

Assembled from your step-by-step solutions.
"""

import numpy as np

# Step 1 - clean_text
import re
def clean_text(text: str) -> str:
    # TODO: Lowercase text and replace non-alphabetic chars with spaces
    # lowercase the input
    text = text.lower()
    return re.sub(r"[^a-zA-Z]", " ", text).strip()

# Step 2 - tokenize
def tokenize(text: str) -> list[str]:
    # TODO: Split cleaned text on whitespace into non-empty word tokens
    return text.split()

# Step 3 - tokenize_corpus
def tokenize_corpus(texts: list) -> list:
    # TODO: Apply clean_text and tokenize to every document so the full corpus becomes a list of token lists.
    corpus = []
    for text in texts:
        cleaned_text = clean_text(text)
        tokenized_text = tokenize(cleaned_text)
        corpus.append(tokenized_text)
    return corpus

# Step 4 - split_train_val_test_indices
def split_train_val_test_indices(n_samples: int, val_fraction: float, test_fraction: float, seed: int = 0) -> tuple:
    # TODO: Produce shuffled index arrays that partition n_samples into train/val/test
    np.random.seed(seed)
    indices = np.arange(n_samples)
    np.random.shuffle(indices)

    n_val = int(n_samples * val_fraction)
    n_test = int(n_samples * test_fraction)
    n_train = n_samples - n_val - n_test

    train_idx = indices[:n_train]
    val_idx = indices[n_train:n_train + n_val]
    test_idx = indices[n_train + n_val:]

    return train_idx, val_idx, test_idx

# Step 5 - count_word_frequencies
def count_word_frequencies(tokenized_docs: list) -> dict:
    # TODO: Return a dict mapping each unique token to its total count...
    word_count_dict = {}
    for single_list in tokenized_docs:
        for token in single_list:
            word_count_dict[token] = word_count_dict.get(token, 0) + 1
    return word_count_dict

# Step 6 - build_vocabulary
def build_vocabulary(word_counts: dict, max_size: int) -> dict:
    result_dict = {}
    max_word_list = sorted(word_counts.items(), key = lambda item: (-item[1], item[0]), reverse = False)[:max_size]
    for index, (word, count) in enumerate(max_word_list):
        result_dict[word] = index
    return result_dict

# Step 7 - tokens_to_bow
def tokens_to_bow(tokens: list, vocab: dict) -> np.ndarray:
#     # TODO: Convert one document's token list into a bag-of-words count vector...
#     # create an array with 0 value which has length of vocab.
    bow = np.zeros(len(vocab), dtype = float)
    for token in tokens:
        if token in vocab:
            token_value = vocab[token]
            bow[token_value] += 1
    return bow

# Step 8 - corpus_to_bow_matrix
def corpus_to_bow_matrix(tokenized_docs: list, vocab: dict) -> np.ndarray:
    # TODO: Stack per-document BoW vectors into a 2-D count matrix for a whole corpus.
    number_of_docs = len(tokenized_docs)
    vocabulary_size = len(vocab)

    matrix = np.zeros((number_of_docs, vocabulary_size), dtype=float)
    for row_index, tokens in enumerate(tokenized_docs):
        matrix[row_index] = tokens_to_bow(tokens, vocab)
    return matrix

# Step 9 - compute_document_frequencies
def compute_document_frequencies(bow_matrix: np.ndarray) -> np.ndarray:
    # TODO: Count docs where each term appears at least once (df, shape (V,))
    # Find where count > 0
    x = [[(lambda value: value > 0)(value) for value in values] for row_index,values in enumerate(bow_matrix)]
    # Count across documents
    result = np.array([sum(i) for i in zip(*x)])
    return result

# Step 10 - compute_idf
import math

def compute_idf(df: np.ndarray, n_docs: int) -> np.ndarray:
    # TODO: Compute smoothed IDF idf_j = log((n_docs + 1) / (df_j + 1)) + 1
    numerator = n_docs + 1
    denominator = df + 1
    result = np.log(numerator / denominator) + 1
    return result

# Step 11 - transform_tfidf
def transform_tfidf(bow_matrix: np.ndarray, idf: np.ndarray) -> np.ndarray:
    # TODO: Multiply BoW counts by the fitted IDF vector to produce TF-IDF features.
    result = bow_matrix * idf
    return result

# Step 12 - fit_tfidf
def fit_tfidf(bow_train: np.ndarray) -> np.ndarray:
    # TODO: Fit IDF on the training BoW matrix by chaining DF and IDF.
    df = compute_document_frequencies(bow_train)
    return compute_idf(df, bow_train.shape[0])

# Step 13 - sigmoid
def sigmoid(z: np.ndarray) -> np.ndarray:
    result = np.array([1/(1+np.exp(-v)) for v in z])
    return result

    # TODO: Map logits to probabilities with a numerically stable logistic sigmoid.

# Step 14 - logistic_predict_proba
def logistic_predict_proba(X: np.ndarray, w: np.ndarray, b: float) -> np.ndarray:
    # TODO: Return P(y=1|x) for each row via linear scores and sigmoid
    logistic_vector = X @ w + b 
    prob = sigmoid(logistic_vector)
    return prob

# Step 15 - binary_cross_entropy
def binary_cross_entropy(y_true: np.ndarray, y_prob: np.ndarray, w: np.ndarray, l2_lambda: float) -> float:
    # TODO: Compute mean binary cross-entropy plus L2 penalty on the weights.
    length_y_label = len(y_true)
    losses = []
    for i, j in zip(y_true, y_prob):
        single_loss = (np.log(j)*i + (1-i)*np.log(1-j))* -1
        losses.append(single_loss)
    mean_bce = np.sum(losses)/length_y_label

    # Part 2: calculate L2 penalty
    weight_sum = np.sum(w**2)

    l2_score = 0.5 * l2_lambda * weight_sum
    # Total loss
    total_loss = mean_bce + l2_score
    
    
    return total_loss

# Step 16 - logistic_gradients
def logistic_gradients(X: np.ndarray, y_true: np.ndarray, y_prob: np.ndarray, w: np.ndarray, l2_lambda: float) -> tuple:
    """Compute gradients of BCE+L2 w.r.t. weights and bias for one full batch.

    Args:
        X: Feature matrix of shape (N, D).
        y_true: Binary labels of shape (N,).
        y_prob: Predicted probabilities of shape (N,).
        w: Weight vector of shape (D,).
        l2_lambda: L2 regularization strength.

    Returns:
        Tuple (dw, db) with dw shape (D,) and db a float.
    """

    X_T = np.transpose(X)
    length_output = len(y_true)
    error = y_prob - y_true
    dw = X_T @ error / length_output + l2_lambda * w
    db = np.sum(error) / length_output
    return dw, db



    # TODO: Compute gradients of BCE+L2 w.r.t. weights and bias for one full batch.

# Step 17 - initialize_logistic_params
def initialize_logistic_params(n_features: int) -> tuple:
    # TODO: Return a zero weight vector of shape (n_features,) and bias 0.0
    w = np.zeros(n_features, dtype=float)
    b = 0.0
    return w, b

# Step 18 - gradient_descent_step
def gradient_descent_step(X: np.ndarray, y: np.ndarray, w: np.ndarray, b: float, lr: float, l2_lambda: float) -> tuple:
    # TODO: Run one full-batch gradient descent update; return (w_new, b_new, loss).
    # prediction
    y_prob = logistic_predict_proba(X, w,b)
    # Loss before updating
    loss = binary_cross_entropy(y, y_prob, w, l2_lambda)
    # calculate gradient
    dw, db = logistic_gradients(X, y, y_prob, w, l2_lambda)
    # update paramter
    w_new = w - lr * dw
    b_new = b - lr * db
    return w_new, b_new, float(loss)

# Step 19 - train_logistic_regression (not yet solved)
# TODO: implement

# Step 20 - predict_labels (not yet solved)
# TODO: implement

# Step 21 - confusion_counts (not yet solved)
# TODO: implement

# Step 22 - metrics_from_counts (not yet solved)
# TODO: implement

# Step 23 - tune_decision_threshold (not yet solved)
# TODO: implement

# Step 24 - evaluate_predictions (not yet solved)
# TODO: implement

# Step 25 - vectorize_texts (not yet solved)
# TODO: implement

# Step 26 - predict_text (not yet solved)
# TODO: implement

# Step 27 - collect_prediction_errors (not yet solved)
# TODO: implement

