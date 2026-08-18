import re
import math
import numpy as np
from collections import Counter


# =========================================================
# 1. TOKENIZATION
# =========================================================

def tokenize(text):
    return re.findall(
        r"[a-z]+(?:'[a-z]+)?",
        text.lower()
    )


# =========================================================
# 2. CONFUSION SETS
# =========================================================

CONFUSION_SETS = [
    {"write", "right", "rite"},
    {"peace", "piece"},
    {"their", "there", "they're"}
]


# =========================================================
# 3. FIND CONFUSION SET
# =========================================================

def get_confusion_set(word):

    for confusion_set in CONFUSION_SETS:
        if word in confusion_set:
            return confusion_set

    return None


# =========================================================
# 4. BUILD LANGUAGE MODEL
# =========================================================

def build_model(corpus):

    words = tokenize(corpus)

    unigram_counts = Counter(words)

    bigram_counts = Counter(
        zip(words[:-1], words[1:])
    )

    vocabulary_size = len(unigram_counts)
    total_words = len(words)

    return (
        unigram_counts,
        bigram_counts,
        vocabulary_size,
        total_words
    )


# =========================================================
# 5. BIGRAM PROBABILITY
# =========================================================

def bigram_probability(
        previous_word,
        current_word,
        unigram_counts,
        bigram_counts,
        vocabulary_size):

    bigram_count = bigram_counts[
        (previous_word, current_word)
    ]

    previous_count = unigram_counts[
        previous_word
    ]

    return (
        bigram_count + 1
    ) / (
        previous_count + vocabulary_size
    )


# =========================================================
# 6. FEATURE EXTRACTION
# =========================================================

def extract_features(
        candidate,
        previous_word,
        next_word,
        original_word,
        unigram_counts,
        bigram_counts,
        vocabulary_size,
        total_words):

    features = []

    # -----------------------------------------------------
    # Feature 1: previous-word bigram probability
    # -----------------------------------------------------

    if previous_word:

        p_previous = bigram_probability(
            previous_word,
            candidate,
            unigram_counts,
            bigram_counts,
            vocabulary_size
        )

        features.append(math.log(p_previous))

    else:
        features.append(0.0)

    # -----------------------------------------------------
    # Feature 2: next-word bigram probability
    # -----------------------------------------------------

    if next_word:

        p_next = bigram_probability(
            candidate,
            next_word,
            unigram_counts,
            bigram_counts,
            vocabulary_size
        )

        features.append(math.log(p_next))

    else:
        features.append(0.0)

    # -----------------------------------------------------
    # Feature 3: unigram probability
    # -----------------------------------------------------

    unigram_probability = (
        unigram_counts[candidate] + 1
    ) / (
        total_words + vocabulary_size
    )

    features.append(
        math.log(unigram_probability)
    )

    # -----------------------------------------------------
    # Feature 4: exact match
    # -----------------------------------------------------

    features.append(
        1 if candidate == original_word else 0
    )

    # -----------------------------------------------------
    # Feature 5: word length
    # -----------------------------------------------------

    features.append(
        len(candidate)
    )

    return features


# =========================================================
# 7. CREATE TRAINING DATA
# =========================================================

def create_training_data(
        corpus,
        unigram_counts,
        bigram_counts,
        vocabulary_size,
        total_words):

    words = tokenize(corpus)

    X = []
    y = []

    for i, correct_word in enumerate(words):

        confusion_set = get_confusion_set(
            correct_word
        )

        if confusion_set is None:
            continue

        previous_word = (
            words[i - 1]
            if i > 0
            else None
        )

        next_word = (
            words[i + 1]
            if i < len(words) - 1
            else None
        )

        for candidate in confusion_set:

            features = extract_features(
                candidate,
                previous_word,
                next_word,
                correct_word,
                unigram_counts,
                bigram_counts,
                vocabulary_size,
                total_words
            )

            X.append(features)

            if candidate == correct_word:
                y.append(1)
            else:
                y.append(0)

    return np.array(X, dtype=float), np.array(y, dtype=float)


# =========================================================
# 8. SIGMOID
# =========================================================

def sigmoid(z):

    # Prevent overflow
    z = np.clip(z, -500, 500)

    return 1 / (1 + np.exp(-z))


# =========================================================
# 9. MANUAL BINARY LOGISTIC REGRESSION
# =========================================================

class LogisticRegressionManual:

    def __init__(
            self,
            learning_rate=0.05,
            epochs=5000):

        self.learning_rate = learning_rate
        self.epochs = epochs
        self.weights = None
        self.bias = 0.0

    def fit(self, X, y):

        n_samples, n_features = X.shape

        self.weights = np.zeros(
            n_features
        )

        self.bias = 0.0

        for epoch in range(self.epochs):

            # Linear combination
            z = np.dot(
                X,
                self.weights
            ) + self.bias

            # Prediction
            predictions = sigmoid(z)

            # Gradients
            dw = (
                1 / n_samples
            ) * np.dot(
                X.T,
                predictions - y
            )

            db = (
                1 / n_samples
            ) * np.sum(
                predictions - y
            )

            # Update
            self.weights -= (
                self.learning_rate * dw
            )

            self.bias -= (
                self.learning_rate * db
            )

    def predict_probability(self, X):

        z = np.dot(
            X,
            self.weights
        ) + self.bias

        return sigmoid(z)


# =========================================================
# 10. NORMALIZE FEATURES
# =========================================================

def normalize_features(X):

    mean = X.mean(axis=0)
    std = X.std(axis=0)

    # Avoid division by zero
    std[std == 0] = 1

    X_normalized = (
        X - mean
    ) / std

    return X_normalized, mean, std


# =========================================================
# 11. CORRECT INPUT TEXT
# =========================================================

def correct_text(
        text,
        model,
        feature_mean,
        feature_std,
        unigram_counts,
        bigram_counts,
        vocabulary_size,
        total_words):

    words = tokenize(text)

    corrected_words = words.copy()

    results = []

    for i, original_word in enumerate(words):

        confusion_set = get_confusion_set(
            original_word
        )

        if confusion_set is None:
            continue

        previous_word = (
            words[i - 1]
            if i > 0
            else None
        )

        next_word = (
            words[i + 1]
            if i < len(words) - 1
            else None
        )

        candidate_scores = {}

        for candidate in confusion_set:

            features = extract_features(
                candidate,
                previous_word,
                next_word,
                original_word,
                unigram_counts,
                bigram_counts,
                vocabulary_size,
                total_words
            )

            features = np.array(
                features,
                dtype=float
            )

            # Same normalization used during training
            features = (
                features - feature_mean
            ) / feature_std

            probability = model.predict_probability(
                features.reshape(1, -1)
            )[0]

            candidate_scores[
                candidate
            ] = probability

        best_candidate = max(
            candidate_scores,
            key=candidate_scores.get
        )

        corrected_words[i] = best_candidate

        results.append(
            (
                original_word,
                candidate_scores,
                best_candidate
            )
        )

    return corrected_words, results


# =========================================================
# 12. DISPLAY RESULTS
# =========================================================

def display_results(results):

    print("\n" + "=" * 60)
    print("HOMOPHONE CORRECTION RESULTS")
    print("=" * 60)

    if not results:

        print(
            "\nNo confusion-set words detected."
        )

        return

    for original, scores, best in results:

        print(
            f"\nInput word: {original}"
        )

        print(
            "Candidate probabilities:"
        )

        for candidate, probability in sorted(
                scores.items(),
                key=lambda x: x[1],
                reverse=True):

            print(
                f"  {candidate:<10}"
                f"P(y=1) = {probability:.4f}"
            )

        print(
            f"Selected word: {best}"
        )


# =========================================================
# 13. MAIN
# =========================================================

def main():

    print("=" * 60)
    print("BINARY LOGISTIC REGRESSION")
    print("HOMOPHONE ERROR CORRECTOR")
    print("=" * 60)

    # -----------------------------------------------------
    # Load corpus
    # -----------------------------------------------------

    try:

        with open(
            "corpus.txt",
            "r",
            encoding="utf-8"
        ) as file:

            corpus = file.read()

    except FileNotFoundError:

        print(
            "\nERROR: corpus.txt not found."
        )

        return

    # -----------------------------------------------------
    # Build language model
    # -----------------------------------------------------

    (
        unigram_counts,
        bigram_counts,
        vocabulary_size,
        total_words
    ) = build_model(corpus)

    print(
        "\nCorpus loaded successfully."
    )

    # -----------------------------------------------------
    # Training data
    # -----------------------------------------------------

    X, y = create_training_data(
        corpus,
        unigram_counts,
        bigram_counts,
        vocabulary_size,
        total_words
    )

    print(
        "Training examples:",
        len(X)
    )

    # -----------------------------------------------------
    # Normalize training features
    # -----------------------------------------------------

    X_normalized, feature_mean, feature_std = \
        normalize_features(X)

    # -----------------------------------------------------
    # Train manual logistic regression
    # -----------------------------------------------------

    model = LogisticRegressionManual(
        learning_rate=0.05,
        epochs=5000
    )

    model.fit(
        X_normalized,
        y
    )

    print(
        "Manual Logistic Regression trained."
    )

    print(
        "\nLearned weights:"
    )

    for i, weight in enumerate(
            model.weights):

        print(
            f"Feature {i + 1}: {weight:.6f}"
        )

    print(
        f"Bias: {model.bias:.6f}"
    )

    # -----------------------------------------------------
    # User input
    # -----------------------------------------------------

    text = input(
        "\nEnter text containing possible "
        "homophone errors:\n"
    )

    # -----------------------------------------------------
    # Correct
    # -----------------------------------------------------

    corrected_words, results = correct_text(
        text,
        model,
        feature_mean,
        feature_std,
        unigram_counts,
        bigram_counts,
        vocabulary_size,
        total_words
    )

    # -----------------------------------------------------
    # Display
    # -----------------------------------------------------

    display_results(results)

    print("\n" + "=" * 60)
    print("CORRECTED TEXT")
    print("=" * 60)

    print(
        " ".join(corrected_words)
    )


if __name__ == "__main__":
    main()