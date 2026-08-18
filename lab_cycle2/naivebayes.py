import math
import random
import re
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
# 2. CREATE UNIGRAM + BIGRAM FEATURES
# =========================================================

def extract_features(text):

    words = tokenize(text)

    features = []

    # Unigrams
    features.extend(words)

    # Bigrams
    for i in range(len(words) - 1):
        features.append(
            words[i] + "_" + words[i + 1]
        )

    return features


# =========================================================
# 3. DATASET
# =========================================================

dataset = [

    # -------------------------
    # POSITIVE
    # -------------------------

    ("I loved this movie", "positive"),
    ("This movie was excellent", "positive"),
    ("The film was amazing", "positive"),
    ("I really enjoyed the movie", "positive"),
    ("The story was wonderful", "positive"),
    ("The acting was fantastic", "positive"),
    ("This was a great experience", "positive"),
    ("The movie was very good", "positive"),
    ("I liked the film", "positive"),
    ("The characters were excellent", "positive"),
    ("The story was beautiful", "positive"),
    ("The movie was enjoyable", "positive"),
    ("I would recommend this movie", "positive"),
    ("The film was brilliant", "positive"),
    ("The actors were fantastic", "positive"),
    ("This is a great film", "positive"),
    ("I enjoyed every minute", "positive"),
    ("The movie made me happy", "positive"),
    ("The ending was wonderful", "positive"),
    ("A very entertaining movie", "positive"),

    ("I am very happy today", "positive"),
    ("I had a wonderful day", "positive"),
    ("I feel great today", "positive"),
    ("I am extremely happy", "positive"),
    ("The experience was excellent", "positive"),
    ("I really liked the result", "positive"),
    ("I enjoyed the experience", "positive"),
    ("The result made me happy", "positive"),
    ("This was a pleasant experience", "positive"),
    ("I am satisfied with the result", "positive"),
    ("The movie was really good", "positive"),
    ("The service was excellent", "positive"),
    ("I had a great time", "positive"),
    ("Everything was wonderful", "positive"),
    ("I feel very good", "positive"),
    ("It was a happy moment", "positive"),
    ("I felt happy after the event", "positive"),
    ("The day was wonderful", "positive"),
    ("The result was very good", "positive"),
    ("I had an amazing experience", "positive"),

    # -------------------------
    # NEGATIVE
    # -------------------------

    ("I hated this movie", "negative"),
    ("This movie was terrible", "negative"),
    ("The film was awful", "negative"),
    ("I really disliked the movie", "negative"),
    ("The story was boring", "negative"),
    ("The acting was terrible", "negative"),
    ("This was a bad experience", "negative"),
    ("The movie was very bad", "negative"),
    ("I disliked the film", "negative"),
    ("The characters were boring", "negative"),
    ("The story was disappointing", "negative"),
    ("The movie was unpleasant", "negative"),
    ("I would not recommend this movie", "negative"),
    ("The film was horrible", "negative"),
    ("The actors were terrible", "negative"),
    ("This is a bad film", "negative"),
    ("I disliked every minute", "negative"),
    ("The movie made me angry", "negative"),
    ("The ending was disappointing", "negative"),
    ("A very boring movie", "negative"),

    ("I was really sad", "negative"),
    ("I felt very sad", "negative"),
    ("I was angry after he broke the plate", "negative"),
    ("The situation made me angry", "negative"),
    ("I hated the experience", "negative"),
    ("The experience was horrible", "negative"),
    ("I was disappointed with the result", "negative"),
    ("The result was terrible", "negative"),
    ("I had a terrible day", "negative"),
    ("I felt bad after the incident", "negative"),
    ("The movie made me angry", "negative"),
    ("The service was awful", "negative"),
    ("I did not enjoy the experience", "negative"),
    ("I was not happy with the result", "negative"),
    ("The experience was not good", "negative"),
    ("I did not like the movie", "negative"),
    ("I was unhappy with the result", "negative"),
    ("The result was disappointing", "negative"),
    ("I felt terrible after the incident", "negative"),
    ("I had a bad day", "negative"),
    ("I was sad after he kicked me out", "negative"),
    ("I am bound to get angry", "negative"),
    ("It was a sad moment", "negative"),
    ("I feel very bad today", "negative"),
    ("I had a horrible experience", "negative"),
]


# =========================================================
# 4. TRAIN / TEST SPLIT
# =========================================================

random.seed(42)

random.shuffle(dataset)

split = int(
    0.8 * len(dataset)
)

train_data = dataset[:split]
test_data = dataset[split:]


# =========================================================
# 5. TRAIN NAIVE BAYES
# =========================================================

def train_naive_bayes(training_data):

    class_counts = Counter()

    feature_counts = {
        "positive": Counter(),
        "negative": Counter()
    }

    total_features = {
        "positive": 0,
        "negative": 0
    }

    vocabulary = set()

    for text, label in training_data:

        class_counts[label] += 1

        features = extract_features(text)

        for feature in features:

            feature_counts[label][feature] += 1

            total_features[label] += 1

            vocabulary.add(feature)

    return (
        class_counts,
        feature_counts,
        total_features,
        vocabulary
    )


# =========================================================
# 6. PREDICT
# =========================================================

def predict(
        text,
        class_counts,
        feature_counts,
        total_features,
        vocabulary,
        k):

    features = extract_features(text)

    total_documents = sum(
        class_counts.values()
    )

    scores = {}

    for sentiment in [
        "positive",
        "negative"
    ]:

        # -------------------------------------------------
        # Prior
        # -------------------------------------------------

        prior = (
            class_counts[sentiment]
            / total_documents
        )

        log_probability = math.log(
            prior
        )

        # -------------------------------------------------
        # Add-k smoothing
        # -------------------------------------------------

        denominator = (
            total_features[sentiment]
            + k * len(vocabulary)
        )

        for feature in features:

            count = feature_counts[
                sentiment
            ][feature]

            probability = (
                count + k
            ) / denominator

            log_probability += math.log(
                probability
            )

        scores[sentiment] = \
            log_probability

    prediction = max(
        scores,
        key=scores.get
    )

    return prediction, scores


# =========================================================
# 7. EVALUATION
# =========================================================

def evaluate(
        test_data,
        class_counts,
        feature_counts,
        total_features,
        vocabulary,
        k):

    correct = 0

    predictions = []

    for text, actual in test_data:

        predicted, scores = predict(
            text,
            class_counts,
            feature_counts,
            total_features,
            vocabulary,
            k
        )

        predictions.append(
            (
                text,
                actual,
                predicted
            )
        )

        if predicted == actual:
            correct += 1

    accuracy = (
        correct / len(test_data)
    ) * 100

    return accuracy, predictions


# =========================================================
# 8. DISPLAY PREDICTIONS
# =========================================================

def display_predictions(
        predictions):

    print("\nPredictions:")
    print("-" * 75)

    for text, actual, predicted in predictions:

        if actual == predicted:
            status = "CORRECT"
        else:
            status = "WRONG"

        print(
            f"{status:<8} "
            f"Actual: {actual:<9} "
            f"Predicted: {predicted:<9} "
            f"| {text}"
        )


# =========================================================
# 9. MAIN
# =========================================================

def main():

    print("=" * 75)
    print("NAIVE BAYES SENTIMENT CLASSIFIER")
    print("UNIGRAM + BIGRAM FEATURES")
    print("ADD-k SMOOTHING")
    print("=" * 75)

    print(
        f"\nTotal samples: {len(dataset)}"
    )

    print(
        f"Training samples: {len(train_data)}"
    )

    print(
        f"Testing samples: {len(test_data)}"
    )

    # -----------------------------------------------------
    # Train
    # -----------------------------------------------------

    (
        class_counts,
        feature_counts,
        total_features,
        vocabulary
    ) = train_naive_bayes(
        train_data
    )

    print(
        f"Feature vocabulary: {len(vocabulary)}"
    )

    # -----------------------------------------------------
    # Compare k values
    # -----------------------------------------------------

    k_values = [
        0.25,
        0.75,
        1.0
    ]

    results = {}

    print("\n" + "=" * 75)
    print("ADD-k SMOOTHING COMPARISON")
    print("=" * 75)

    for k in k_values:

        accuracy, predictions = evaluate(
            test_data,
            class_counts,
            feature_counts,
            total_features,
            vocabulary,
            k
        )

        results[k] = accuracy

        print(
            f"k = {k:<5} "
            f"Accuracy = {accuracy:.2f}%"
        )

    # -----------------------------------------------------
    # Best k
    # -----------------------------------------------------

    best_k = max(
        results,
        key=results.get
    )

    print("\n" + "=" * 75)

    print(
        f"Best k: {best_k}"
    )

    print(
        f"Best accuracy: "
        f"{results[best_k]:.2f}%"
    )

    print("=" * 75)

    # -----------------------------------------------------
    # Detailed results
    # -----------------------------------------------------

    _, best_predictions = evaluate(
        test_data,
        class_counts,
        feature_counts,
        total_features,
        vocabulary,
        best_k
    )

    display_predictions(
        best_predictions
    )

    # -----------------------------------------------------
    # Interactive testing
    # -----------------------------------------------------

    print("\n" + "=" * 75)
    print("INTERACTIVE SENTIMENT ANALYSIS")
    print("=" * 75)

    while True:

        text = input(
            "\nEnter a sentence "
            "(or 'quit'):\n"
        )

        if text.lower() == "quit":
            break

        predicted, scores = predict(
            text,
            class_counts,
            feature_counts,
            total_features,
            vocabulary,
            best_k
        )

        print(
            "\nPredicted sentiment:",
            predicted
        )

        print(
            "Positive score:",
            scores["positive"]
        )

        print(
            "Negative score:",
            scores["negative"]
        )


if __name__ == "__main__":
    main()