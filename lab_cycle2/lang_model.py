import re
import random
import math
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
# 2. LOAD CORPUS AND SPLIT INTO TRAIN / TEST
# =========================================================

def load_corpus():

    try:
        with open(
            "corpus.txt",
            "r",
            encoding="utf-8"
        ) as file:

            text = file.read()

    except FileNotFoundError:

        print("ERROR: corpus.txt not found.")
        return [], []

    # Split into sentences
    sentences = [
        s.strip()
        for s in re.split(
            r"[.!?]+",
            text
        )
        if s.strip()
    ]

    random.seed(42)
    random.shuffle(sentences)

    split_point = int(
        0.8 * len(sentences)
    )

    train_sentences = sentences[
        :split_point
    ]

    test_sentences = sentences[
        split_point:
    ]

    return train_sentences, test_sentences


# =========================================================
# 3. BUILD UNIGRAM MODEL
# =========================================================

def build_unigram_model(
        train_sentences):

    counts = Counter()

    total_words = 0

    for sentence in train_sentences:

        words = tokenize(sentence)

        for word in words:

            counts[word] += 1
            total_words += 1

    return counts, total_words


# =========================================================
# 4. BUILD BIGRAM MODEL
# =========================================================

def build_bigram_model(
        train_sentences):

    unigram_counts = Counter()
    bigram_counts = Counter()

    for sentence in train_sentences:

        words = tokenize(sentence)

        if not words:
            continue

        words = [
            "<s>"
        ] + words + [
            "</s>"
        ]

        for word in words:
            unigram_counts[word] += 1

        for i in range(
                len(words) - 1):

            bigram_counts[
                (
                    words[i],
                    words[i + 1]
                )
            ] += 1

    return (
        unigram_counts,
        bigram_counts
    )


# =========================================================
# 5. DISPLAY UNIGRAMS
# =========================================================

def display_unigrams(
        unigram_counts,
        total_words):

    print("\nUNIGRAM MODEL")
    print("-" * 60)

    for word, count in \
            unigram_counts.most_common():

        probability = (
            count / total_words
        )

        print(
            f"{word:<15}"
            f"Count = {count:<5}"
            f"P = {probability:.6f}"
        )


# =========================================================
# 6. DISPLAY BIGRAMS
# =========================================================

def display_bigrams(
        unigram_counts,
        bigram_counts):

    print("\nBIGRAM MODEL")
    print("-" * 60)

    for (w1, w2), count in \
            bigram_counts.most_common():

        probability = (
            count /
            unigram_counts[w1]
        )

        print(
            f"({w1}, {w2})"
            f"\tCount = {count:<5}"
            f"P = {probability:.6f}"
        )


# =========================================================
# 7. UNIGRAM TEXT PROBABILITY
# =========================================================

def unigram_text_probability(
        text,
        unigram_counts,
        total_words):

    words = tokenize(text)

    if not words:
        return 0.0

    probability = 1.0

    for word in words:

        if word not in unigram_counts:

            return 0.0

        probability *= (
            unigram_counts[word]
            / total_words
        )

    return probability


# =========================================================
# 8. BIGRAM TEXT PROBABILITY
# =========================================================

def bigram_text_probability(
        text,
        unigram_counts,
        bigram_counts):

    words = tokenize(text)

    if not words:
        return 0.0

    words = [
        "<s>"
    ] + words + [
        "</s>"
    ]

    probability = 1.0

    for i in range(
            len(words) - 1):

        previous = words[i]
        current = words[i + 1]

        bigram_count = \
            bigram_counts[
                (previous, current)
            ]

        previous_count = \
            unigram_counts[
                previous
            ]

        if (
            bigram_count == 0
            or previous_count == 0
        ):

            return 0.0

        probability *= (
            bigram_count
            / previous_count
        )

    return probability


# =========================================================
# 9. RANDOM UNIGRAM SENTENCE
# =========================================================

def generate_unigram_sentence(
        unigram_counts,
        length=10):

    words = [
        word
        for word in unigram_counts
        if word not in {
            "<s>",
            "</s>"
        }
    ]

    weights = [
        unigram_counts[word]
        for word in words
    ]

    generated = random.choices(
        words,
        weights=weights,
        k=length
    )

    return " ".join(generated)


# =========================================================
# 10. RANDOM BIGRAM SENTENCE
# =========================================================

def generate_bigram_sentence(
        unigram_counts,
        bigram_counts,
        max_length=15):

    current = "<s>"

    sentence = []

    for _ in range(max_length):

        candidates = []

        weights = []

        for (w1, w2), count in \
                bigram_counts.items():

            if w1 == current:

                candidates.append(w2)
                weights.append(count)

        if not candidates:
            break

        next_word = random.choices(
            candidates,
            weights=weights,
            k=1
        )[0]

        if next_word == "</s>":
            break

        sentence.append(
            next_word
        )

        current = next_word

    return " ".join(sentence)


# =========================================================
# 11. UNIGRAM PERPLEXITY
# =========================================================

def unigram_perplexity(
        test_sentences,
        unigram_counts,
        total_words):

    words = []

    for sentence in test_sentences:
        words.extend(
            tokenize(sentence)
        )

    log_probability = 0.0

    count = 0

    for word in words:

        probability = (
            unigram_counts[word]
            / total_words
        )

        # Unsmoothed model:
        # unseen word => infinite perplexity

        if probability == 0:

            return float("inf")

        log_probability += math.log(
            probability
        )

        count += 1

    return math.exp(
        -log_probability / count
    )


# =========================================================
# 12. BIGRAM PERPLEXITY
# =========================================================

def bigram_perplexity(
        test_sentences,
        unigram_counts,
        bigram_counts):

    log_probability = 0.0

    count = 0

    for sentence in test_sentences:

        words = tokenize(sentence)

        if not words:
            continue

        words = [
            "<s>"
        ] + words + [
            "</s>"
        ]

        for i in range(
                len(words) - 1):

            previous = words[i]
            current = words[i + 1]

            bigram_count = \
                bigram_counts[
                    (previous, current)
                ]

            previous_count = \
                unigram_counts[
                    previous
                ]

            # Unsmoothed model
            if (
                bigram_count == 0
                or previous_count == 0
            ):

                return float("inf")

            probability = (
                bigram_count
                / previous_count
            )

            log_probability += math.log(
                probability
            )

            count += 1

    if count == 0:
        return float("inf")

    return math.exp(
        -log_probability / count
    )


# =========================================================
# 13. MAIN MENU
# =========================================================

def main():

    print("=" * 60)
    print("UNIGRAM AND BIGRAM LANGUAGE MODEL")
    print("=" * 60)

    # -----------------------------------------------------
    # Load data
    # -----------------------------------------------------

    train_sentences, test_sentences = \
        load_corpus()

    if not train_sentences:
        return

    print(
        "\nTraining sentences:",
        len(train_sentences)
    )

    print(
        "Testing sentences:",
        len(test_sentences)
    )

    # -----------------------------------------------------
    # Build models
    # -----------------------------------------------------

    unigram_counts, total_words = \
        build_unigram_model(
            train_sentences
        )

    (
        bigram_unigram_counts,
        bigram_counts
    ) = build_bigram_model(
        train_sentences
    )

    print(
        "Training vocabulary:",
        len(unigram_counts)
    )

    while True:

        print("\n" + "=" * 60)
        print("MENU")
        print("=" * 60)

        print("1. Display unigram probabilities")
        print("2. Display bigram probabilities")
        print("3. Calculate probability of text")
        print("4. Generate unigram sentence")
        print("5. Generate bigram sentence")
        print("6. Calculate perplexity")
        print("7. Exit")

        choice = input(
            "\nEnter your choice: "
        )

        # =================================================
        # OPTION 1
        # =================================================

        if choice == "1":

            display_unigrams(
                unigram_counts,
                total_words
            )

        # =================================================
        # OPTION 2
        # =================================================

        elif choice == "2":

            display_bigrams(
                bigram_unigram_counts,
                bigram_counts
            )

        # =================================================
        # OPTION 3
        # =================================================

        elif choice == "3":

            text = input(
                "\nEnter text: "
            )

            unigram_probability = \
                unigram_text_probability(
                    text,
                    unigram_counts,
                    total_words
                )

            bigram_probability = \
                bigram_text_probability(
                    text,
                    bigram_unigram_counts,
                    bigram_counts
                )

            print(
                "\nUnigram probability:",
                unigram_probability
            )

            print(
                "Bigram probability:",
                bigram_probability
            )

        # =================================================
        # OPTION 4
        # =================================================

        elif choice == "4":

            try:
                length = int(
                    input(
                        "\nSentence length: "
                    )
                )
            except ValueError:
                length = 10

            print(
                "\nGenerated Unigram Sentence:"
            )

            print(
                generate_unigram_sentence(
                    unigram_counts,
                    length
                )
            )

        # =================================================
        # OPTION 5
        # =================================================

        elif choice == "5":

            print(
                "\nGenerated Bigram Sentence:"
            )

            print(
                generate_bigram_sentence(
                    bigram_unigram_counts,
                    bigram_counts
                )
            )

        # =================================================
        # OPTION 6
        # =================================================

        elif choice == "6":

            print(
                "\nCalculating perplexity..."
            )

            unigram_pp = \
                unigram_perplexity(
                    test_sentences,
                    unigram_counts,
                    total_words
                )

            bigram_pp = \
                bigram_perplexity(
                    test_sentences,
                    bigram_unigram_counts,
                    bigram_counts
                )

            print(
                "\nUnigram Perplexity:",
                unigram_pp
            )

            print(
                "Bigram Perplexity:",
                bigram_pp
            )

            print("\nComparison:")

            if unigram_pp < bigram_pp:

                print(
                    "Unigram model has lower "
                    "perplexity."
                )

            elif bigram_pp < unigram_pp:

                print(
                    "Bigram model has lower "
                    "perplexity."
                )

            else:

                print(
                    "Both models have equal "
                    "perplexity."
                )

        # =================================================
        # OPTION 7
        # =================================================

        elif choice == "7":

            print(
                "\nExiting..."
            )

            break

        else:

            print(
                "\nInvalid choice."
            )


if __name__ == "__main__":
    main()