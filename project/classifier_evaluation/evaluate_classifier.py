import pandas as pd
from collections import defaultdict

from agent import build_chain, identify_attack_type_smart

def normalize_labels(label_text):
    """
    Converts:
    'prompt_injection,data_extraction'
    ->
    {'prompt_injection','data_extraction'}

    Converts:
    'none'
    ->
    set()
    """

    if pd.isna(label_text):
        return set()

    labels = {
        item.strip().lower()
        for item in str(label_text).split(",")
        if item.strip()
    }

    if labels == {"none"}:
        return set()

    return labels


def evaluate():

    print("Loading dataset...")

    df = pd.read_csv("evaluation/evaluation_dataset.csv")

    chain = build_chain()

    total = 0
    exact_matches = 0

    class_stats = defaultdict(
        lambda: {
            "tp": 0,
            "fp": 0,
            "fn": 0
        }
    )

    print("\nRunning Evaluation...\n")

    for _, row in df.iterrows():

        message = row["message"]

        expected = normalize_labels(
            row["expected_label"]
        )

        predicted = set(
            identify_attack_type_smart(
                chain,
                message
            )
        )

        total += 1

        if predicted == expected:
            exact_matches += 1

        else:
            print("=" * 70)
            print("❌ MISCLASSIFIED SAMPLE")
            print("-" * 70)
            print("MESSAGE:")
            print(message)
            print()
            print("EXPECTED:")
            print(sorted(expected))
            print()
            print("PREDICTED:")
            print(sorted(predicted))
            print()

        all_labels = expected.union(predicted)

        for label in all_labels:

            if label in expected and label in predicted:
                class_stats[label]["tp"] += 1

            elif label not in expected and label in predicted:
                class_stats[label]["fp"] += 1

            elif label in expected and label not in predicted:
                class_stats[label]["fn"] += 1

    print("\n")
    print("=" * 70)
    print("FINAL RESULTS")
    print("=" * 70)

    accuracy = (exact_matches / total) * 100

    print(f"Total Samples : {total}")
    print(f"Exact Matches : {exact_matches}")
    print(f"Accuracy      : {accuracy:.2f}%")

    print("\nPER CLASS METRICS")

    for label, stats in sorted(class_stats.items()):

        tp = stats["tp"]
        fp = stats["fp"]
        fn = stats["fn"]

        precision = (
            tp / (tp + fp)
            if (tp + fp) > 0
            else 0
        )

        recall = (
            tp / (tp + fn)
            if (tp + fn) > 0
            else 0
        )

        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0
        )

        print("\n----------------------------")
        print(f"Class     : {label}")
        print(f"TP        : {tp}")
        print(f"FP        : {fp}")
        print(f"FN        : {fn}")
        print(f"Precision : {precision:.2f}")
        print(f"Recall    : {recall:.2f}")
        print(f"F1 Score  : {f1:.2f}")

    print("\n" + "=" * 70)
    print("Evaluation Completed")
    print("=" * 70)


if __name__ == "__main__":
    evaluate()