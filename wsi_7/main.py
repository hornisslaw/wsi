"""
Introduction to Artificial Intelligence, Exercise 7:
Bayesian models.
Author: Robert Kaczmarski 293377
"""
from __future__ import annotations

import argparse
import random
import numpy as np

from collections import defaultdict
from typing import Union

from plots import scatter_plot, confusion_matrix_plot

ROW = list[float]
DATA = list[ROW]
LABELS = list[str]
PARAMETERS = dict[str, float]
DATA_BY_CLASS = dict[str, DATA]
FEATURES = dict[str, Union[float, list[PARAMETERS]]]
SUMMARY_DICT = dict[str, PARAMETERS]
DATASET = list[list[Union[float, str]]]


def read_data(d: str) -> DATASET:
    data = []
    lines = d.splitlines()
    for line in lines:
        if line:
            data.append(
                [float(l) if not l.startswith("Iris") else l for l in line.split(",")]
            )
    return data


def split_data(
    data: DATASET, ratio: float, shuffle: bool = False
) -> tuple[DATA, LABELS, DATA, LABELS]:

    if shuffle:
        random.shuffle(data)

    train_size = int(len(data) * ratio)

    train_data = [row[:4] for row in data[:train_size]]
    test_data = [row[:4] for row in data[train_size:]]
    train_labels = [row[-1] for row in data[:train_size]]
    test_labels = [row[-1] for row in data[train_size:]]

    return (train_data, train_labels, test_data, test_labels)


def unique_classes(data: DATASET, class_column_index: int) -> list[str]:
    """Extract unique class values from dataset class"""
    return list(set(list(zip(*data))[class_column_index]))


def group_by_class(data: DATA, labels: LABELS) -> DATA_BY_CLASS:
    """Split the dataset by class values"""
    separated = defaultdict(list)
    for d, l in zip(data, labels):
        separated[l].append(d)

    return dict(separated)


def prior(groups: DATA_BY_CLASS, data: DATA, target_class_value: str) -> float:
    """Calculate probability of each class occurrance"""
    return len(groups[target_class_value]) / len(data)


def summarize(features: DATA) -> list[PARAMETERS]:
    summary = []
    for feature in zip(*features):
        summary.append(
            {
                "stdev": np.std(feature),
                "mean": np.mean(feature),
            }
        )
    return summary


def train(train_data: DATA, train_labels: LABELS) -> SUMMARY_DICT:
    groups = group_by_class(train_data, train_labels)
    summaries = {}
    for target, features in groups.items():
        summaries[target] = {
            "prior": prior(groups, train_data, target),
            "summary": summarize(features),
        }
    return summaries


def normal_distribution(x: float, mean: float, stdev: float) -> float:
    """Gaussian normal distribution"""
    var = np.power(stdev, 2)
    fraction = np.divide(1, np.sqrt(2 * np.pi * var))
    exponent = np.exp(-np.power(x - mean, 2) / (2 * var))
    return fraction * exponent


def calc_likelihood(test_row: ROW, features: FEATURES) -> float:
    likelihood: float = 1
    for i in range(len(features["summary"])):
        feature: float = test_row[i]
        mean = features["summary"][i]["mean"]
        stdev = features["summary"][i]["stdev"]
        normal_dist_prob = normal_distribution(feature, mean, stdev)
        likelihood *= normal_dist_prob

    return likelihood


def calculate_joint_probabilities(test_row: ROW, summaries: SUMMARY_DICT) -> PARAMETERS:
    """Nominator of Bayes theorem"""
    joint_probs = {}
    for target, features in summaries.items():
        likelihood = calc_likelihood(test_row, features)
        prior_probability = features["prior"]
        joint_probs[target] = prior_probability * likelihood
    return joint_probs


def posterior_probabilities(test_row: ROW, summaries: SUMMARY_DICT) -> PARAMETERS:
    posterior_probs = {}
    joint_probabilities = calculate_joint_probabilities(test_row, summaries)
    evidence = sum(joint_probabilities.values())
    for target, joint in joint_probabilities.items():
        posterior_probs[target] = joint / evidence

    return posterior_probs


def test(test_data: DATA, summaries: SUMMARY_DICT) -> LABELS:
    result = []
    for row in test_data:
        posterior_probs = posterior_probabilities(row, summaries)
        max_posterior = max(posterior_probs, key=posterior_probs.get)
        result.append(max_posterior)
    return result


def accuracy(test_labels: LABELS, predicted: LABELS) -> float:
    correct = 0
    for t, p in zip(test_labels, predicted):
        if t == p:
            correct += 1
    return correct / len(test_labels)


def display_results(results: list[float], ratio: float, shuffle: bool) -> None:
    maximum = np.max(results)
    minimum = np.min(results)
    mean = np.mean(results)
    var = np.var(results)
    print(f"Ratio: {ratio}, shuffle {shuffle}")
    print(f"Max: {maximum:.2f}%")
    print(f"Min: {minimum:.2f}%")
    print(f"Mean: {mean:.2f}%")
    print(f"Var: {var:.2f}%")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--data_file")
    args = parser.parse_args()

    with open(args.data_file) as f:
        data = read_data(f.read())

    ratio = 0.8
    shuffle = True
    results = []
    tests = 1
    for i in range(0, tests):
        random.seed(i)
        train_data, train_labels, test_data, test_labels = split_data(
            data.copy(), ratio, shuffle
        )
        # print(f"Train data size: {len(train_data)}, test data size: {len(test_data)}")
        summaries = train(train_data, train_labels)
        predicted = test(test_data, summaries)
        acc = accuracy(test_labels, predicted)
        results.append(acc * 100)

        # print(f"acc: {acc*100:.2f}%")

    display_results(results, ratio, shuffle)
    # scatter_plot(data)
    classes = unique_classes(data, class_column_index=-1)
    # print(f"Dataset has {len(classes)} unique classes, {classes}")
    # confusion_matrix_plot(test_labels, predicted, classes)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
