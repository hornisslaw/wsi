"""
Introduction to Artificial Intelligence, Exercise 4:
Classification and regression.
Author: Robert Kaczmarski 293377
"""
from __future__ import annotations

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from collections import Counter


class Node:
    def __init__(self, answer: str = "", is_leaf: bool = False) -> None:
        self.attribute: str = ""
        self.attribute_value = ""
        self.answer = answer
        self.is_leaf = is_leaf
        self.children = []

    def search_tree(self, input: pd.Series(str)) -> str:
        if self.is_leaf:
            return self.answer

        for c in self.children:
            if str(input[self.attribute]) == c.attribute_value:
                return c.search_tree(input)
        if self.answer == "":
            return "ignore"

        return self.answer

    def __str__(self):
        representation = (
            f"Answer: {self.answer}\n"
            f"Split Attribute: {self.attribute}\n"
            f"Leaf: {self.is_leaf}\n"
        )
        return representation


def id3(examples: pd.DataFrame, target_attribute: str, attributes: list[str]) -> Node:
    # empty examples TODO: change to parents node subset
    # also this condition is never executed in for the tested set
    if len(examples) == 0:
        print("EMPTY EXAMPLES")
        return Node(examples[target_attribute].mode().iloc[0], True)
    # all the same class values
    elif len(examples[target_attribute].unique()) <= 1:
        return Node(examples[target_attribute].mode().iloc[0], True)
    # empty attributes list -> return most common value of target attribute in examples:
    elif len(attributes) == 0:
        print("EMPTY ATTRIBURE LIST")
        return Node(examples[target_attribute].mode().iloc[0], True)

    root = Node()

    best_attribute = ""
    max_info_gain = -np.inf

    for attribute in attributes:
        current_gain = info_gain(examples, attribute, target_attribute)

        if current_gain > max_info_gain:
            max_info_gain = current_gain
            best_attribute = attribute

    attribute_values = examples[best_attribute].unique()
    splitted = split_to_subbsets(examples, best_attribute, attribute_values)
    new_attributes = list(attributes)
    new_attributes.remove(best_attribute)

    root.attribute = best_attribute

    for a_v, subset in splitted.items():
        new_node = id3(subset, target_attribute, new_attributes)
        new_node.attribute_value = a_v
        root.children.append(new_node)

    return root


def entropy(attribute_column: pd.Series(str)) -> float:
    data = Counter(attribute_column)
    entropy_results = []

    values_sum = sum(v for v in data.values())
    for v in data.values():
        proportion = v / values_sum
        entropy_results.append(-proportion * np.log2(proportion))

    total_entropy = np.sum(entropy_results)

    return total_entropy


def split_to_subbsets(
    data: pd.DataFrame, attribute: str, attribute_values: np.ndarray
) -> dict[str, pd.DataFrame]:
    splitted = {}
    grouped = data.groupby(attribute)

    for value in attribute_values:
        splitted[value] = grouped.get_group(value)

    return splitted


def info_gain(
    examples: pd.DataFrame, feature_attribute: str, target_attribute: str
) -> float:
    total_entropy = entropy(examples[target_attribute])
    data = Counter(examples[feature_attribute])

    splitted = split_to_subbsets(examples, feature_attribute, data.keys())

    weighted_entropy = 0

    values_sum = sum(v for v in data.values())
    for k, v in splitted.items():
        proportion = data[k] / values_sum
        weighted_entropy += proportion * entropy(v[target_attribute])

    return total_entropy - weighted_entropy


def predict(node: Node, test_data: pd.DataFrame) -> list[str]:
    prediction = []
    for i in range(0, len(test_data)):
        prediction.append(node.search_tree(test_data.iloc[i]))

    return prediction


def check_prediction(
    answers_for_test_data: list[str], predictions: list[str], class_instances: list[str]
) -> tuple(dict, dict):
    correct_predictions = {}
    wrong_predictions = {}
    wrong_predictions["Wrong_pairs"] = []

    for c in class_instances:
        correct_predictions[c] = 0
        wrong_predictions[c] = 0

    for a, p in zip(answers_for_test_data, predictions):
        if a == p:
            correct_predictions[a] += 1
        else:
            wrong_predictions["Wrong_pairs"].append((a, p))
            wrong_predictions[a] += 1

    return correct_predictions, wrong_predictions


def kfold_split(data: pd.DataFrame, k: int):
    sets = []
    x, r = divmod(len(data), k)

    for i in range(k):
        sets.append(data[i * x + min(i, r) : (i + 1) * x + min(i + 1, r)])

    return sets


def analize(data: pd.DataFrame, attributes: list[str], target: str) -> None:
    target = "class"
    attributes.pop()
    for attr in attributes:
        values = data[attr].unique()
        print(attr)
        subsets = split_to_subbsets(data, attr, values)
        print(type(data), type(attr), type(values))

        for key, subset in subsets.items():
            c = Counter(subset[target])
            print(f"key: {key},  {c}")
            plt.title(attr + ":" + key)
            plt.bar(c.keys(), c.values())
            plt.show()


def display_results(correct: dict[str, int], wrong: dict[str, int]) -> None:
    percentage_correct = {}
    for c, w in zip(correct.items(), wrong.items()):
        if c[1] + w[1] == 0:
            percentage = -1
        else:
            percentage = 100 * c[1] / (c[1] + w[1])
            percentage_correct[c[0]] = percentage
        print(f"{c[0]}, {percentage:.2f}%")
    return percentage_correct


def main() -> int:
    filename = "nursery.data"
    attributes = [
        "parents",
        "has_nurs",
        "form",
        "children",
        "housing",
        "finance",
        "social",
        "health",
        "class",
    ]
    class_instances = ["not_recom", "recommend", "very_recom", "priority", "spec_prior"]
    df = pd.read_csv(filename, names=attributes)
    input = pd.read_csv("test.data", names=attributes)
    target_attribute = attributes.pop()

    """ Analize """
    # analize(df, attributes, target_attribute)

    """ Split """
    # train_data = df.sample(frac=0.8, random_state=1)
    # train_data = train_data.sort_index()
    # print(train_data)
    # test_data = df.drop(train_data.index)
    # for i in range(1, 10):
    #     fraction = i/10
    #     train_data = df.sample(frac=fraction, random_state=1)
    #     test_data = df.drop(train_data.index)
    #     answers_for_test_data = test_data[target_attribute]

    #     node = id3(train_data, target_attribute, attributes)

    #     prediction = predict(node, test_data)
    #     correct, wrong = check_prediction(
    #                                     answers_for_test_data,
    #                                     prediction,
    #                                     class_instances
    #                     )
    #     print(f"Results for training: {fraction}")
    #     print(f"Correct: {correct}")
    #     print(f"Wrong: {wrong}")
    #     display_results(correct, wrong)

    """ Sort """
    # train_data = df.sample(frac=0.9, random_state=2)
    # # train_data = train_data.sort_index()
    # test_data = df.drop(train_data.index)
    # answers_for_test_data = test_data[target_attribute]
    # node = id3(train_data, target_attribute, attributes)
    # prediction = predict(node, test_data)
    # correct, wrong = check_prediction(
    #                                     answers_for_test_data,
    #                                     prediction,
    #                                     class_instances
    #                     )
    # print(f"Correct: {correct}")
    # print(f"Wrong: {wrong}")
    # display_results(correct, wrong)

    """ K-fold cross validation """
    # shullfe rows
    # df = df.sample(frac=1, random_state=10)
    # for k in [3,5,7,10,20]:
    #     results = {}
    #     for c in class_instances:
    #         results[c] = []

    #     sets = kfold_split(df, k)
    #     for i, s in enumerate(sets):
    #         print(f"ITERTION {i}")
    #         temp_test_data = s
    #         temp_train_data = pd.concat(sets).drop(temp_test_data.index)
    #         answers_for_test_data = temp_test_data[target_attribute]
    #         temp_node = id3(temp_train_data, target_attribute, attributes)
    #         prediction = predict(temp_node, temp_test_data)
    #         correct, wrong = check_prediction(
    #                                     answers_for_test_data,
    #                                     prediction,
    #                                     class_instances
    #                     )
    #         for key, value in display_results(correct, wrong).items():
    #             results[key].append(value)

    #     print(f"Results for {k}-fold cross validation")
    #     for key, value in results.items():
    #         minimum = np.min(value)
    #         maximum = np.max(value)
    #         mean = np.mean(value)
    #         var = np.var(value)
    #         print(f"Class {key}")
    #         print(f"min {minimum:.2f}, max {maximum:.2f}, mean {mean:.2f}, var {var:.2f}")
    #     print(results)

    """ Confusion matrix """
    # train_data = df.sample(frac=0.9, random_state=1)
    # test_data = df.drop(train_data.index)
    # answers_for_test_data = test_data[target_attribute]
    # node = id3(train_data, target_attribute, attributes)
    # prediction = predict(node, test_data)
    # correct, wrong = check_prediction(
    #                                     answers_for_test_data,
    #                                     prediction,
    #                                     class_instances
    #                     )
    # print(f"Correct: {correct}")
    # print(f"Wrong: {wrong}")
    # for w in wrong["Wrong_pairs"]:
    #     print(w)
    # display_results(correct, wrong)

    """ Tree example """
    train_data = input.sample(frac=0.9, random_state=1)
    test_data = input.drop(train_data.index)
    answers_for_test_data = test_data[target_attribute]
    node = id3(train_data, target_attribute, attributes)
    prediction = predict(node, test_data)
    correct, wrong = check_prediction(
        answers_for_test_data, prediction, class_instances
    )
    print(f"Correct: {correct}")
    print(f"Wrong: {wrong}")
    for w in wrong["Wrong_pairs"]:
        print(w)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())