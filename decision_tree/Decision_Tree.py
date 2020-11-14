from pprint import pprint
import math
import random


def main():

    training_file = '../project_data/data/bag-of-words/bow.train.libsvm'
    training_set = read_file(training_file)

    test_file = '../project_data/data/bag-of-words/bow.train.libsvm'
    test_set = read_file(test_file)

    # experiment_1(training_set, test_set)
    train_acc, test_acc = experiment_2(training_set, test_set)
    print('train_acc: ' + str(train_acc))
    print('test_acc: ' + str(test_acc))

    return


def experiment_1(data_set, test_set):

    # Build Decision Tree using training files.
    tree = build_decision_tree(data_set)
    train_accuracy = accuracy(tree, data_set['examples'])
    test_accuracy = accuracy(tree, test_set['examples'])
    return train_accuracy, test_accuracy


def experiment_2(data_set, test_set):

    # Build Decision Tree using training files and depth-limit.
    tree = build_decision_tree(data_set, 50)
    #pprint(tree)

    train_accuracy = accuracy(tree, data_set['examples'])
    test_accuracy = accuracy(tree, test_set['examples'])
    return train_accuracy, test_accuracy


def read_file(file_name):
    # Build dictionary data set from input file.
    data_set = {'examples': [], 'features': set(), '# of features': 0}

    # Read in CSV data formatted in sparse vector as: <label> <index1>:value1> <index2>:value2> ...
    with open(file_name, 'r') as f:
        for line in f.readlines():
            new_line = line.split()
            new_line = new_line[0:-1]
            new_example = {'label': int(new_line[0])}
            for data in new_line[1:]:
                arr = data.split(':')
                idx = int(arr[0])
                val = arr[1]
                new_example[idx] = int(val)
                data_set['features'].add(idx)

            data_set['examples'].append(new_example)

    data_set['# of features'] = max(data_set['features'])

    return data_set


def build_decision_tree(data_set, depth_limit=None):

    if depth_limit:
        tree = {'depth': 0, 'root': ID3_depth_limit(data_set, data_set['features'], 0, depth_limit)}
    else:
        tree = {'depth': 0, 'root': ID3_depth_limit(data_set, data_set['features'], 0)}

    return tree


def ID3_depth_limit(tree, features, depth, depth_limit=None):
    labels = all_known_feature_values(tree, 'label')

    if len(labels) == 1:
        return {'value': tree['examples'][0]['label'], 'branches': {}}

    feat = determine_best_attribute(tree, features)
    root = {'value': feat, 'branches': {}}
    values_of_features = all_known_feature_values(tree, feat)

    for v in values_of_features:
        # Add a new tree branch for attribute A taking value v.
        # Let S_v be the subset of examples in S with A=v.
        sub_set = subset(tree, feat, v)

        # If S_v is empty:
        if len(sub_set['examples']) < 1:
            # Add leaf node with the common value of Label in S
            label = common_label(tree)
            root['branches'][v] = {'value': label, 'branches': {}}
        # Otherwise, below this branch, add the subtree ID3(S_v, Attributes - {A})
        else:
            # Account for depth limit, if it exists.
            if depth_limit:
                if depth >= depth_limit - 1:
                    # Add leaf node with the common value of Label in S
                    label = common_label(sub_set)
                    root['branches'][v] = {'value': label, 'branches': {}}
                else:
                    root['branches'][v] = ID3_depth_limit(sub_set, sub_set['features'], depth_limit, depth+1)
            else:
                root['branches'][v] = ID3_depth_limit(sub_set, sub_set['features'], depth + 1)
    root['depth'] = depth

    return root


def traverse_tree(tree, example):

    current_node = tree

    # If this is a leaf node, return this leaf's value.
    if len(current_node['branches'].keys()) < 1:
        return current_node['value']

    # Otherwise, traverse the branches as possible features of the example.
    for feature in current_node['branches'].keys():
        # If the example has this feature, take this branch.
        if feature in example.keys():
            return traverse_tree(current_node['branches'][feature], example)

    # If no feature was returned, choose the label most common.
    label = most_common_leaf_label(current_node)
    return label


def most_common_leaf_label(node):

    labels_counts = {}
    max_label_count = 0
    max_label = None

    pprint(node)

    # Check this branches children for the most common leaf value.
    for branch in node['branches'].keys():
        if len(node['branches'][branch]['branches'].keys()) < 1:
            label = node['branches'][branch]['value']
        else:
            label = most_common_leaf_label(node['branches'][branch])
        if label in labels_counts.keys():
            labels_counts[label] += 1
        else:
            labels_counts[label] = 1
        if labels_counts[label] > max_label_count:
            max_label_count = labels_counts[label]
            max_label = label

    return max_label


def determine_best_attribute(data_set, features):

    best_info_gain = 0
    best_gain_att = None

    for a in features:
        gain = information_gain(data_set, a)
        if gain > best_info_gain:
            best_info_gain = gain
            best_gain_att = a

    return best_gain_att


# The information gain of the given attribute is the expected reduction
#   in entropy caused by partitioning on this attribute.
def information_gain(data_set, attribute):
    gain = entropy(data_set)
    summation = 0

    values_of_att = [0,1]
    for value in values_of_att:
        subset_of_value = subset(data_set, attribute, value)
        if len(subset_of_value['examples']) > 0:
            entropy_of_subset = entropy(subset_of_value)
            summation += (len(subset_of_value['examples']) / len(data_set['examples'])) * entropy_of_subset

    return gain - summation


# Calculates the entropy of a given data set. The ratio of positive labels is defined as the number of labels greater
#   than 0, over the total number of labels, and the ratio of negative labels is the number of remaining labels, over
#   the total number of labels.
def entropy(data_set):

    positive_examples = 0
    negative_examples = 0
    total_examples = 0

    for example in data_set['examples']:
        if example['label'] > 0:
            positive_examples += 1
        else:
            negative_examples += 1
        total_examples += 1

    proportion_of_p = positive_examples / total_examples
    proportion_of_n = negative_examples / total_examples

    if positive_examples == 0:
        return (-1) * proportion_of_n * math.log2(proportion_of_n)

    if negative_examples == 0:
        return (-1) * proportion_of_p * math.log2(proportion_of_p)

    calculated_entropy = (-1) * proportion_of_p * math.log2(proportion_of_p) - proportion_of_n * math.log2(proportion_of_n)

    return calculated_entropy


# Given a set and a particular attribute, returns a set of all values the attribute
#   can take.
def all_known_feature_values(data_set, feature):

    known_values = set()

    for i in range(len(data_set['examples'])):
        if feature in data_set['examples'][i].keys():
            known_values.add(data_set['examples'][i][feature])

    if feature is not 'label' and (len(known_values) == 1 and 1 in known_values):
        known_values.add(0)

    return known_values


# Given a data set, feature, and value, produces a subset of examples with that attribute having
#   that particular value. Allows for sparse vectors and value = 0.
def subset(data_set, feature, value):

    sub_set = {'examples': [], 'features': set()}

    for example in data_set['examples']:
        if feature in example.keys():
            if example[feature] == value:
                sub_set['examples'].append(example)
        else:
            if value == 0:
                sub_set['examples'].append(example)

    sub_set['features'] = data_set['features'].copy()
    sub_set['features'].remove(feature)

    return sub_set


def common_label(data_set):

    labels_counts = {}
    max_label_count = 0
    max_label = None

    for example in data_set['examples']:
        label = example['label']
        if label in labels_counts.keys():
            labels_counts[label] += 1
        else:
            labels_counts[label] = 1
        if labels_counts[label] > max_label_count:
            max_label_count = labels_counts[label]
            max_label = label

    return max_label


def accuracy(tree, test_examples):

    correct = .0
    size = len(test_examples)

    for example in test_examples:
        test_label = traverse_tree(tree['root'], example)
        label = example['label']
        if test_label == label:
            correct += 1

    return correct / size


if __name__ == '__main__':
    main()
