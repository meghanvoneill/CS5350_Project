from pprint import pprint
from math import copysign
import random
import copy


def main():

    weights, bias = training()

    evaluate(weights, bias, '-averaged-perceptron-0-01-102020')

    return


def training():

    learning_rates = [1, .1, .01, .001]
    bias = random.uniform(-.01, .01)

    training_file = '../project_data/data/tfidf/tfidf.train.libsvm'
    training_set = read_file(training_file)
    training_epochs = 15

    test_file = '../project_data/data/tfidf/tfidf.train.libsvm'
    test_set = read_file(test_file)

    # Hold out segment of training set for testing.
    random.shuffle(training_set['examples'])

    # Perceptron
    weight_vector, new_bias, predictions, updates, accuracies = averaged_perceptron(training_set['examples'], training_epochs,
                                                                           training_set['# of features'] + 1, bias,
                                                                           learning_rates[2])
    predictions_on_test = predict_all(weight_vector, new_bias, test_set['examples'])
    training_acc = accuracy(predictions, training_set['examples'])
    test_acc = accuracy(predictions_on_test, test_set['examples'])

    print('Perceptron\n')
    print('Training Accuracy: ' + str(training_acc))
    print('Test Accuracy: ' + str(test_acc))
    print('Updates: ' + str(updates))
    print('\n')

    return weight_vector, new_bias


def evaluate(weights, bias, file_name_addendum):

    eval_file = '../project_data/data/tfidf/tfidf.eval.anon.libsvm'
    ids_file = '../project_data/data/eval.ids'
    predictions_file_name = 'tfidf-eval-predictions' + file_name_addendum + '.csv'
    eval_set = read_eval_file(eval_file, ids_file)

    predict_eval_write_file(weights, bias, eval_set['examples'], predictions_file_name)

    return


def gather_stats(data):

    


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
                new_example[idx] = float(val)
                data_set['features'].add(idx)

            data_set['examples'].append(new_example)

    data_set['# of features'] = max(data_set['features'])

    return data_set


def read_eval_file(eval_file_name, ids_file_name):
    # Build dictionary data set from input file.
    data_set = {'examples': [], 'features': set(), '# of features': 0}

    # Read in CSV data formatted in sparse vector as: <label> <index1>:value1> <index2>:value2> ...
    with open(eval_file_name, 'r') as f:
        with open(ids_file_name, 'r') as i:
            ids = i.readlines()
            index = 0
            for line in f.readlines():
                new_line = line.split()
                new_line = new_line[0:-1]
                new_example = {'label': int(new_line[0]), 'id': int(ids[index])}
                for data in new_line[1:]:
                    arr = data.split(':')
                    idx = int(arr[0])
                    val = arr[1]
                    new_example[idx] = float(val)
                    data_set['features'].add(idx)
                index += 1
                data_set['examples'].append(new_example)

    data_set['# of features'] = max(data_set['features'])

    return data_set


def perceptron(data, epochs, features, bias=0, learning_rate=.1):

    weight_vector = weight_init(-.01, .01, features)
    updates = 0
    accuracies = {}

    for epoch in range(epochs):
        print('Epoch: ' + str(epoch))
        random.shuffle(data)
        predictions = []
        for example in data:
            # Predict y' = sgn(w_t^T x_i)
            prediction = predict(weight_vector, example, bias)
            predictions.append(prediction)
            # If y' != y_i, update w_{t+1} = w_t + r(y_i x_i)
            #print('prediction: ' + str(prediction))
            #print('label: ' + str(example['label']))
            # If the prediction does not match the label, update the weights and bias.
            if prediction != example['label']:
                updates += 1
                #pprint(example)
                #print('w_v before: ' + str(weight_vector[0:5]))
                #print('label = ' + str(example['label']))
                weight_vector, bias = update(weight_vector, bias, example, learning_rate, example['label'] if example['label'] == 1 else -1)
                #print('w_v after: ' + str(weight_vector[0:5]))

        acc = accuracy(predictions, data)
        print(acc)
        accuracies[epoch + 1] = acc

    return weight_vector, bias, predictions, updates, accuracies


def averaged_perceptron(data, epochs, features, bias=0, learning_rate=.1):

    weight_vector = weight_init(-.011, .011, features)
    averaged_weight_vector = weight_init(-.011, .011, features)
    bias_of_a = random.uniform(-.011, .011)
    updates = 0
    accuracies = {}

    for epoch in range(epochs):
        print('Epoch: ' + str(epoch))
        random.shuffle(data)
        predictions = []
        for example in data:
            # Predict y' = sgn(w_t^T x_i)
            prediction = predict(weight_vector, example, bias)
            predictions.append(prediction)
            # If y' != y_i, update w_{t+1} = w_t + r(y_i x_i)
            # print('prediction: ' + str(prediction))
            # print('label: ' + str(example['label']))
            # If the prediction does not match the label, update the weights and bias.
            if prediction != example['label']:
                updates += 1
                # pprint(example)
                # print('w_v before: ' + str(weight_vector[0:5]))
                # print('label = ' + str(example['label']))
                weight_vector, bias = update(weight_vector, bias, example, learning_rate,
                                             example['label'] if example['label'] == 1 else -1)
                # print('w_v after: ' + str(weight_vector[0:5]))
            # Update the averaged weight vector and bias.
            averaged_weight_vector = add_weights(weight_vector, averaged_weight_vector)
            bias_of_a = (bias_of_a + bias) / 2

        acc = accuracy(predictions, data)
        print(acc)
        accuracies[epoch + 1] = acc

    return weight_vector, bias, predictions, updates, accuracies


def add_weights(weight_vector_1, weight_vector_2):

    new_weight_vector = []

    for i in range(len(weight_vector_1)):
        w1 = weight_vector_1[i]
        w2 = weight_vector_2[i]
        new_val = (w1 + w2) / 2
        new_weight_vector.append(new_val)

    return new_weight_vector


def weight_init(bottom, top, length):
    weights = []

    for i in range(length):
        rand_number = random.uniform(bottom, top)
        weights.append(rand_number)

    return weights


def predict(weights, example, bias=0):
    prediction = 0

    # Evaluate each feature to find the cross product of weights^T and example.
    for i in range(len(weights)):
        if i in example.keys():
            prediction += weights[i] * example[i]

    return int(copysign(1, prediction + bias)) if int(copysign(1, prediction + bias)) == 1 else 0


def predict_all(weights, bias, examples):
    predictions = []

    for example in examples:
        # Predict y' = sgn(w_t^T x_i)
        prediction = predict(weights, example, bias)
        predictions.append(prediction)

    return predictions


def predict_eval_write_file(weights, bias, examples, file_name):

    with open(file_name, 'w') as f:
        f.write('example_id,label\n')
        for example in examples:
            # Predict y' = sgn(w_t^T x_i)
            id = example['id']
            prediction = predict(weights, example, bias)
            f.write(str(id) + ',' + str(prediction) + '\n')

    return


def update(weights, bias, x, rate, sign):
    new_weights = copy.deepcopy(weights)
    new_bias = bias + rate * sign

    #print('sign: ' + str(sign))
    #print('rate: ' + str(rate))

    for key in x.keys():
        if key != 'label':
            #print('old weight: ' + str(weights[key]))
            #print('x[key}: ' + str(x[key]))
            new_weights[key] = weights[key] + (sign * rate * x[key])
            #print('new weight: ' + str(new_weights[key]))

    return new_weights, new_bias


def accuracy(predictions, examples):
    summation = 0.0

    for index in range(len(predictions)):
        if predictions[index] == examples[index]['label']:
            summation += 1

    return summation / len(predictions)


if __name__ == '__main__':
    main()
