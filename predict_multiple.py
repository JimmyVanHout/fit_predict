import numpy
import os
import pandas
import sys
import tensorflow

COLUMNS_TO_REMOVE = ["id", "location", "date", "class_num"]
DEFAULT_INPUT_NUM_CLASSES = 100

def get_data_and_labels(member_ids_group, input_num_classes):
    data_and_labels = {}
    for id, df in member_ids_group:
        if len(df) >= input_num_classes:
            df = df.sort_values(by=["date"])
            df = df.drop(columns=COLUMNS_TO_REMOVE)
            l = df.pop("max_heart_rate")[:input_num_classes].mean()
            d = df[:input_num_classes].to_numpy()
            data_and_labels[id] = (d, l)
    return data_and_labels

def split_data_and_labels(data_and_labels):
    data, labels = [list(x) for x in zip(*data_and_labels)]
    for i in range(len(data)):
        data[i] = data[i].flatten()
    data = numpy.asarray(data)
    labels = numpy.asarray(labels)
    return data, labels

if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.realpath(__file__)))
    if len(sys.argv) < 2:
        raise Exception("Missing data file name")
    data_file_name = sys.argv[1]
    input_num_classes = DEFAULT_INPUT_NUM_CLASSES if len(sys.argv) <= 2 else int(sys.argv[2])
    if data_file_name not in os.listdir():
        raise FileNotFoundError("Data file not found")
    data = pandas.read_csv(data_file_name)
    member_ids_group = data.groupby(["id"])
    data_and_labels = get_data_and_labels(member_ids_group, input_num_classes)
    data_and_labels = list(data_and_labels.values())
    data, expected_labels = split_data_and_labels(data_and_labels)
    expected_labels = list(map(lambda label: round(label), expected_labels))
    model = tensorflow.keras.models.load_model("fcnn_multiple")
    predicted_labels = model.predict(data)
    predicted_labels = [round(label[0]) for label in predicted_labels]
    count = 1
    print("Results:\n")
    for predicted_label, expected_label in [*zip(predicted_labels, expected_labels)]:
        print("Expected {expected_label}, received {predicted_label}. Absolute error: {abs_err}".format(expected_label=expected_label, predicted_label=predicted_label, abs_err=abs(expected_label - predicted_label)))
    abs_errs = numpy.asarray([abs(el - pl) for el, pl in zip(expected_labels, predicted_labels)])
    mean_abs_err = round(abs_errs.mean())
    abs_err_std_dev = round(abs_errs.std())
    print("Mean absolute error: {mean_abs_err}".format(mean_abs_err=mean_abs_err))
    print("Absolute error standard deviation: {abs_err_std_dev}".format(abs_err_std_dev=abs_err_std_dev))
    sys.exit(0)
