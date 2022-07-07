import numpy
import os
import pandas
import sys
import tensorflow

COLUMNS_TO_REMOVE = ["id", "location", "date", "class_num"]

if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.realpath(__file__)))
    if len(sys.argv) < 2:
        raise Exception("Missing data file name")
    data_file_name = sys.argv[1]
    if data_file_name not in os.listdir():
        raise FileNotFoundError("Data file not found")
    data = pandas.read_csv(data_file_name)
    data = data.drop(columns=COLUMNS_TO_REMOVE)
    expected_labels = data.pop("max_heart_rate")
    model = tensorflow.keras.models.load_model("fcnn_single")
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
