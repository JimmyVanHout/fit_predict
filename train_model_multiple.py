import numpy
import os
import pandas
import random
import sqlalchemy
import sys
import tensorflow

DATABASE_FILE_NAME = "data.db"
COLUMNS_TO_REMOVE = ["id", "location", "date", "class_num"]
DATA_SPLITS = [0.6, 0.2, 0.2]
MODEL_FILE_NAME = "fcnn_multiple"
DEFAULT_INPUT_NUM_CLASSES = 100

def read_database():
    sqla_engine = sqlalchemy.create_engine("sqlite:///" + DATABASE_FILE_NAME)
    df = pandas.read_sql("members", sqla_engine)
    return df

def tvt_split(data_and_labels, splits):
    tvt = []
    end_index = 0
    for i in range(len(splits)):
        start_index = min(len(data_and_labels) - 1, end_index)
        end_index = min(len(data_and_labels), start_index + round(len(data_and_labels) * splits[i]))
        d = data_and_labels[start_index:end_index]
        tvt.append(d)
    return tvt

def get_model(data_shape):
    input_processing = tensorflow.keras.Input(shape=data_shape)
    hidden_processing = tensorflow.keras.Sequential([
        tensorflow.keras.layers.Dense(128, activation="relu"),
        tensorflow.keras.layers.Dense(128, activation="relu"),
    ])(input_processing)
    output_processing = tensorflow.keras.layers.Dense(1)(hidden_processing)
    model = tensorflow.keras.Model(inputs=input_processing, outputs=output_processing)
    return model

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

def print_sample(data_and_labels, input_num_classes):
    print("Number of members with at least " + str(input_num_classes) + " classes: " + str(len(data_and_labels)))
    print()
    print("Data (with appropriate columns dropped) and labels:\n")
    for i in range(5):
        member_id = list(data_and_labels.keys())[i]
        print("Class Data for Member ID: " + str(member_id))
        print()
        print("Data (First Five Shown) -> Label:")
        print("{data} -> {label}".format(data=data_and_labels[member_id][0][:5], label=data_and_labels[member_id][1]))
        print("...")
        print()
    print("...")
    print()

if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.realpath(__file__)))
    input_num_classes = DEFAULT_INPUT_NUM_CLASSES if len(sys.argv) <= 1 else int(sys.argv[1])
    if DATABASE_FILE_NAME not in os.listdir():
        raise FileNotFoundError("Database not found")
    print("Reading data from database...")
    data = read_database()
    print("Finished reading data from database")
    print()

    member_ids_group = data.groupby(["id"])
    data_and_labels = get_data_and_labels(member_ids_group, input_num_classes)
    print_sample(data_and_labels, input_num_classes)
    data_and_labels = list(data_and_labels.values())

    # randomize order of data and labels list
    random.shuffle(data_and_labels)

    # get training, validation, and test data and labels
    training, validation, test = tvt_split(data_and_labels, DATA_SPLITS)

    # split into data and labels
    training_data, training_labels = split_data_and_labels(training)
    validation_data, validation_labels = split_data_and_labels(validation)
    test_data, test_labels = split_data_and_labels(test)

    # set up model
    model = get_model(training_data.shape[1:])
    model.compile(optimizer="rmsprop", loss="mse", metrics=["mae"])
    callbacks = [
        tensorflow.keras.callbacks.ModelCheckpoint(filepath=MODEL_FILE_NAME, save_best_only=True, monitor="val_mae"),
        tensorflow.keras.callbacks.EarlyStopping(monitor="val_mae", patience=6)
    ]

    # train model
    history = model.fit(training_data, training_labels, epochs=60, validation_data=(validation_data, validation_labels), callbacks=callbacks)
    model = tensorflow.keras.models.load_model(os.path.join(os.getcwd(), MODEL_FILE_NAME))

    # test model
    test_mean_squared_error, test_mean_absolute_error = model.evaluate(test_data, test_labels)
    print("Mean squared error: {mean_squared_error} Mean absolute error: {mean_absolute_error}".format(mean_squared_error=round(test_mean_squared_error, 2), mean_absolute_error=round(test_mean_absolute_error, 2)))

    sys.exit(0)
