import os
import pandas
import sqlalchemy
import sys
import tensorflow

DATABASE_FILE_NAME = "data.db"
COLUMNS_TO_REMOVE = ["id", "location", "date", "class_num"]
DATA_SPLITS = [0.6, 0.2, 0.2]
MODEL_FILE_NAME = "fcnn_single"

def read_database():
    sqla_engine = sqlalchemy.create_engine("sqlite:///" + DATABASE_FILE_NAME)
    df = pandas.read_sql("members", sqla_engine)
    return df

def tvt_split(data, data_splits):
    tvt = []
    end_index = 0
    for i in range(len(data_splits)):
        start_index = min(len(data) - 1, end_index)
        end_index = min(len(data), start_index + round(len(data) * data_splits[i]))
        d = data.iloc[start_index:end_index]
        tvt.append(d)
    return tvt

def get_model():
    input_processing = tensorflow.keras.Input(shape=(2,))
    hidden_processing = tensorflow.keras.Sequential([
        tensorflow.keras.layers.Dense(64, activation="relu"),
        tensorflow.keras.layers.Dense(64, activation="relu"),
    ])(input_processing)
    output_processing = tensorflow.keras.layers.Dense(1)(hidden_processing)
    model = tensorflow.keras.Model(inputs=input_processing, outputs=output_processing)
    return model

if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.realpath(__file__)))
    if DATABASE_FILE_NAME not in os.listdir():
        raise FileNotFoundError("Database not found")
    print("Reading data from database...")
    data = read_database()
    print("Finished reading data from database")
    print()

    # drop columns that won't be used for training
    data = data.drop(columns=COLUMNS_TO_REMOVE)
    print("Data with appropriate columns dropped:\n")
    print(data)
    print()

    # randomize order of rows in data frame
    data = data.sample(frac=1)

    # get training, validation, and test data and labels
    training, validation, test = tvt_split(data, DATA_SPLITS)

    # split into data and labels
    training_labels = training.pop("max_heart_rate")
    training_data = training
    validation_labels = validation.pop("max_heart_rate")
    validation_data = validation
    test_labels = test.pop("max_heart_rate")
    test_data = test

    # set up model
    model = get_model()
    model.compile(optimizer="rmsprop", loss="mse", metrics=["mae"])
    callbacks = [
        tensorflow.keras.callbacks.ModelCheckpoint(filepath=MODEL_FILE_NAME, save_best_only=True, monitor="val_mae"),
        tensorflow.keras.callbacks.EarlyStopping(monitor="val_mae", patience=4)
    ]

    # train model
    history = model.fit(training_data, training_labels, epochs=40, validation_data=(validation_data, validation_labels), callbacks=callbacks)
    model = tensorflow.keras.models.load_model(os.path.join(os.getcwd(), MODEL_FILE_NAME))

    # test model
    test_mean_squared_error, test_mean_absolute_error = model.evaluate(test_data, test_labels)
    print("Mean squared error: {mean_squared_error} Mean absolute error: {mean_absolute_error}".format(mean_squared_error=round(test_mean_squared_error, 2), mean_absolute_error=round(test_mean_absolute_error, 2)))

    sys.exit(0)
