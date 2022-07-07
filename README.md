# Fit Predict

## About

This set of programs yields a regression model that is able to predict the maximum heart rate of a member of a group workout gym from an artificially generated data set. This is done by:

1. Creating an artificial yet realistic set of records of classes of all gym members.

1. Populating this data into an SQLite database.

1. Extracting the data from the database.

1. Training one of two different fully connected neural networks on the data.

1. Evaluating the model on the provided test data set.

After the model has completed training, it can be used to make predictions.

## Installation

Install the program from [GitHub](https://github.com/JimmyVanHout/fit_predict):

```
git clone https://github.com/JimmyVanHout/fit_predict.git
```

Install the dependencies:

```
pip3 install numpy
pip3 install pandas
pip3 install sqlalchemy
pip3 install tensorflow
```

To train the model on a GPU on your local machine, you must install [CUDA](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html). You *may* need to also install [cuDNN](https://docs.nvidia.com/deeplearning/cudnn/install-guide/index.html).

Alternatively, you can run all of the programs, including model training, in [Google Colab](https://colab.research.google.com/drive/1DHR1RDCctDW5vJsScnUjwE8q0quqA2Mn#scrollTo=TfaEK6zZGvig).

## Creating the Database

To create the database, run:

```
python3 create_database.py [<num_members>]
```

This will create a database with 40,000 members by default, or `num_members` members if specified. The data set has been designed to be fairly, though certainly not completely, realistic, in order to demonstrate the predictive power of the trained model. Each record contains data for a single member at a single class and has the following form:

```
<id>, <location>, <date>, <age>, <max_heart_rate>, <medication>, <class_num>
```

Important features of the data set include:

* The number of classes a member takes is taken from a normal distribution with a mean of 300 and a standard deviation of 100.

* The minimum age of a member is 20 and the maximum age is 70.

* Each member has a probability of taking a heart rate-lowering medication based on their age according to an exponential function:

    ```
    f(x) = (10 ** ((x - 70) / 50)) / 2
    ```

    where `x` is the member's age. Thus, `f(20) = 0.05` and `f(70) = 0.5` (i.e. a 20 year old has a 5% probability of taking the medication while a 70 year old has a 50% probability). Once a member's medication status is assigned, it will not change, although the training programs *are* designed to handle the event that a member takes the medication on some days and not others.

* The maximum heart rate of a member is taken from a normal distribution with a standard deviation of 5. The mean results from the following equation, which is presented in [Age-Predicted Maximal Heart Rate Revisited](https://pubmed.ncbi.nlm.nih.gov/11153730/) (Tanaka, Monahan & Seals, 2001):

    ```
    max_hr = 208 - 0.7 * age
    ```

    If a member is taking the heart rate-lowering medication, this value is decreased by 10 before being used as the mean of the normal distribution from which the actual max heart rate of the member will be taken.

* The maximum heart rate a member generates during a given workout is generated from a normal distribution with a standard deviation of 5 and a mean equal to a randomly chosen number in the range [0, 10] subtracted from the maximum heart rate the member can achieve.

## Training a Model

Once the database has been populated, you can choose from two models to train.

The first model can be trained by running:

```
python3 train_model_simple.py
```

This model takes as input each individual workout record, which has the following form after unnecessary columns are dropped:

```
<age>, <medication>
```

> The number of classes the member has taken at the time they took this class, inclusive of this class, is dropped because it does not strongly correlate with the max heart rate due to the way the data set has been generated. Similarly, any other column that would not have a relation to and thus would not correlate strongly with the max heart rate variable was dropped.

The model is saved in the directory `fcnn_single`.

The second model is trained by running:

```
python3 train_model_multiple.py [<input_num_classes>]
```

where `<input_num_classes>` is the input to the model. More specifically, the model takes as input the set of the first `<input_num_classes>` workouts, sorted by date, of a member. Each workout record has the same form as shown previously. If `<input_num_classes>` is not specified, the default of 100 is used. If a member does not have at least `<input_num_classes>` of workouts, their data cannot be used for training.

The model is saved in the directory `fcnn_multiple`.

At the end of training either model, the model will be evaluated on the test data set and the mean squared error and mean average error will be displayed.

## Using a Model to Predict

To predict using the first model, run:

```
python3 predict_single.py <data_file_name>
```

where `<data_file_name>` is the name of the data file you want to predict on, formatted as a CSV file in the following way:

```
id, location, date, age, max_heart_rate, medication, class_num
<id_1>, <location_1>, <date_1>, <age_1>, <max_heart_rate_1>, <medication_1>, <class_num_1>
<id_2>, <location_2>, <date_2>, <age_2>, <max_heart_rate_2>, <medication_2>, <class_num_2>
...
```

To predict using the second model, run:

```
python3 predict_multiple.py <data_file_name> <input_num_classes>
```

where `<data_file_name>` is the name of the data file you want to predict on, formatted as before, and `<input_num_classes>` is the number of each member's workouts, beginning from the earliest, that will be used as input into the model. Just as during training, if a member does not have at least `<input_num_classes>` of workouts, their data cannot be used during model prediction.

A GPU is not required for model prediction.

## Accuracy

When trained on the artificially generated data set, the models have approximately the following error values:

Model Name | Mean Squared Error | Mean Absolute Error
--- | --- | ---
`fcnn_single` | 49.14 | 5.61
`fcnn_multiple` | 31.38 | 4.49

When used for prediction on the sample test data set in `test.csv`, the models have approximately the following error values:

Model Name | Mean Absolute Error | Absolute Error Standard Deviation
--- | --- | ---
`fcnn_single` | 5 | 4
`fcnn_multiple` | 4 | 4

## Support

If you need support, you can file an issue on [GitHub](https://github.com/JimmyVanHout/fit_predict/issues).
