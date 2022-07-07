import datetime
import os
import pandas
import random
import sqlite3
import sys

LOCATIONS = ["Washington, D.C.", "Boca Raton, FL", "Los Angeles, CA", "San Diego, CA", "College Park, MD"]
START_YEAR = 2015
START_MONTH = 1
START_DAY = 1
END_YEAR = 2022
END_MONTH = 7
END_DAY = 2
POP_MEAN_NUM_CLASSES = 300 # model population mean number of classes taken
POP_STD_DEV_NUM_CLASSES = 100 # model population standard deviation of number of classes taken
MEDICATION = 1
NO_MEDICATION = 0
NUM_MEMBERS = 40000
MIN_AGE = 20
MAX_AGE = 70
MEDICATION_DECREASE = -10 # decrease to an individual's max heart rate caused by medication
B = 10 ** (1 / 50) # b in p med calc f(x) = a * b ** x
A = 0.5 / (B ** 70) # a in p med calc f(x) = a * b ** x
DATABASE_FILE_NAME = "data.db"

def get_dates():
    start_date = datetime.datetime(START_YEAR, START_MONTH, START_DAY)
    end_date = datetime.datetime(END_YEAR, END_MONTH, END_DAY)
    td = end_date - start_date
    dates = [start_date + datetime.timedelta(days=duration) for duration in range(td.days)]
    return dates

def get_max_hr(age, taking_medication):
    max_hr = 208 - 0.7 * age
    if taking_medication:
        max_hr += MEDICATION_DECREASE
    max_hr = round(random.gauss(max_hr, 5))
    return max_hr

def get_class_max_hr(max_hr, num_classes):
    class_max_hr = round(random.gauss(max_hr - random.randrange(11), 5))
    class_max_hr = max(0, min(max_hr, class_max_hr))
    return class_max_hr

def get_p_med(age):
    p_med = A * (B ** age)
    return p_med

def get_member_data(id):
    member_data = []
    age = random.randrange(MIN_AGE, MAX_AGE + 1)
    p_med = get_p_med(age)
    medication = random.choices([NO_MEDICATION, MEDICATION], weights=[1 - p_med, p_med])[0]
    num_classes = max(0, round(random.gauss(POP_MEAN_NUM_CLASSES, POP_STD_DEV_NUM_CLASSES)))
    class_dates = sorted(random.sample(dates, num_classes))
    max_hr = get_max_hr(age, medication)
    for i in range(num_classes):
        class_max_hr = get_class_max_hr(max_hr, num_classes)
        class_data = {
            "id": id,
            "location": random.choice(locations),
            "date": class_dates[i],
            "age": age,
            "max_heart_rate": class_max_hr,
            "medication": medication,
            "class_num": i + 1,
        }
        member_data.append(class_data)
    return member_data

def get_members_data(num_members):
    members_data = []
    id = 1
    for i in range(num_members):
        member_data = get_member_data(id)
        members_data += member_data
        id += 1
    return members_data

def write_to_database(df, database_file_name):
    connection = sqlite3.connect(database_file_name)
    cursor = connection.cursor()
    df.to_sql("members", connection, index=False)
    connection.close()

if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.realpath(__name__)))
    num_members = NUM_MEMBERS if len(sys.argv) <= 1 else int(sys.argv[1])
    if DATABASE_FILE_NAME in os.listdir():
        raise FileExistsError("Database already exists. Will not write to it.")
    locations = LOCATIONS
    dates = get_dates()
    members_data = get_members_data(num_members)
    members_data = pandas.DataFrame(data=members_data)
    members_data = members_data.sort_values(by=["date"])
    print("Members Data:\n")
    print(members_data)
    print()
    medication_groups = members_data.drop(columns=["id"]).groupby(["medication"], dropna=False)
    medication_groups_means = medication_groups.mean()
    print("Mean Age and Mean Max Heart Rate (bpm) by Medication:\n")
    print(medication_groups_means)
    print()
    correlation_matrix = members_data.drop(columns=["id"]).corr()
    print("Correlation Matrix:\n")
    print(correlation_matrix)
    print()
    print("Writing to database...")
    write_to_database(members_data, DATABASE_FILE_NAME)
    print("Finished writing to database")
    sys.exit(0)
