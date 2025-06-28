# one time script for creating dummy data

import csv

import os

# Get the project root directory
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

# Path to the database directory
filename = os.path.join(project_root, "database", "student_data.csv")

headers = ["student_id", "student_fname", "student_lname", "basics", "algebra", "geometry", "miscellaneous", "modelling", "probability", "statistics"]

data_entry = {
    "student_id": "ywa3324",
    "student_fname": "Gina",
    "student_lname": "Wang",
    "basics": 1,
    "algebra": 1,
    "geometry": 0,
    "miscellaneous": 0,
    "modelling": 0,
    "probability": 0,
    "statistics": 0
}

with open(filename, mode="w", newline="") as file:
    writer = csv.DictWriter(file, fieldnames=headers)
    writer.writeheader() 
    writer.writerow(data_entry)