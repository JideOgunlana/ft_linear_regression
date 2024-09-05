import csv
import sys
import os
import matplotlib.pyplot as plt

class LinearRegression:
    def __init__(self):
        self.theta_0 = 0.0
        self.theta_1 = 0.0
        self.learning_rate = 0.01
        self.data = []
        self.load_data()
        self.normalize_data()

    def load_data(self):
        if len(sys.argv) > 1:
            self.filename = sys.argv[1]
        else:
            self.filename = input("Enter the dataset file name: ")

        if not os.path.isfile(self.filename):
            sys.exit(f"Error: File {self.filename} does not exist.")
        
        if not os.access(self.filename, os.R_OK):
            sys.exit(f"Error: Access denied for {self.filename}.")
        
        with open(self.filename, 'r') as file:
            reader = csv.reader(file)
            self.data = [row for row in reader]
        
        if len(self.data) < 2:
            sys.exit("Error: Data file is empty or has insufficient data.")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python linear_regression.py <dataset.csv> [flags]")
        sys.exit(1)

    lr = LinearRegression()
    lr.train()
    lr.plot_data()
