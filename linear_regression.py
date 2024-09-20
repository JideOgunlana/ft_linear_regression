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
        # Check if 'data.csv' is present in the current directory
        default_filename = "data.csv"
        
        # Check if the file exists and is accessible
        if not os.path.isfile(default_filename) or not os.access(default_filename, os.R_OK):
            sys.exit(f"Error: '{default_filename}' not found or cannot be read. Please ensure 'data.csv' is in the same directory as linear_regression.py.")
        
        self.filename = default_filename
        print(f"Loading data from {self.filename}")

        # Read the data from the CSV file
        with open(self.filename, 'r') as file:
            reader = csv.reader(file)
            self.data = [row for row in reader]
        
        # Check if the file has enough data
        if len(self.data) < 2:
            sys.exit("Error: Data file is empty or has insufficient data.")

    def normalize_data(self):
        mileages = [float(row[0]) for row in self.data[1:]]
        prices = [float(row[1]) for row in self.data[1:]]
        
        self.mileage_mean = sum(mileages) / len(mileages)
        self.mileage_std = (sum((x - self.mileage_mean) ** 2 for x in mileages) / len(mileages)) ** 0.5
        
        self.price_mean = sum(prices) / len(prices)
        self.price_std = (sum((x - self.price_mean) ** 2 for x in prices) / len(prices)) ** 0.5
        
        for row in self.data[1:]:
            row[0] = (float(row[0]) - self.mileage_mean) / self.mileage_std
            row[1] = (float(row[1]) - self.price_mean) / self.price_std

    def compute_mean_squared_error(self):
        total_error = 0.0
        for row in self.data[1:]:
            mileage = float(row[0])
            price = float(row[1])
            prediction = self.theta_0 + self.theta_1 * mileage
            error = prediction - price
            total_error += error ** 2
        return total_error / len(self.data[1:])

    def update_parameters(self):
        gradient_0 = 0.0
        gradient_1 = 0.0
        m = len(self.data) - 1  # Number of data points

        for row in self.data[1:]:  # Skip header
            mileage = float(row[0])
            price = float(row[1])
            prediction = self.theta_0 + self.theta_1 * mileage
            error = prediction - price
            gradient_0 += error
            gradient_1 += error * mileage

        self.theta_0 -= (self.learning_rate / m) * gradient_0
        self.theta_1 -= (self.learning_rate / m) * gradient_1

    def train(self, epochs=1000):
        prev_mse = float('inf')
        for _ in range(epochs):
            self.update_parameters()
            cur_mse = self.compute_mean_squared_error()
            if abs(prev_mse - cur_mse) < 1e-6:
                break
            prev_mse = cur_mse
        self.save_parameters()
        print(f"Training complete. Theta_0: {self.theta_0}, Theta_1: {self.theta_1}")

    def save_parameters(self):
        with open("parameters.txt", "w") as file:
            file.write(f"{self.theta_0}\n")
            file.write(f"{self.theta_1}\n")
            file.write(f"{self.mileage_mean}\n")
            file.write(f"{self.mileage_std}\n")
            file.write(f"{self.price_mean}\n")
            file.write(f"{self.price_std}\n")

    def plot_data(self):
        miles = [float(row[0]) * self.mileage_std + self.mileage_mean for row in self.data[1:]]
        prices = [float(row[1]) * self.price_std + self.price_mean for row in self.data[1:]]

        plt.scatter(miles, prices, color='red', label='Data')
        plt.plot(miles, [self.theta_0 * ((mile - self.mileage_mean) / self.mileage_std) + self.theta_1 for mile in miles], color='blue', label='Fit')
        plt.xlabel('Mileage')
        plt.ylabel('Price')
        plt.title('Linear Regression')
        plt.legend()
        plt.show()

if __name__ == "__main__":
    lr = LinearRegression()
    lr.train()
    lr.plot_data()
