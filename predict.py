import csv
import sys

def load_parameters(filename="parameters.txt"):
    try:
        with open(filename, 'r') as file:
            theta_0 = float(file.readline().strip())
            theta_1 = float(file.readline().strip())
            mileage_mean = float(file.readline().strip())
            mileage_std = float(file.readline().strip())
            price_mean = float(file.readline().strip())
            price_std = float(file.readline().strip())
        return theta_0, theta_1, mileage_mean, mileage_std, price_mean, price_std
    except Exception as e:
        sys.exit(f"Error loading parameters: {e}")

def predict_price(theta_0, theta_1, normalized_mileage):
    return theta_0 + theta_1 * normalized_mileage

def main():
    theta_0, theta_1, mileage_mean, mileage_std, price_mean, price_std = load_parameters()

    while True:
        try:
            mileage = float(input("Enter mileage: "))
            if mileage < 0:
                print("Mileage cannot be negative. Please try again.")
                continue
            
            # Normalize the mileage
            normalized_mileage = (mileage - mileage_mean) / mileage_std
            # Predict normalized price
            predicted_normalized_price = predict_price(theta_0, theta_1, normalized_mileage)
            # Denormalize the price
            predicted_price = predicted_normalized_price * price_std + price_mean
            print(f"Estimated price for mileage {mileage}: ${predicted_price:.2f}")
            break
        except ValueError:
            print("Invalid input. Please enter a numeric value for mileage.")

if __name__ == "__main__":
    main()
