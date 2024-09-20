import sys

# Load parameters from a file or set defaults
def load_parameters(filename="parameters.txt"):
    try:
        with open(filename, 'r') as file:
            theta_0 = float(file.readline().strip())
            theta_1 = float(file.readline().strip())
            mileage_mean = float(file.readline().strip())
            mileage_std = float(file.readline().strip())
            price_mean = float(file.readline().strip())
            price_std = float(file.readline().strip())
    except FileNotFoundError:
        theta_0 = 0.0
        theta_1 = 0.0
        mileage_mean = 0.0
        mileage_std = 1.0   # To avoid division by zero during normalization
        price_mean = 0.0
        price_std = 1.0
    except Exception as e:
        sys.exit(f"Error loading parameters: {e}")
    
    return theta_0, theta_1, mileage_mean, mileage_std, price_mean, price_std

# Prediction function: theta_0 + (theta_1 * normalized_mileage)
def predict_price(theta_0, theta_1, normalized_mileage):
    return theta_0 + theta_1 * normalized_mileage

# Main program
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

            predicted_price = max(predicted_price, 0)

            print(f"Estimated price: {predicted_price:.2f}")
            break
        except ValueError:
            print("Invalid input. Please enter a numeric value for mileage.")

if __name__ == "__main__":
    main()
