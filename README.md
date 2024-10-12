# ft_linear_regression

## Description

This project implements a simple linear regression model to estimate the price of a car based on its mileage. The model uses the formula:

```
estimated_price(mileage) = θ0 + (θ1 * mileage)
```

## Files

It includes two Python scripts and a data.csv file

- `linear_regression.py` — Trains the model using a dataset.
- `predict.py` — Predicts car prices based on mileage using the trained model.
- `data.csv` — The dataset used for training. It should be a comma-separated file with two columns:

```km,price
240000,3650
139800,3800
...
```

## Dependencies

This project requires **Python 3.x** and the following libraries:

- `csv` (standard library)
- `sys` (standard library)
- `os` (standard library)
- `matplotlib` (for plotting)

To install `matplotlib`, run:

```bash
pip install matplotlib
```


## Usage

1. **Train the model**

   Make sure your dataset is in a CSV format with two columns: mileage and price.

   ```bash
   python3 linear_regression.py
   ```


2. **Predict Car Price**
   
   Upon completion of model training, use the predict.py to make a car price prediction by entering the command below in the terminal and following the prompt: 
   
   ```bash
   python3 predict.py
   ```