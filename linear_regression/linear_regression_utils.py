import pathlib

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def load_data():
    paypal_data_file = pathlib.Path('data', 'paypal_total_payment_volume.csv')

    paypal_data = pd.read_csv(paypal_data_file, sep=';').rename({
        'PayPal: total payment volume Q1 2014- Q1 2019': 'quarter',
        'Net total payment volume in billion U.S. dollars': 'payment_volume'
    }, axis=1)

    x = paypal_data['quarter'].values
    y = paypal_data['payment_volume'].values

    return x, y


def train_test_split(x, y, test_size):
    first_test_index = round(len(x) * (1 - test_size))
    x_train, y_train = x[:first_test_index], y[:first_test_index]
    x_test, y_test = x[first_test_index:], y[first_test_index:]
    return x_train, y_train, x_test, y_test


def plot_data(x, y):
    plt.subplots(figsize=(10, 8))
    plt.scatter(x, y, s=50, marker='+', label='Alle Daten')
    plt.title('PayPal: Gesamtes Bezahlvolumen Q1 2014 - Q1 2019')
    plt.xlabel('Quartal')
    plt.xticks(range(len(x)))
    plt.ylabel('Bezahlvolumen (Millarden USD)')
    plt.legend()
    plt.show()


def train_model(x_train, y_train, polynomial_degree):
    max_polynomial_degree = len(x_train) - 1
    if (polynomial_degree > max_polynomial_degree):
        polynomial_degree = max_polynomial_degree
        print(
            "Der Polynomgrad ist zu hoch für die verfügbare Anzahl von "
            f"{len(x_train)} Trainingsdatenpunkten. Der Grad wird reduziert "
            f"auf den maximal möglichen Grad {max_polynomial_degree}."
        )
    return np.poly1d(np.polyfit(x_train, y_train, polynomial_degree))


def evaluate_model(
    model, x_train, y_train, x_test, y_test, model_compare=None
):
    # re-combine data to full set
    x = np.concatenate([x_train, x_test])
    y = np.concatenate([y_train, y_test])
    # compute intermedate points in order to achieve smooth plot
    x_fine = np.arange(0, x.max(), 0.1)
    # compute prediction for full data
    y_pred = model(x_fine)
    if model_compare:
        y_pred_compare = model_compare(x_fine)
    # start plotting
    plt.subplots(figsize=(10, 8))
    # plot train and test data
    plt.scatter(x_train, y_train, s=50, marker='+', label='Trainingsdaten')
    plt.scatter(
        x_test, y_test,
        s=50, color='red', marker='X', label='Testdaten'
    )
    # plot model curve
    plt.plot(x_fine, y_pred, alpha=0.5, label='Modell')
    if model_compare:
        plt.plot(x_fine, y_pred_compare, alpha=0.5, label='Vergleichsmodell')
    # tune and display plot
    plt.title('PayPal: Gesamtes Bezahlvolumen Q1 2014 - Q1 2019')
    plt.xlabel('Quartal')
    plt.xticks(range(len(x)))
    plt.ylim((y.min() - y.std(), y.max() + y.std()))
    plt.ylabel('Bezahlvolumen (Millarden USD)')
    plt.legend()
    plt.show()


# functions for validation notebook


def train_val_test_split(x, y, test_size, val_size):
    first_test_index = round(len(x) * (1 - test_size))
    first_val_index = round(len(x) * (1 - test_size - val_size))
    x_train, y_train = x[:first_val_index], y[:first_val_index]
    x_val, y_val = \
        x[first_val_index:first_test_index], \
        y[first_val_index:first_test_index]
    x_test, y_test = x[first_test_index:], y[first_test_index:]
    return x_train, y_train, x_val, y_val, x_test, y_test


def mae(y, y_pred):
    return np.mean(np.abs(y - y_pred))


def evaluate_model_val(
    model, x_train, y_train, x_val, y_val, x_test, y_test, model_compare=None
):
    # re-combine data to full set
    x = np.concatenate([x_train, x_val, x_test])
    y = np.concatenate([y_train, y_val, y_test])
    # compute intermedate points in order to achieve smooth plot
    x_fine = np.arange(0, x.max(), 0.1)
    # compute prediction for full data
    y_pred = model(x_fine)
    if model_compare:
        y_pred_compare = model_compare(x_fine)
    # start plotting
    plt.subplots(figsize=(10, 8))
    # plot train, val, test data
    plt.scatter(x_train, y_train, s=50, marker='+', label='Trainingsdaten')
    plt.scatter(
        x_val, y_val,
        s=50, color='green', marker='o', label='Validierungsdaten'
    )
    plt.scatter(
        x_test, y_test,
        s=50, color='red', marker='X', label='Testdaten'
    )
    # plot model curve
    plt.plot(x_fine, y_pred, alpha=0.5, label='Modell')
    if model_compare:
        plt.plot(x_fine, y_pred_compare, alpha=0.5, label='Vergleichsmodell')
    # tune and display plot
    plt.title('PayPal: Gesamtes Bezahlvolumen Q1 2014 - Q1 2019')
    plt.xlabel('Quartal')
    plt.xticks(range(len(x)))
    plt.ylim((y.min() - y.std(), y.max() + y.std()))
    plt.ylabel('Bezahlvolumen (Millarden USD)')
    plt.legend()
    plt.show()
