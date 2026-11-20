import typing

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score, classification_report, accuracy_score

import keras

sns.set_style('whitegrid')


def load_data(red_path: str, white_path: str) -> pd.DataFrame:
    """
    Loading and combining datasets of red and white wine.

    Note: Don't forget to add the 'type' characteristic
    """
    red_df = pd.read_csv(red_path, sep=';')
    red_df['type'] = 'red'
    white_df = pd.read_csv(white_path, sep=';')
    white_df['type'] = 'white'
    data = pd.concat([red_df, white_df], ignore_index=True)

    return data


def remove_outliers(df: pd.DataFrame, features: pd.Index) -> pd.DataFrame:
    """Removing outliers from the dataset using the IQR method."""
    df_clean = df.copy()

    for col in features:
        Q1 = df_clean[col].quantile(0.25)
        Q3 = df_clean[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        df_clean = df_clean[(df_clean[col] >= lower_bound) & (df_clean[col] <= upper_bound)]

    return df_clean


def scale_features(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame
) -> tuple[np.ndarray, np.ndarray]:
    """Standardization of features."""
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled


def encode_categorical_variables(data: pd.DataFrame) -> pd.DataFrame:
    """
    Encoding categorical variables.
    
    This time you need to use LabelEncoder
    """
    data_encoded = data.copy()

    le = LabelEncoder()
    data_encoded['type'] = le.fit_transform(data_encoded['type'])

    return data_encoded


def prepare_regression_data(
        data: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Preparing data for the regression task."""
    X = data.drop(['quality'], axis=1)
    y = data['quality']

    return X, y


def build_regression_model(input_dim: int) -> keras.Model:
    """
    Creating a neural network regressor.
    
    Use `keras.layers.Input` and `keras.layers.Dense`.
    """
    model = keras.Sequential()

    model.add(keras.layers.Input(shape=(input_dim,)))
    model.add(keras.layers.Dense(64, activation='relu'))
    model.add(keras.layers.Dense(32, activation='relu'))
    model.add(keras.layers.Dense(1))
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])

    return model


def train_regression_model(model: keras.Model,
                           X_train: pd.DataFrame,
                           y_train: pd.Series,
                           epochs: int = 50,
                           batch_size: int = 32) -> keras.callbacks.History:
    """Training the neural network regressor."""
    history = model.fit(
        X_train,
        y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=0.2,
        verbose=0
    )

    return history


def evaluate_regression_model(
        model: keras.Model, X_test: pd.DataFrame,
        y_test: pd.Series) -> dict[str, float | np.ndarray]:
    """Evaluating the neural network regressor."""
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    results = {'MSE': mse, 'R2': r2}

    return results


def prepare_classification_data(
        data: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """
    Preparing data for the classification task.
    
    Use keras.utils.to_categorical to convert labels
    into a one-hot encoded vector.
    """
    X = data.drop(['quality'], axis=1)
    y_categories = pd.cut(data['quality'], bins=[0, 5, 6, 10], labels=[0, 1, 2])
    y = keras.utils.to_categorical(y_categories, num_classes=3)

    return X, y


def build_classification_model(input_dim: int,
                               num_classes: int) -> keras.Model:
    """
    Creating a neural network classifier.
    
    Use `keras.layers.Input` and `keras.layers.Dense`.
    """
    model = keras.Sequential()

    model.add(keras.layers.Input(shape=(input_dim,)))
    model.add(keras.layers.Dense(64, activation='relu'))
    model.add(keras.layers.Dense(32, activation='relu'))
    model.add(keras.layers.Dense(num_classes, activation='softmax'))
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    return model


def train_classification_model(model,
                               X_train,
                               y_train,
                               epochs=50,
                               batch_size=32) -> keras.callbacks.History:
    """Training the neural network classifier."""
    history = model.fit(
        X_train,
        y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=0.2,
        verbose=0
    )

    return history


def evaluate_classification_model(
        model: keras.Model, X_test: pd.DataFrame,
        y_test: pd.Series) -> dict[str, float | str | dict]:
    """Evaluating the neural network classifier."""
    y_pred_probs = model.predict(X_test)
    y_pred = np.argmax(y_pred_probs, axis=1)
    y_true = np.argmax(y_test, axis=1)
    report = classification_report(y_true, y_pred, zero_division=0)
    accuracy = accuracy_score(y_true, y_pred)
    results = {'report': report, 'accuracy': accuracy}
    return results


def main():
    # Loading data
    data = load_data('winequality-red.csv', 'winequality-white.csv')

    # Removing outliers
    features = data.columns[:-2]  # Without 'quality' and 'type'
    data = remove_outliers(data, features)

    # Encoding categorical variables
    data = encode_categorical_variables(data)

    # Regression task
    X_reg, y_reg = prepare_regression_data(data)
    X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(
        X_reg, y_reg, test_size=0.2, random_state=42)

    # Feature scaling
    X_train_scaled_reg, X_test_scaled_reg = scale_features(
        X_train_reg, X_test_reg)

    reg_model = build_regression_model(X_train_scaled_reg.shape[1])
    train_regression_model(reg_model, X_train_scaled_reg, y_train_reg)
    regression_results = evaluate_regression_model(reg_model,
                                                   X_test_scaled_reg,
                                                   y_test_reg)
    print("Regression results:")
    print(f"MSE - {regression_results['MSE']:.4f}")
    print(f"R2 - {regression_results['R2']:.4f}")

    # Classification task
    X_clf, y_clf = prepare_classification_data(data)

    X_train_clf, X_test_clf, y_train_clf, y_test_clf = train_test_split(
        X_clf, y_clf, test_size=0.2, random_state=42)

    # Feature scaling
    X_train_scaled_clf, X_test_scaled_clf = scale_features(
        X_train_clf, X_test_clf)

    clf_model = build_classification_model(X_train_scaled_clf.shape[1],
                                           y_clf.shape[1])
    train_classification_model(clf_model, X_train_scaled_clf, y_train_clf)

    classification_results = evaluate_classification_model(
        clf_model, X_test_scaled_clf, y_test_clf)
    print("\nClassification results:")
    print(f"Accuracy: {classification_results['accuracy']:.4f}")
    print("Classification report:")
    print(classification_results['report'])


if __name__ == "__main__":
    main()
