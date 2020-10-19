from sklearn.datasets.samples_generator import make_regression, make_classification
from sklearn.model_selection import train_test_split


def dataloader(task, n_samples, n_features, noise, test_size, random_state):
    if task == 'classification':
        X, y = make_classification(n_samples=n_samples, n_features=n_features, random_state=random_state)
    elif task == 'regression':
        X, y = make_regression(n_samples=n_samples, n_features=n_features, noise=noise, random_state=random_state)

    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    return x_train, x_test, y_train, y_test