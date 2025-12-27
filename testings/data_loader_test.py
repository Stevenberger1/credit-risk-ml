
from src.data import get_train_test_data
X_train, X_test, y_train, y_test = get_train_test_data()

print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)
print("y_train distribution:\n", y_train.value_counts(normalize=True))
print("y_test distribution:\n", y_test.value_counts(normalize=True))