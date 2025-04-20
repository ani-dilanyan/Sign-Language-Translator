import pickle
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load preprocessed hand landmark data
with open('./data.pickle', 'rb') as f:
    data_dict = pickle.load(f)

data_list = data_dict['data']
labels = np.asarray(data_dict['labels'])

# Determine the maximum length of feature vectors
max_length = max(len(features) for features in data_list)

# Ensure all feature vectors are the same length by padding with zeros
data = np.array([features + [0] * (max_length - len(features)) for features in data_list])

# Split dataset into training (80%) and testing (20%) sets with stratified sampling
x_train, x_test, y_train, y_test = train_test_split(
    data, labels, test_size=0.2, shuffle=True, stratify=labels
)

# Initialize and train a Random Forest classifier
model = RandomForestClassifier()
model.fit(x_train, y_train)

# Make predictions on the test set
y_predict = model.predict(x_test)

# Calculate accuracy score
accuracy = accuracy_score(y_test, y_predict)

# Print classification accuracy
print(f'{accuracy * 100:.2f}% of samples were classified correctly!')

# Save the trained model to a pickle file
with open('model.p', 'wb') as f:
    pickle.dump({'model': model}, f)

print("Model saved as 'model.p'.")
