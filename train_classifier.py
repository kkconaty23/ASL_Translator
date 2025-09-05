import pickle # For saving and loading the model and data to/from files 

from sklearn.metrics import classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np


data_dict = pickle.load(open('./data.pickle', 'rb')) # Load the data

data = np.asarray(data_dict['data']) # Convert data and labels to numpy arrays
labels = np.asarray(data_dict['labels']) # Convert data and labels to numpy arrays

x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, shuffle=True, stratify=labels) # Split the data into training and testing sets

model = RandomForestClassifier(  # Initialize the model
    n_estimators=200,      # more trees → more stable
    max_depth=None,        # allow full growth
    class_weight="balanced",  # handles class imbalance
    random_state=42
)

model.fit(x_train, y_train) # Train the model

y_predict = model.predict(x_test) # Make predictions on the test set

score = accuracy_score(y_predict, y_test) # Calculate the accuracy of the model

print('{}% of samples were classified correctly !'.format(score * 100)) # Print the accuracy

print("Accuracy: {:.2f}%".format(score * 100))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_predict))
print("\nClassification Report:\n", classification_report(y_test, y_predict))


f = open('model.p', 'wb') # Save the trained model
pickle.dump({'model': model}, f) 
f.close()
