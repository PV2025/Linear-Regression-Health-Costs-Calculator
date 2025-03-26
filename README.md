# Linear-Regression-Health-Costs-Calculator

import pandas as pd

# Convert categorical columns
dataset = pd.get_dummies(dataset, columns=['sex', 'smoker', 'region'], drop_first=True)


from sklearn.model_selection import train_test_split

# Split into features and labels
labels = dataset.pop('expenses')

train_dataset, test_dataset, train_labels, test_labels = train_test_split(
    dataset, labels, test_size=0.2, random_state=42)


from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
train_dataset = scaler.fit_transform(train_dataset)
test_dataset = scaler.transform(test_dataset)


import tensorflow as tf

model = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(1)
])

model.compile(optimizer='adam', loss='mae', metrics=['mae'])

# Train the model
model.fit(train_dataset, train_labels, epochs=100, validation_split=0.2, verbose=0)


loss, mae = model.evaluate(test_dataset, test_labels)
print("Mean Absolute Error:", mae)


mae < 3500
