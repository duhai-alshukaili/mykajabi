import tensorflow as tf
# from tensorflow import keras 
import keras
import numpy as np

# 1. Create a simple model (Sequential API)
model = keras.Sequential([
    keras.layers.Dense(64, activation='relu', input_shape=(10,)),
    keras.layers.Dense(1)
])




# 2. Compile the model
model.compile(optimizer='adam', loss='mse')

# 3. Generate dummy data (100 samples, 10 features)
x_train = np.random.random((100, 10))
y_train = np.random.random((100, 1))

# 4. Train the model
model.fit(x_train, y_train, epochs=1, batch_size=32)

# 5. Run inference (run model on new data)
dummy_input = np.random.random((1, 10))
prediction = model.predict(dummy_input)
print("Prediction:", prediction)

