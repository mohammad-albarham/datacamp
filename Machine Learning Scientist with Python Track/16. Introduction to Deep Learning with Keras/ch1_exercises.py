# Exercise_1 
# Import the Sequential model and Dense layer
from keras.models import Sequential
from keras.layers import Dense

# Create a Sequential model
model = Sequential()

# Add an input layer and a hidden layer with 10 neurons
model.add(Dense(10, input_shape=(2,), activation="relu"))

# Add a 1-neuron output layer
model.add(Dense(1))

# Summarise your model
model.summary()

--------------------------------------------------
# Exercise_2 
#1
# Instantiate a new Sequential model
model = Sequential()

# Add a Dense layer with five neurons and three inputs
model.add(Dense(5, input_shape=(3,), activation="relu"))

# Add a final Dense layer with one neuron and no activation
model.add(Dense(1))

# Summarize your model
model.summary()
#2
selected_option = 2


--------------------------------------------------
# Exercise_3 
from keras.models import Sequential
from keras.layers import Dense

# Instantiate a Sequential model
model = Sequential()

# Build the input and hidden layer
model.add(Dense(3,input_shape = (2,), activation = 'relu'))

# Add the ouput layer
model.add(Dense(1))

--------------------------------------------------
# Exercise_4 
# Instantiate a Sequential model
model = Sequential()

# Add a Dense layer with 50 neurons and an input of 1 neuron
model.add(Dense(50, input_shape=(1,), activation='relu'))

# Add two Dense layers with 50 neurons and relu activation
model.add(Dense(50 , activation='relu'))
model.add(Dense(50 , activation='relu'))

# End your model with a Dense layer and no activation
model.add(Dense(1))

--------------------------------------------------
# Exercise_5 
# Compile your model
model.compile(optimizer= 'adam', loss = 'mse')

print("Training started..., this can take a while:")

# Fit your model on your data for 30 epochs
model.fit( time_steps,y_positions, epochs = 30)

# Evaluate your model 
print("Final lost value:",model.evaluate(time_steps, y_positions))

--------------------------------------------------
# Exercise_6 
#1
# Predict the twenty minutes orbit
twenty_min_orbit = model.predict(np.arange(-10, 11))

# Plot the twenty minute orbit 
plot_orbit(twenty_min_orbit)
#2
# Predict the eighty minute orbit
eighty_min_orbit = model.predict(np.arange(-40, 41))

# Plot the eighty minute orbit 
plot_orbit(eighty_min_orbit)


--------------------------------------------------
