# Exercise_1 
# Import matplotlib
import matplotlib.pyplot as plt

# Load the image
data = plt.imread('bricks.png')

# Display the image
plt.imshow(data)
plt.show()

--------------------------------------------------
# Exercise_2 
# Set the red channel in this part of the image to 1
data[0:10,0:10,0] = 1

# Set the green channel in this part of the image to 0
data[0:10,0:10,1] = 0

# Set the blue channel in this part of the image to 0
data[0:10,0:10,2] = 0

# Visualize the result
plt.imshow(data)
plt.show()

--------------------------------------------------
# Exercise_3 
# The number of image categories
n_categories = 3

# The unique values of categories in the data
categories = np.array(["shirt", "dress", "shoe"])

# Initialize ohe_labels as all zeros
ohe_labels = np.zeros((len(labels), n_categories))

# Loop over the labels
for ii in range(len(labels)):
    # Find the location of this label in the categories variable
    jj = np.where(categories==labels[ii])
    # Set the corresponding zero to one
    ohe_labels[ii,jj] = 1

--------------------------------------------------
# Exercise_4 
# Calculate the number of correct predictions
number_correct = (test_labels* predictions).sum()
print(number_correct)

# Calculate the proportion of correct predictions
proportion_correct = number_correct/len(predictions)
print(proportion_correct)

--------------------------------------------------
# Exercise_5 
# Imports components from Keras
from keras.models import Sequential
from keras.layers import Dense

# Initializes a sequential model
model = Sequential()

# First layer
model.add(Dense(10, activation='relu', input_shape=(784,)))

# Second layer
model.add(Dense(10, activation='relu'))

# Output layer
model.add(Dense(3, activation= 'softmax'))

--------------------------------------------------
# Exercise_6 
# Compile the model
model.compile(optimizer='adam', 
           loss='categorical_crossentropy', 
           metrics=['accuracy'])

--------------------------------------------------
# Exercise_7 
# Reshape the data to two-dimensional array
train_data = train_data.reshape(50, 784)

# Fit the model
model.fit(train_data, train_labels, validation_split=0.2, epochs=3)

--------------------------------------------------
# Exercise_8 
# Reshape test data
test_data = test_data.reshape(10, 784)

# Evaluate the model
model.evaluate(test_data, test_labels)

--------------------------------------------------
