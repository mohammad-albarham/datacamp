# Exercise_1 
# Comparison of booleans
True == False 

# Comparison of integers
-5*15 != 75 

# Comparison of strings
"pyscript" == "PyScript"

# Compare a boolean with an integer

True == 1



--------------------------------------------------
# Exercise_2 
# Comparison of integers
x = -3 * 6


# Comparison of strings
y = "test"


# Comparison of booleans
print(x>= -10)
print("test" <= y)
print(True > False)

--------------------------------------------------
# Exercise_3 
# Create arrays
import numpy as np
my_house = np.array([18.0, 20.0, 10.75, 9.50])
your_house = np.array([14.0, 24.0, 14.25, 9.0])

# my_house greater than or equal to 18

print( my_house >= 18)

# my_house less than your_house

print(my_house <your_house )

--------------------------------------------------
# Exercise_4 
# Define variables
my_kitchen = 18.0
your_kitchen = 14.0

# my_kitchen bigger than 10 and smaller than 18?
print(my_kitchen > 10 and my_kitchen < 18)

# my_kitchen smaller than 14 or bigger than 17?
print(my_kitchen < 14 or my_kitchen > 17)

# Double my_kitchen smaller than triple your_kitchen?
print(my_kitchen * 2 < your_kitchen*3)

--------------------------------------------------
# Exercise_5 
# Create arrays
import numpy as np
my_house = np.array([18.0, 20.0, 10.75, 9.50])
your_house = np.array([14.0, 24.0, 14.25, 9.0])

# my_house greater than 18.5 or smaller than 10
print(np.logical_or(my_house>18.5, my_house < 10))

# Both my_house and your_house smaller than 11

print(np.logical_and(my_house < 11,your_house < 11 ))

--------------------------------------------------
# Exercise_6 
# Define variables
room = "kit"
area = 14.0

# if statement for room
if room == "kit" :
    print("looking around in the kitchen.")

# if statement for area

if area > 15:
    print("big place!")

--------------------------------------------------
# Exercise_7 
# Define variables
room = "kit"
area = 14.0

# if-else construct for room
if room == "kit" :
    print("looking around in the kitchen.")
else :
    print("looking around elsewhere.")

# if-else construct for area
if area > 15 :
    print("big place!")
else:
    print("pretty small.")

--------------------------------------------------
# Exercise_8 
# Define variables
room = "bed"
area = 14.0

# if-elif-else construct for room
if room == "kit" :
    print("looking around in the kitchen.")
elif room == "bed":
    print("looking around in the bedroom.")
else :
    print("looking around elsewhere.")

# if-elif-else construct for area
if area > 15 :
    print("big place!")
elif area >10 : 
    print("medium size, nice!")
else:
    print("pretty small.")

--------------------------------------------------
# Exercise_9 
# Import cars data
import pandas as pd
cars = pd.read_csv('cars.csv', index_col = 0)

# Extract drives_right column as Series: dr

dr = cars['drives_right']
# Use dr to subset cars: sel
sel = cars[dr]

# Print sel
print(sel)

--------------------------------------------------
# Exercise_10 
# Import cars data
import pandas as pd
cars = pd.read_csv('cars.csv', index_col = 0)

# Convert code to a one-liner
sel = cars[cars['drives_right']]

# Print sel
print(sel)

--------------------------------------------------
# Exercise_11 
# Import cars data
import pandas as pd
cars = pd.read_csv('cars.csv', index_col = 0)

# Create car_maniac: observations that have a cars_per_cap over 500

cpc = cars['cars_per_cap']
many_cars = cpc > 500 
car_maniac = cars[many_cars]

# Print car_maniac

print(car_maniac)

--------------------------------------------------
# Exercise_12 
# Import cars data
import pandas as pd
cars = pd.read_csv('cars.csv', index_col = 0)

# Import numpy, you'll need this
import numpy as np

# Create medium: observations with cars_per_cap between 100 and 500


medium = cars[np.logical_and(cars['cars_per_cap']>=100, cars['cars_per_cap']<=500)]

# Print medium
print(medium)

--------------------------------------------------
