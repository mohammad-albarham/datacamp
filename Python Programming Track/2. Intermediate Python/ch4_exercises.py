# Exercise_1 
# Initialize offset
offset = 8 

# Code the while loop
while offset !=0: 
    print("correcting...")
    offset = offset -1 
    print(offset)

--------------------------------------------------
# Exercise_2 
# Initialize offset
offset = -6

# Code the while loop
while offset != 0 :
    print("correcting...")
    if offset > 0 :
      offset = offset - 1 
    else : 
      offset = offset + 1     
    print(offset)

--------------------------------------------------
# Exercise_3 
# areas list
areas = [11.25, 18.0, 20.0, 10.75, 9.50]

# Code the for loop
for i in areas: 
    print(i)

--------------------------------------------------
# Exercise_4 
# areas list
areas = [11.25, 18.0, 20.0, 10.75, 9.50]

# Change for loop to use enumerate() and update print()
for i,y in enumerate(areas) :
    print("room" + str(i) + ':' + str(y))

--------------------------------------------------
# Exercise_5 
# areas list
areas = [11.25, 18.0, 20.0, 10.75, 9.50]

# Code the for loop
for index, area in enumerate(areas) :
    print("room " + str(index+1) + ": " + str(area))

--------------------------------------------------
# Exercise_6 
# house list of lists
house = [["hallway", 11.25], 
         ["kitchen", 18.0], 
         ["living room", 20.0], 
         ["bedroom", 10.75], 
         ["bathroom", 9.50]]
         
# Build a for loop from scratch

for i in house: 
    print("the " + str(i[0]) + ' is ' + str(i[1]) + ' sqm')

--------------------------------------------------
# Exercise_7 
# Definition of dictionary
europe = {'spain':'madrid', 'france':'paris', 'germany':'berlin',
          'norway':'oslo', 'italy':'rome', 'poland':'warsaw', 'austria':'vienna' }
          
# Iterate over europe

for k,v in europe.items():
    print("the capital of " + k + " is " + v)

--------------------------------------------------
# Exercise_8 
# Import numpy as np
import numpy as np

# For loop over np_height

for i in np_height : 
    print(str(i) +" inches")

# For loop over np_baseball

for i in np.nditer(np_baseball): 
    print(i)

--------------------------------------------------
# Exercise_9 
# Import cars data
import pandas as pd
cars = pd.read_csv('cars.csv', index_col = 0)

# Iterate over rows of cars
for lab, row in cars.iterrows(): 
    print(lab)
    print(row)

--------------------------------------------------
# Exercise_10 
# Import cars data
import pandas as pd
cars = pd.read_csv('cars.csv', index_col = 0)
print(cars)
# Adapt for loop
for lab, row in cars.iterrows() :
    print(lab+": "+ str (row['cars_per_cap']) )
    
    

--------------------------------------------------
# Exercise_11 
# Import cars data
import pandas as pd
cars = pd.read_csv('cars.csv', index_col = 0)

# Code for loop that adds COUNTRY column
for lab, row in cars.iterrows(): 
    cars.loc[lab, "COUNTRY"] = row["country"].upper()


# Print cars
print(cars)

--------------------------------------------------
# Exercise_12 
# Import cars data
import pandas as pd
cars = pd.read_csv('cars.csv', index_col = 0)
print(cars)
# Use .apply(str.upper)
cars["COUNTRY"] = cars["country"].apply(str.upper)

--------------------------------------------------
