# Exercise_1 
#1
# Define a Rectangle class
class Rectangle:
    
    def __init__(self, h, w):
        self.h = h
        self.w = w

# Define a Square class
class Square(Rectangle):
    
    def __init__(self, w):
        
        Rectangle.__init__(self, w,w)
    
#2
selected_option = 3
#3
class Rectangle:
    def __init__(self, w,h):
        self.w, self.h = w,h

# Define set_h to set h      
    def set_h(self, h):
        self.h = h
      
# Define set_w to set w          
    def set_w(self, w):
        self.w = w 
      
      
class Square(Rectangle):
    def __init__(self, w):
        self.w, self.h = w, w 

# Define set_h to set w and h
    def set_h(self, h):
        self.h = h
        self.w = h

# Define set_w to set w and h      
    def set_w(self, w):
        self.w = w 
        self.h = w
      
#4
selected_option = 2


--------------------------------------------------
# Exercise_2 
# MODIFY to add class attributes for max number of days and months
class BetterDate:
    
    _MAX_DAYS = 30 
    _MAX_MONTHS = 12
    
    def __init__(self, year, month, day):
      self.year, self.month, self.day = year, month, day
      
    @classmethod
    def from_str(cls, datestr):
        year, month, day = map(int, datestr.split("-"))
        return cls(year, month, day)
    
    # Add _is_valid() checking day and month values
    def _is_valid(self):
        
        return self.month <= BetterDate._MAX_MONTHS and self.day <= BetterDate._MAX_DAYS
  
        
         
bd1 = BetterDate(2020, 4, 30)
print(bd1._is_valid())

bd2 = BetterDate(2020, 6, 45)
print(bd2._is_valid())

--------------------------------------------------
# Exercise_3 
#1
# Create a Customer class

class Customer:
    
    def __init__(self,name, new_bal):
        self.name = name 
        if new_bal <0:
            raise ValueError
        else:
            self._balance = new_bal 
#2
class Customer:
    def __init__(self, name, new_bal):
        self.name = name
        if new_bal < 0:
           raise ValueError("Invalid balance!")
        self._balance = new_bal  
    
    # Add a decorated balance() method returning _balance
    @property
    def balance(self):
        return self._balance
#3
class Customer:
    def __init__(self, name, new_bal):
        self.name = name
        if new_bal < 0:
           raise ValueError("Invalid balance!")
        self._balance = new_bal  

    # Add a decorated balance() method returning _balance        
    @property
    def balance(self):
        return self._balance
     
    # Add a setter balance() method
    @balance.setter
    def balance(self, new_balance):
        # Validate the parameter value
        if new_balance < 0: 
            raise ValueError
        else:
           self._balance = new_balance
        
        # Print "Setter method is called"
        print("Setter method is called")
#4
class Customer:
    def __init__(self, name, new_bal):
        self.name = name
        if new_bal < 0:
           raise ValueError("Invalid balance!")
        self._balance = new_bal  

    # Add a decorated balance() method returning _balance        
    @property
    def balance(self):
        return self._balance

    # Add a setter balance() method
    @balance.setter
    def balance(self, new_bal):
        # Validate the parameter value
        if new_bal < 0:
           raise ValueError("Invalid balance!")
        self._balance = new_bal
        print("Setter method called")

# Create a Customer        
cust = Customer("Belinda Lutz", 2000)

# Assign 3000 to the balance property
cust.balance = 3000

# Print the balance property
print(cust.balance)


--------------------------------------------------
# Exercise_4 
import pandas as pd
from datetime import datetime

# MODIFY the class to turn created_at into a read-only property
class LoggedDF(pd.DataFrame):
    def __init__(self, *args, **kwargs):
        pd.DataFrame.__init__(self, *args, **kwargs)
        self._created_at = datetime.today()
    
    def to_csv(self, *args, **kwargs):
        temp = self.copy()
        temp["created_at"] = self._created_at
        pd.DataFrame.to_csv(temp, *args, **kwargs)   
    
    @property  
    def created_at(self):
        return self._created_at

ldf = LoggedDF({"col1": [1,2], "col2":[3,4]})

# Put into try-except block to catch AtributeError and print a message
try:
    ldf.created_at = '2035-07-13'
except AttributeError:
    print("Could not set attribute")
    

--------------------------------------------------
