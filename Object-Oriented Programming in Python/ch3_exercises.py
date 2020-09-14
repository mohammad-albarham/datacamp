# Exercise_1 
class BankAccount:
   # MODIFY to initialize a number attribute
    def __init__(self, number, balance=0):
        self.balance = balance
        self.number = number
      
    def withdraw(self, amount):
        self.balance -= amount 
    
    # Define __eq__ that returns True if the number attributes are equal 
    def __eq__(self,other):
        return self.number == other.number

# Create accounts and compare them       
acct1 = BankAccount(123, 1000)
acct2 = BankAccount(123, 1000)
acct3 = BankAccount(456, 1000)
print(acct1 == acct2)
print(acct1 == acct3)
    

--------------------------------------------------
# Exercise_2 
class BankAccount:
    def __init__(self, number, balance=0):
        self.number, self.balance = number, balance
      
    def withdraw(self, amount):
        self.balance -= amount 

    # MODIFY to add a check for the type()
    def __eq__(self, other):
        return (self.number == other.number) and \
        (type (self) == type(other) )

acct = BankAccount(873555333)
pn = Phone(873555333)
print(acct == pn)

--------------------------------------------------
# Exercise_3 
#1
selected_option = 2


--------------------------------------------------
# Exercise_4 
#1
selected_option = 2


--------------------------------------------------
# Exercise_5 
#1
class Employee:
    def __init__(self, name, salary=30000):
        self.name, self.salary = name, salary
            
    # Add the __str__() method
    def __str__(self):
        output = """
        Employee name:{name}
        Employee salary:{salary}
        """.format(name = self.name, salary = self.salary)
        return output

emp1 = Employee("Amar Howard", 30000)
print(emp1)
emp2 = Employee("Carolyn Ramirez", 35000)
print(emp2)
#2
class Employee:
    def __init__(self, name, salary=30000):
        self.name, self.salary = name, salary
      

    def __str__(self):
        s = "Employee name: {name}\nEmployee salary: {salary}".format(name=self.name, salary=self.salary)      
        return s
      
    # Add the __repr__method  
    def __repr__(self):
        
        return "Employee(\"{name}\", {salary})".format(name = self.name, salary = self.salary)

emp1 = Employee("Amar Howard", 30000)
print(repr(emp1))
emp2 = Employee("Carolyn Ramirez", 35000)
print(repr(emp2))


--------------------------------------------------
# Exercise_6 
# MODIFY the function to catch exceptions
def invert_at_index(x, ind):
    
    try : 
       return 1/x[ind]
    
    except ZeroDivisionError: 
        print("Cannot divide by zero!")
    except IndexError:
            print("Index out of range!")

    
a = [5,6,0,7]

# Works okay
print(invert_at_index(a, 1))

# Potential ZeroDivisionError
print(invert_at_index(a, 2))

# Potential IndexError
print(invert_at_index(a, 5))

--------------------------------------------------
# Exercise_7 
#1
# Define SalaryError inherited from ValueError
class SalaryError(ValueError):
    pass

# Define BonusError inherited from SalaryError
class BonusError(SalaryError):
    pass



#2
class SalaryError(ValueError): pass
class BonusError(SalaryError): pass

class Employee:
  MIN_SALARY = 30000
  MAX_RAISE = 5000

  def __init__(self, name, salary = 30000):
    self.name = name
    
    # If salary is too low
    if self.salary < Employee.MIN_SALARY:
      # Raise a SalaryError exception
      raise SalaryError("Salary is too low!")

    self.salary = salary
      
#3
class SalaryError(ValueError): pass
class BonusError(SalaryError): pass

class Employee:
  MIN_SALARY = 30000
  MAX_BONUS = 5000

  def __init__(self, name, salary = 30000):
    self.name = name    
    if salary < Employee.MIN_SALARY:
      raise SalaryError("Salary is too low!")      
    self.salary = salary
    
  # Rewrite using exceptions  
  def give_bonus(self, amount):
    if amount > Employee.MAX_BONUS:
       raise BonusError("The bonus amount is too high!")  
        
    elif self.salary + amount <  Employee.MIN_SALARY:
       raise SalaryError("The salary after bonus is too low!")
      
    else:  
      self.salary += amount


--------------------------------------------------
# Exercise_8 
#1
selected_option = 1
#2
selected_option = 3


--------------------------------------------------
