# Implicit Type Conversion
# Integer is implicitly converted to float
integer_num = 10
float_num = 5.5
result = integer_num + float_num # 10 + 5.5

print(result)        # Output: 15.5
print(type(result))  # Output: <class 'float'>

# Explicit Type Conversion
# Converting string to integer
num_str = "100"
num_int = int(num_str)
print(num_int * 2) # Output: 200

# Converting integer to string
age = 25
message = "I am " + str(age) + " years old."
print(message) # Output: I am 25 years old.

# Converting float to integer (truncates the decimal part)
price = 9.99
whole_price = int(price)
print(whole_price) # Output: 9