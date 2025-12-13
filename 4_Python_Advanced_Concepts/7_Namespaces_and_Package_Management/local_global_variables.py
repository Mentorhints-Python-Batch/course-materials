global_var = "I'm global"

def demonstrate_scope():
    local_var = "I'm local"
    print(global_var)  # Can access global variables
    print(local_var)   # Can access local variables

    # # Modifying global variables requires 'global' keyword
    # global global_var
    # global_var = "Modified global"

demonstrate_scope()
print(global_var)  # Modified global
# print(local_var)  # Error: local_var is not defined
