class MyClass:
    # Class namespace
    class_var = "I'm a class variable"

    def __init__(self):
        # Instance/Method namespace
        self.instance_var = "I'm an instance variable"

    def show_namespaces(self):
        print("Class namespace:", [attr for attr in dir(MyClass) if not attr.startswith('_')])
        print("Instance namespace:", [attr for attr in dir(self) if not attr.startswith('_')])

obj = MyClass()
obj.show_namespaces()
