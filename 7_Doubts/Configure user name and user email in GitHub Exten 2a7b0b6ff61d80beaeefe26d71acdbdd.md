# Configure user.name and user.email in GitHub Extension

# Method 1: Using command prompt

1. Open command prompt in the project location.
2. Give the following command in the terminal, 

```powershell
git config --global [user.name](http://user.name/) "Your Name"
git config --global user.email "[youremail@example.com](mailto:youremail@example.com)"
```

Example:

```powershell
git config --global [user.name](http://user.name/) "revantharunachalam"
git config --global user.email "[r](mailto:revanth@rayofreflection.org)evanth.sbioa@gmail.com"
```

1. To verify the username and email is set successfully, 

```powershell
git config --global --list
```

The configuration should reflect here.