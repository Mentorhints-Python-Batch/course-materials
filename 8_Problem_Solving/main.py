from find_parent import find_parent

graph = {
    1: [2], 
    2: [3, 4],
    3: [5, 6, 7],
    4: [8, 9, 10, 11],
    5: [12, 13, 14], 
    6: [15, 16, 17, 18],
}

print()
print()
# parent = find_parent(2, graph)
# if (parent == -999999):
#     print("Parent not found")
help(find_parent)