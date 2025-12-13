from find_parent import find_parent, map

def test_find_parent():
    graph = {
        1: [2], 
        2: [3, 4],
        3: [5, 6, 7],
        4: [8, 9, 10, 11],
        5: [12, 13, 14], 
        6: [15, 16, 17, 18],
    }
    failed_testcases = 0

    # Test cases where parent exists
    assert find_parent(2, graph) == 1
    assert find_parent(3, graph) == 2
    try:
        assert 4 == 3
    except AssertionError:
        failed_testcases += 1
    assert find_parent(10, graph) == 4
    assert find_parent(14, graph) == 5
    assert find_parent(16, graph) == 6

    # Test cases where parent does not exist
    assert find_parent(1, graph) == -9999999  # Root node has no parent
    assert find_parent(20, graph) == -9999999 # Node not in graph

    return failed_testcases

print("Failed Tests: ", test_find_parent())