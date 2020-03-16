#!/usr/bin/python
# -*- coding: utf-8 -*-

from collections import namedtuple
Item = namedtuple("Item", ['index', 'value', 'weight'])

def greedy (capacity, items):
    value = 0
    weight = 0
    taken = [0]*len(items)
    
    #sort by value
    #items.sort (key=lambda s: (-s.value))
    
    #sort by weight
    #items.sort (key=lambda s: (s.weight))
    
    #sort by weight density
    items.sort (key=lambda s: (-s.value/s.weight, -s.value))
    
    #print (items)
    
    for item in items:
        temp = weight + item.weight
        if temp <= capacity:
            taken[item.index] = 1
            value += item.value
            weight = temp
    return (value, taken)

def dp (capacity, items):
    value = 0
    weight = 0
    taken = [0]*len(items)
    
    l = len (items)
    
    if capacity < 1:
        return []
    if l == 0:
        return []
    
    # Initialize the matrix contains optimal value gien (capacity, #items) 
    #mat = [[0] * (l+1)] * (capacity+1)
    mat = [[0] * (l+1) for i in range(capacity+1) ]
    
    # Fill the matrix
    for idx in range (1, l+1):
        for k in range (1, capacity+1):
            if items[idx-1].weight > k:
                mat[k][idx] = mat[k][idx-1] 
                continue
            else:
                mat[k][idx] = max (mat[k][idx-1], items[idx-1].value + mat[k-items[idx-1].weight][idx-1])
    
    #import pdb; pdb.set_trace()
    value = mat[capacity][l]
    taken = [0]*l
    
    # Trace back
    check = 0
    k = capacity
    for idx in range (l, 0, -1):
        while k >= 0:
            #import pdb; pdb.set_trace()
            if check == 0:
                check = mat[k][idx]
            if check == mat[k][idx-1]:
                check = 0
                break
            else:
                taken[idx-1] = 1
                check = mat[k][idx] - items[idx-1].value
                k -= items[idx-1].weight
                break
                
    
    return (value, taken)

def solve_it(input_data):
    # Modify this code to run your optimization algorithm

    # parse the input
    lines = input_data.split('\n')

    firstLine = lines[0].split()
    item_count = int(firstLine[0])
    capacity = int(firstLine[1])

    items = []

    for i in range(1, item_count+1):
        line = lines[i]
        parts = line.split()
        items.append(Item(i-1, int(parts[0]), int(parts[1])))

    # a trivial algorithm for filling the knapsack
    # it takes items in-order until the knapsack is full
    """value = 0
    weight = 0
    taken = [0]*len(items)

    for item in items:
        if weight + item.weight <= capacity:
            taken[item.index] = 1
            value += item.value
            weight += item.weight"""
    
    # greedy solution
    #value, taken = greedy (capacity, items)
    
    # dp solution
    value, taken = dp (capacity, items)
    
    # prepare the solution in the specified output format
    output_data = str(value) + ' ' + str(0) + '\n'
    output_data += ' '.join(map(str, taken))
    return output_data


"""if __name__ == '__main__':
    import sys
    if len(sys.argv) > 1:
        file_location = sys.argv[1].strip()
        with open(file_location, 'r') as input_data_file:
            input_data = input_data_file.read()
        print(solve_it(input_data))
    else:
        print('This test requires an input file.  Please select one from the data directory. (i.e. python solver.py ./data/ks_4_0)')
"""
if __name__ == '__main__':
    file_location = "./data/ks_lecture_dp_2"
    
    with open(file_location, 'r') as input_data_file:
        input_data = input_data_file.read()
        
    print(solve_it(input_data))
    
    
    
    
    
    
