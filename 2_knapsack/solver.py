#!/usr/bin/python
# -*- coding: utf-8 -*-

from collections import namedtuple
import numpy as np
Item = namedtuple("Item", ['index', 'value', 'weight'])
Node = namedtuple("Node", ['level', 'value', 'weight', 'idxTaken'])

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
    mat = np.zeros ((capacity+1, l+1), dtype=np.uint32)
    
    # Fill the matrix
    for idx in range (1, l+1):
        for k in range (1, capacity+1):
            if items[idx-1].weight > k:
                mat[k][idx] = mat[k][idx-1] 
                continue
            else:
                mat[k][idx] = max (mat[k][idx-1], items[idx-1].value + \
                        mat[k-items[idx-1].weight][idx-1])
    
    value = mat[capacity][l]
    taken = [0]*l
    
    # Trace back
    check = 0
    k = capacity
    for idx in range (l, 0, -1):
        while k >= 0:
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
##### first try solution for breadth first branch and bound
def getBound(node, capacity, items):
    l = len (items)
    temp = 0 

    if node.weight > capacity:
        return 0

    # initialize
    level = node.level
    totWeight = node.weight
    bound = node.value 

    j = level + 1
    if j < l:
        temp = totWeight + items[j].weight
    else: 
        return bound

    while (j < l and temp < capacity):
        bound += items[j].value
        totWeight = temp

        j += 1
        if j < l:
            temp = totWeight + items[j].weight

    if j < l:
        bound += items[j].value * (capacity - totWeight)*1.0 / items[j].weight
    return bound

def breadth_first_branch_bound (capacity, items):

    import queue

    l = len (items)
    # sort the item list by value density first
    items.sort (key=lambda s: (-s.value/s.weight, -s.value))

    #################################################################
    v = Node (-1, 0, 0, [])

    maxProfit = 0
    bestTaken = []

    q = queue.Queue ()
    q.put (v)

    while not q.empty():
        u = q.get()

        if u.level + 1 > l-1:
            break

        # If take the item -> 
        level = u.level + 1
        weight = u.weight + items[level].weight
        value = u.value + items[level].value
        idxTaken = u.idxTaken + [1]
        v = Node (level, value, weight, idxTaken)

        boundV = getBound (v, capacity, items)

        if v.weight <= capacity and boundV > maxProfit:
            q.put (v)

        if v.weight <= capacity and v.value >= maxProfit:
            maxProfit = v.value
            bestTaken = v.idxTaken

        #import pdb; pdb.set_trace()

        # If don't take the item -> 
        level = u.level + 1
        weight = u.weight 
        value = u.value 
        idxTaken = u.idxTaken + [0]
        v = Node (level, value, weight, idxTaken)

        boundV = getBound (v, capacity, items)

        if boundV > maxProfit:
            q.put (v)


    #import pdb; pdb.set_trace()
    taken = [0] * l
    for i, chosen in enumerate (bestTaken):
        taken[items[i].index] = chosen


    return (maxProfit, taken)
########################################################

##### optimal solution for breadth first branch and bound
# - Important to have a good structure and define the meaning of Node \
#        (firstly drawing the decision tree maybe the good approach)
# - Important to implement getBound() function based on the Node structure
Node2 = namedtuple ("Node2", "level value weight taken")
def getBound2 (node, capacity, items):
    l = len (items)

    totWeight = node.weight
    bound = node.value

    j = node.level + 1
    while j < l:
        totWeight += items[j].weight
        if (totWeight <= capacity):
            bound += items[j].value
            j += 1
        else:
            totWeight -= items[j].weight
            break

    if totWeight < capacity:
        bound += (capacity - totWeight) * 1.0 / items[j-1].weight * items[j-1].value

    return bound

def bfs_branch_bound (capacity, items):
    import queue

    maxProfit = 0
    bestTaken = []
    root = Node2 (-1, 0, 0, [])

    l = len (items)

    q = queue.Queue()
    #q = queue.LifoQueue()

    q.put (root)

    items.sort (key=lambda s: (-s.value/s.weight, -s.value))

    while not q.empty():
        u = q.get()

        # Take the item
        level = u.level + 1
        if level > l-1:
            break

        value = u.value + items[level].value
        weight = u.weight + items[level].weight
        taken = u.taken + [1]
        
        v = Node2 (level, value, weight, taken)

        boundV = getBound2 (v, capacity, items)

        if v.weight <= capacity:
            if boundV > maxProfit:
                q.put (v)
            if v.value >= maxProfit:
                maxProfit = v.value
                bestTaken = v.taken
        print ("Take the item: {}, taken: {}".format (v, taken))
        
        # Don't take the item
        level = u.level + 1
        value = u.value
        weight = u.weight
        taken = u.taken + [0]

        v = Node2 (level, value, weight, taken)

        boundV = getBound2 (v, capacity, items)

        if boundV >= maxProfit:
            q.put (v)

        print ("Don't take the item: {}, taken: {}".format (v, taken))

    taken = [0] * l
    for i, t in enumerate (bestTaken):
        taken[items[i].index] = t

    return maxProfit, taken


# Depth first search branch and bound
import heapq
class PriorityQueue ():
    def __init__ (self):
        self.queue = []

    def push (self, item, priority):
        heapq.heappush (self.queue, (-priority, item)) # add a tuple to the queue, \
                # with the descending order

    def pop (self):
        return heapq.heappop (self.queue)[-1]

    def empty(self):
        return (len (self.queue) == 0)

# Always consider the node with largest bound
# Same with bfs but use a priority queue to store node
def best_first_branch_bound (capacity, items):
    import queue

    maxProfit = 0
    bestTaken = []
    root = Node2 (-1, 0, 0, [])

    l = len (items)

    q = PriorityQueue()

    #import pdb; pdb.set_trace()
    q.push (root, 0)

    items.sort (key=lambda s: (-s.value/s.weight, -s.value))

    while not q.empty():
        u = q.pop()

        # Take the item
        level = u.level + 1
        if level > l-1:
            break

        value = u.value + items[level].value
        weight = u.weight + items[level].weight
        taken = u.taken + [1]
        
        v = Node2 (level, value, weight, taken)

        boundV = getBound (v, capacity, items)

        if v.weight <= capacity:
            if boundV > maxProfit:
                q.push (v, boundV)
            if v.value >= maxProfit:
                maxProfit = v.value
                bestTaken = v.taken
        #print ("Take the item: {}, bound: {}, maxProfit: {}".\
        #        format (v, boundV, maxProfit))
        
        # Don't take the item
        level = u.level + 1
        value = u.value
        weight = u.weight
        taken = u.taken + [0]

        v = Node2 (level, value, weight, taken)

        boundV = getBound (v, capacity, items)

        if boundV >= maxProfit:
            q.push (v, boundV)

        #print ("Don't take the item: {}, bound: {}, maxProfit: {}".\
        #        format (v, boundV, maxProfit))

    taken = [0] * l
    for i, t in enumerate (bestTaken):
        taken[items[i].index] = t

    return maxProfit, taken
########################################################


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
    #value, taken = dp (capacity, items)

    # branch and bound solution
    #value, taken = breadth_first_branch_bound (capacity, items)
    #value, taken = bfs_branch_bound (capacity, items)
    value, taken = best_first_branch_bound (capacity, items)
    
    # prepare the solution in the specified output format
    #output_data = str(value) + ' ' + str(0) + '\n'
    output_data = str(value) + ' ' + str(1) + '\n'
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
    file_location = "./data/ks_4_0"
    
    with open(file_location, 'r') as input_data_file:
        input_data = input_data_file.read()
        
    print(solve_it(input_data))
    
    
    
    
    
    
