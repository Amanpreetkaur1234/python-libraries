import heapq
def best_first_search(graph,heuristic,start,goal):
    priority_queue =[]
    heapq.heappush(priority_queue,(heuristic[start],start,[start]))
    visited =set()
    while priority_queue:
        current_h,current_node,path = heapq.heappop(priority_queue)
        print(f"Visited:{current_node} with Heuristic :{current_h}")
        if current_node == goal:
            print("goal reached!")
            print("path:","->".join(path))
            return path
        visited.add(current_node)
        for neighbour in graph.get(current_node,[]):
            if neighbour not in visited:
                heapq.heappush(
                    priority_queue, (heuristic[neighbour],neighbour,path+[neighbour])

                )
    print("no path found from",start,"to",goal)
    return None
graph ={
    'A': ['B','C','D'],
    'B': ['E'],
    'C':['F','E'],
    'D':['F'],
    'E':['H'],
    'F':['G'],
    'H':['G'],
    'G':[]

}   
heuristic = {
    'A': 40,
    'B':32,
    'C':25,
    'D':35,
    'E':19,
    'F':17,
    'H':10,
    'G':0
}   
start_node = input("enter the start node:").strip().upper()
goal_node = input("enter the goal Node :").strip().upper()
if start_node not in graph or goal_node not in graph :
    print("invalid start or goal Node.")
else:
    best_first_search(graph,heuristic,start_node,goal_node)      
        