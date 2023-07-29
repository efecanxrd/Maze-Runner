from queue import PriorityQueue
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
import tkinter as tk

def a_star(grid, start, goal):
    gScore = {start: 0}
    fScore = {start: heuristic(start, goal)}
    closedSet = set()
    openSet = PriorityQueue()
    openSet.put((fScore[start], start))
    
    cameFrom = {}
    
    while not openSet.empty():
        current = openSet.get()[1]
        
        if current == goal:
            return cameFrom, gScore[current]
        
        closedSet.add(current)
        
        for neighbor in get_neighbors(grid, current):
            if neighbor in closedSet:
                continue
                
            tentative_gScore = gScore[current] + dist_between(current, neighbor)
            
            if neighbor not in [i[1] for i in openSet.queue]:
                openSet.put((fScore.get(neighbor, float('inf')), neighbor))
            elif tentative_gScore >= gScore.get(neighbor, float('inf')):
                continue
                
            cameFrom[neighbor] = current
            gScore[neighbor] = tentative_gScore
            fScore[neighbor] = gScore[neighbor] + heuristic(neighbor, goal)
            
def heuristic(a,b):
    (x1,y1) = a
    (x2,y2) = b
    return abs(x1-x2) + abs(y1-y2)

def dist_between(a,b):
    return 1

def get_neighbors(grid,current):
    neighbors=[]
    x,y=current
   
    if x > 0 and grid[x-1][y]!='3':
        neighbors.append((x-1,y))
    if y > 0 and grid[x][y-1]!='3':
        neighbors.append((x,y-1))
    if x < len(grid)-1 and grid[x+1][y]!='3':
        neighbors.append((x+1,y))
    if y < len(grid[0])-1 and grid[x][y+1]!='3':
        neighbors.append((x,y+1))
        
    return neighbors

def reconstruct_path(cameFrom,current,start):
     path=[current]
     while current!=start:
         current=cameFrom[current]
         path.insert(0,current)
         
     return path

mode = '1'
grid=[]
with open('grid1.txt','r') as file:
        for line in file:
            grid.append(list(line.strip()))

def degistir():
    global grid
    global mode
    if mode == '1':
        grid = []
        with open('grid2.txt','r') as file:
            for line in file:
                grid.append(list(line.strip()))
        mode = '2'

    else:
        grid = []
        with open('grid1.txt','r') as file:
            for line in file:
                grid.append(list(line.strip()))
        mode = '1'

def run():
    global grid
    grid=np.array(grid)

    start=(0,np.argmin(grid[0]=='3'))
    goal = (len(grid) - 1, len(grid[0]) - 1)

    a = datetime.now()
    cameFrom,cost=a_star(grid,start=start,goal=goal)
    b = datetime.now()

    path=reconstruct_path(cameFrom=cameFrom,current=goal,start=start)

    for i in range(len(path)):
        x,y=path[i]
        grid[x][y]='X'
        
    print(f'Cost: {cost}')
    print(f'Path: {path}')
    print('Şu kadar zamanda bulundu: ',(b - a).microseconds / 1000,'ms')
    if mode == '1':
        plt.xticks(np.arange(0, 21, 1))
        plt.yticks(np.arange(0, 21, 1))
        plt.xlabel('')
        plt.ylabel('')
    elif mode == '2':
        plt.xticks(np.arange(0, 11, 1))
        plt.yticks(np.arange(0, 11, 1))     
        plt.xlabel('')
        plt.ylabel('')
    plt.imshow(np.where(grid=='X',10,np.where(grid=='3',5,np.where(grid=='2',4,np.where(grid=='0',2,np.where(grid==' ',np.nan,np.nan))))))
    plt.show()



if __name__ == '__main__':
    root = tk.Tk()
    ana_cerceve = tk.Frame(root)
    ana_cerceve.pack()

    axe_butonu = tk.Button(ana_cerceve, text="Url Değiştir", command=degistir)
    axe_butonu.pack(side="left")

    calistir_butonu = tk.Button(ana_cerceve, text="Çalıştır", command=run)
    calistir_butonu.pack(side="left")

    root.title("App")
    root.geometry("100x100")
    root.mainloop()