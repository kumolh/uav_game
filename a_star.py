from settings import *
from heapq import heappush, heappop
class node:
    def __init__(self, x, y, x0, y0, x1, y1, father=None):
        self.x = x
        self.y = y
        self.to_end, self.to_start = self.distance(x0, y0, x1, y1)
        self.father = father
    
    def set_father(self, father):
        self.father = father

    def distance(self, x0, y0, x1, y1):
        return abs(self.x - x1) + abs(self.y - y1), abs(self.x - x0) + abs(self.y - y0)

    def __le__(self, node):
        t1, t2 = node.to_end, node.to_start
        if self.to_end + self.to_start < t1 + t2: return True
        elif self.to_end + self.to_start > t1 + t2: return False
        return self.to_end < t1

    def __lt__(self, node):
        t1, t2 = node.to_end, node.to_start
        if self.to_end + self.to_start < t1 + t2: return True
        elif self.to_end + self.to_start > t1 + t2: return False
        return self.to_end < t1

def a_star(map: list[list[str]], x0: int, y0: int, x1: int, y1: int)-> list[list[int]]:
    m, n = len(map), len(map[0])
    visited = [[x != ' ' for x in row] for row in map]
    visited[x1][y1] = False
    ans = []
    q = []
    cur = node(x0, y0, x0, y0, x1, y1)
    visited[cur.x][cur.y] = True
    heappush(q, cur)
    while q:
        cur = heappop(q)
        if cur.to_end == 0: break
        visited[cur.x][cur.y] = True
        for dx, dy in [[-1, 0], [1, 0], [0, 1], [0, -1]]:
            x = cur.x + dx
            y = cur.y + dy
            if 0 <= x < m and 0 <= y < n and not visited[x][y]:
                new_node = node(x, y, x0, y0, x1, y1, cur)
                visited[new_node.x][new_node.y] = True
                heappush(q, new_node)
        
        for dx, dy in [[-1, -1], [1, 1], [-1, 1], [1, -1]]:
            x = cur.x + dx
            y = cur.y + dy
            if 0 <= x < m and 0 <= y < n and not visited[x][y] and map[cur.x][y] != 'x' and map[x][cur.y] != 'x':
                new_node = node(x, y, x0, y0, x1, y1, cur)
                visited[new_node.x][new_node.y] = True
                heappush(q, new_node)
    while cur:
        ans.append([cur.x, cur.y])
        cur = cur.father
    return ans[::-1]

# if __name__ == '__main__':
#     map = WORLD_MAP
#     x0, y0 = MAPY - 3, 2
#     x1, y1 = MAPY - 2, MAPX - 1
#     path = a_star(map, x0, y0, x1, y1)
#     print(path)
