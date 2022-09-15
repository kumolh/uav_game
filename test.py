
def third():
    lsts = ['aaaa', 'aa']
    lst = [4, 2]
    n, m = 6, 2
    s = 'aaaaaa'
    lsts.sort()
    tmp = [len(t) for t in lsts]
    if sorted(lst) != sorted(tmp):
        print(0)
    else:
        ans = 0
        visited = [False] * m
        def dfs(i, num):
            nonlocal ans
            if num == m:
                ans += 1
                return
            x = 0
            while x < m:
                if visited[x]: 
                    x += 1
                    continue
                if lsts[x] == s[i:i+len(lsts[x])]:
                    visited[x] = True
                    dfs(i+len(lsts[x]), num+1)
                    visited[x] = False
                while x + 1 < m and lsts[x+1] == lsts[x]:
                    x += 1
                x += 1
        dfs(0, 0)
        print(ans)
def f1():
    n, m, d = 3, 3, 2
    lsts = [[270, 0], [270, 1], [250, 2]]
    lsts.sort(key=lambda x: (-x[0], x[1]))
    print(lsts[d-1][1] + 1)

def f2():
    [n, k] = 3, 3
    s = '3 2 1'
    lst = [int[x] for x in s.strip().split(' ')]
    lst.sort()
    left = 0
    right = len(lst) - 1
    ans = 0
    # print(1)
    while left < right:
        while left < right and lst[left] * lst[right] < k:
            left += 1
        if left < right:
            ans += (right - left) * 2
        right -= 1
    print(ans)

def f3():
    n = 6
    s = '1 2 2 3 3'
    lst = [int(x) for x in s.strip().split(' ')]
    from collections import deque 
    paths = [[] for _ in range(n+1)]
    degree = [0] * (n + 1)
    for i, o in enumerate(lst):
        paths[i+2].append(o)
        paths[o].append(i+2)
        degree[i+2] += 1
        degree[o] += 1
    q = deque()
    for i in range(1, 1 + n):
        if degree[i] == 1:
            q.append(i)
    visited = [False] * (n + 1)
    ans = 0
    while q:
        l = len(q)
        for _ in range(l):
            idx = q.popleft()
            for i in paths[idx]:
                degree[i] -= 1
                if not visited[i] and not visited[idx]:
                    ans += 1
                    visited[i] = True
                    visited[idx] = True
                if degree[i] == 1:
                    q.append(i)
    print(ans)

def f4():
    class ListNode:
        def __init__(self):
            self.data = None
            self.next = None

    class Solution:
        def reverseBetween(self, head, left, right):
            # Write Code Here 
            stack = []
            dummy = ListNode()
            dummy.data = -1000
            dummy.next = head
            p = dummy
            idx = 1
            while p.next and idx < left:
                p = p.next
                idx += 1
            q = p
            # return q
            p = p.next
            while p and idx - 1 < right:
                stack.append(p)
                idx += 1
                p = p.next
            while stack:
                q.next = stack.pop()
                q = q.next
            q.next = p
            return dummy.next

    head = None
    head_curr = None
    for x in [1,2, 3, 4, 5]:
        head_temp = ListNode()
        head_temp.data = int(x)
        head_temp.next = None
        if head == None:
            head = head_temp
            head_curr = head
        else:
            head_curr.next = head_temp
            head_curr = head_temp

    left = 2
    right = 3


    s = Solution()
    res = s.reverseBetween(head, left, right)

    while res != None:
        print(str(res.data) + " "),
        res = res.next
    print("")

def f5():
    class Node:
        def __init__(self, data=0, left=None, right=None):
            self.data = data
            self.left = left
            self.right = right


    lst = [10, 6, 14, 4, 8, 12, 16]
    # for x in input().split():
    #     lst.append(int(x))
    lst.sort()
    head = Node(lst[0])
    p = head
    for i in range(1, len(lst)):
        tmp = Node(lst[i])
        p.right = tmp
        tmp.left = p
        p = tmp
    tail = p
    while head:
        print(str(head.data), end=' ')
        head = head.right
    while tail:
        if tail.left:
            print(str(tail.data), end=' ')
        else:
            print(str(tail.data))
        tail = tail.left

if __name__ == '__main__':
    f5()
    # t = 'abcaacc'
    # s = 'a*c'
    # ans = 0
    # def check(idx):
    #     # nonlocal s, t
    #     if s[0] != '*' and t[idx] != s[0]: 
    #         return False
    #     if s[-1] != '*' and t[idx+len(s)-1] != s[-1]: 
    #         return False
    #     for i in range(len(s)):
    #         if s[i] != '*' and t[idx+i] != s[i]: 
    #             return False
    #     return True
    # for i in range(len(t) - len(s) + 1):
    #     if check(i): 
    #         ans += 1
    # print(ans)
    # 
    # third()

    # n, m = 5, 3
    # lst = [2, 3, 4]
    # visited = [False] * (n + 1)
    # num = 0
    # ans = []
    # for idx in range(m-1, -1, -1):
    #     if visited[lst[idx]]: continue
    #     num += 1
    #     visited[lst[idx]] = True
    #     if num != n:
    #         print(lst[idx], end=" ")
    #     else:
    #         print(lst[idx])
    # for i in range(1, 1+n):
    #     if not visited[i]:
    #         num += 1
    #         visited[i] = True
    #         if num != n:
    #             print(i, end=" ")
    #         else:
    #             print(i)
#     小美在回家路上碰见很多缠人的可爱猫猫！因为猫猫太可爱了以及小美十分有爱心，每遇到一只猫猫，小美忍不住停下来花费T的时间抚摸猫猫让猫猫不再缠着小美。而一路上小美能捡到很多亮闪闪的小玩具，这里我们给每个小玩具的种类都编了号，从1~k，一共k种小玩具，对于每个所属种类i的小玩具，小美可以选择将它送给遇到的一只猫猫玩，这样的话可以只花费ti的时间就可以让这只猫猫心满意足的离开。小美想知道，在她以最佳的对小玩具的用法下，她最少耗费多少时间在打发猫猫（即只考虑摸猫时间以及用小玩具打发猫的时间）。

# 注意，每个捡到的小玩具只能用一次。



# 输入描述
# 第一行三个正整数n、k和T，分别表示小美回家遇见的事件数、小玩具种类总数以及摸猫时间！

# 接下来一行k个数t1,t2, ...,tk, 含义如题所述，为每种小玩具打发猫猫所用时间。

# 接下来一行n个数e1,e2, ...,en ，表示n次事件，对第i次事件，如果ei=0，则表示遇到了一只猫猫，小美可以选择花费T的时间去抚摸，或者用一个小玩具送给猫猫来打发它(如果小美有的话)。如果ei>0，则表示小美在这里捡到了一个小玩具，种类为ei。初始时候小美身上没有任何小玩具，她可以携带任意多个小玩具。

# 对于所有数据，1≤n≤50000，1≤k≤50000，0≤ei≤k, 1≤T,ti≤104

# 样例输入
# 6 2 100
# 1 50
# 0 1 2 0 1 0
# 样例输出
# 102
# 样例解释

# 一开始没有小玩具，遇到一只猫猫只能抚摸，花费了100的时间。

# 接下来获得了小玩具1和2，然后遇到一只猫猫，用了小玩具1，只花费了1的时间。

# 接下来又获得一个小玩具1之后又遇见一只猫猫，因为又有小玩具1了，所以还是只用花费1的时间。

# 总共用时102