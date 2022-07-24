def find_longest(lst):
    if len(lst) < 2: return len(lst)
    lst.sort(key = lambda x: (x[0], -x[1]))
    ans = [lst[0][1]]

    for idx in range(1, len(lst)):
        # change length
        if lst[idx][1] > ans[-1]: # 不可能长度相等，因为降序排列
            ans.append(lst[idx][1])
            continue
        # change width of ans list （只看长度）肯定可以放进去，因为长度至少和之前的相等，宽度甚至可以更短
        l, r = 0, len(ans) - 1
        while l < r:
            mid = (l + r) >> 1
            if ans[mid] > lst[idx][1]:
                l = mid + 1
            else:
                r = mid
        ans[l] = lst[idx][1]
    return len(ans)

if __name__ == '__main__':
    lst1 = [[1,2], [3,4],[1,1],[4,1]]
    ans = find_longest(lst1)
    print(ans)