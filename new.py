# 我们在某项工艺中，至少会用到3个零件，同时要求：用到的每个零件必须按顺序输入且每个零件的尺寸差都要一样。
# 现在我们按顺序给出一系列的零件，要求在其中找出符合要求的序列总数。
# 如：1,4,7,5,3,1,1,1中共有5个符合要求，分别是：[1,4,7]; [7,5,3]; [7,5,3,1]; [5,3,1]; [1,1,1];  故输出：5
# 再如：3,4,5,6中共有3个符合要求，分别是：[3,4,5]; [4,5,6]; [3,4,5,6];  故输出：3

def find_valid_num(lst:list[int])->int:
    n = len(lst)
    if n < 3: 
        return 0
    num = ans = 0
    lst.append(-10000)
    for i in range(2, n+1):
        if lst[i] - lst[i-1] == lst[i-1] - lst[i-2]:
            num = 3 if num == 0 else num + 1
        else:
            # (num - 2) + (num - 3) + .. 1
            ans += (num - 1) * (num - 2) // 2
            num = 0
    # if num != 0:
    #     ans += (num - 1) * (num - 2) // 2
    return ans

if __name__ == '__main__':
    lst1 = [1,4,7,5,3,1,1,1]
    lst2 = [3,4,5,6]
    print('answer1 is {}'.format(find_valid_num(lst1)))
    print('answer2 is {}'.format(find_valid_num(lst2)))

