# TS-GU的主文件
from gurobipy import *
import matplotlib.pyplot as plt
import numpy as np
import xlrd, math, random
from collections import deque
import time

# 其他内容
stations_f = ['YBW', 'TXR', 'DWR', 'SDS',
            'XSJ', 'JDR', 'CJB', 'XTR',
            'GMR', 'CCR', 'ZJC', 'SWJ', 'LXR', "CIP"]
stations_b = ['CIP', 'LXR', 'SWJ', 'ZJC', 'CCR', 'GMR', 'XTR', 'CJB', 'JDR', 'XSJ', 'SDS', 'DWR', 'TXR', 'YBW']
# 定义（集合）下标范围
t_range = range(1, 61)  # 7:30-8:29
t_m_range = range(1, 99)  # 覆盖所有站点的时间区间art
t_n_range = range(1, 73)  # 覆盖所有站点的时间区间bus
k_art_range = range(1, 9)  # art行程数，本文设置了8趟行程，10分钟一趟
g_range = range(1, 3)   # g ∈ [1, 2]
d_range = range(1, 3)   # d ∈ [1, 2]
n_range = range(1, 3)   # n ∈ [1, 2]
s_art_range = range(1, 15)  # 公交站序号，14个站
s_bus_range = range(1, 4)  # 公交站序号，3个站
t_s_art = [2, 2, 2, 2, 2, 2, 3, 2, 2, 2, 2, 1, 1]  # 智轨的行驶时间
t_s_bus = [5, 5]  # 公交的行驶时间
proportion_a_to_b = [1, 1, 1, 1, 0, 0, 0, 0, 0, 0]  # 在换乘站art乘客选择bus的比例（各时段）
s_d_trans = [3, 12]  # art两个方向的换乘站序号
T_art_depot_in = range(44, 104)  # 针对入库决策变量的集合
T_art_depot_out = range(-4, 56)  # 针对出库决策变量的集合
T_art_turn_in = range(41, 101)  # 针对入折返线决策变量的集合
T_art_turn_out = range(-1, 59)  # 针对出折返线决策变量的集合
t_1_range = range(6, 66)
HS = [49, 59]  # 出发高铁的时间
HD = [22, 35]  # 到达高铁的时间
T_bus_depot_in = range(16, 76)
T_bus_depot_out = range(-2, 58)
T_bus_turn_in = range(15, 75)
T_bus_turn_out = range(-1, 59)
n_d_trans = [0.8, 0.2]  # 接驳公交选择换乘方向d的art的乘客比例
t_d_trans = [5, 31]  # 从起点站到换乘站的旅行时间
allow_range = {1: range(1, 11), 2: range(6, 21), 3: range(11, 31), 4: range(16, 41),
                   5: range(21, 46), 6: range(31, 51), 7: range(41, 56), 8: range(51, 61)}  # 各决策变量允许选择的范围
n_choice_range = [1, 1, 1, 1, 1, 1, 1, 1, 1, 2]  # 选择初始解和产生领域用的range

# 定义参数值
random.seed(4)  # 固定种子
h_art_min = 5
h_art_max = 10
h_bus_min = 5
c_art = 100
c_bus = 50
h_hsr_min = 25
h_hsr_max = 40
dw_art = 1
dw_bus = 1
t_turn_in = 2
t_turn_out = 2
t_depot_in = 5
t_depot_out = 5
u_cur_in = 2
u_cur_out = 2
u_depot_in = 3
u_depot_out = 3
M = 3000  # 特大的数
u_g = {1: 4, 2: 2}  # 各折返线可存储的unit数
u_cur = 2
t_art_total = 38
t_bus_total = 12

# 读取ART到达率信息
data = xlrd.open_workbook("arrive_rate_f.xls")
table = data.sheets()[0]
nrows = table.nrows
arrival_list_f = []  # 用于存放读取到的到达率
for i in range(1, nrows):
    ls_list = table.row_values(i)
    del ls_list[0]  # 去除标签
    arrival_list_f.append(ls_list)

data1 = xlrd.open_workbook("arrive_rate_b.xls")
table1 = data1.sheets()[0]
nrows1 = table1.nrows
arrival_list_b = []  # 用于存放读取到的到达率
for i in range(1, nrows1):
    ls_list = table1.row_values(i)
    del ls_list[0]  # 去除标签
    arrival_list_b.append(ls_list)

# 定义到达率
arrival_rate = {}
# 初始化
for d in d_range:
    for s in s_art_range:
        for t in t_m_range:
            arrival_rate[(d, s, t)] = 0
# 写入
for d in d_range:
    for s in s_art_range:
        if d == 1:
            for t in t_range:
                time_section = math.floor((t-1)/10)
                arrival_rate[(d, s, t)] = arrival_list_f[s-1][time_section]
        else:
            for t in t_range:
                time_section = math.floor((t-1)/10)
                arrival_rate[(d, s, t)] = arrival_list_b[14-s][time_section]  # 方向，车站序号，时间

# 读取bus到达率
data4 = xlrd.open_workbook("arrive_rate_bus.xls")
table4 = data4.sheets()[0]
arrival_list_bus = []  # 用于存放读取到的到达率
ls_list = table4.row_values(1)
del ls_list[0]  # 去除标签
# 定义到达率
arrival_rate_bus = {}
for t in t_n_range:
    arrival_rate_bus[t] = 0
# 写入
for t in t_range:
    arrival_rate_bus[t] = ls_list[t-1]

# 读取ART的OD信息
data2 = xlrd.open_workbook("od_f_total.xls")
table2 = data2.sheets()[0]
nrows2 = table2.nrows
OD_list_f = []  # 存储读取的OD信息
for i in range(2, nrows2):
    od_inf = table2.row_values(i)
    OD_list_f.append([(stations_f.index(od_inf[0])+1, stations_f.index(od_inf[1])+1), od_inf[2]])

data3 = xlrd.open_workbook("od_b_total.xls")
table3 = data3.sheets()[0]
nrows3 = table3.nrows
OD_list_b = []  # 存储读取的OD信息
for i in range(2, nrows3):
    od_inf = table3.row_values(i)
    OD_list_b.append([(stations_b.index(od_inf[0])+1, stations_b.index(od_inf[1])+1), od_inf[2]])

# 定义OD出行需求
eta_f = {}  # 乘客从一个站前往另一个站的比率
eta_b = {}  # 乘客从一个站前往另一个站的比率
for i in OD_list_f:
    eta_f[(i[0][0], i[0][1])] = i[1]
for i in OD_list_b:
    eta_b[(i[0][0], i[0][1])] = i[1]

# 读取bus的OD信息
data5 = xlrd.open_workbook("od_bus_total.xls")
table5 = data5.sheets()[0]
nrows5 = table5.nrows
OD_list_bus = []  # 存储读取的OD信息
for i in range(2, nrows5):
    od_inf = table5.row_values(i)
    OD_list_bus.append([(od_inf[0], od_inf[1]), od_inf[2]])
# 定义OD出行需求
etb = {}
for i in OD_list_bus:
    etb[(i[0][0], i[0][1])] = i[1]

# 用于生成初始解x的8个时间
def sample_times(
    n=k_art_range[-1],
    low=t_range[0],
    high=t_range[-1],  # 时刻取值范围 [low,high]
    min_gap=h_art_min,
    max_gap=h_art_max,
    max_tries=1000
):
    for _ in range(max_tries):
        times = []
        prev = None
        ok = True
        for i in range(n):
            remain = n - i - 1
            if i == 0:
                # 第 1 趟：必须 ≤ low+max_gap
                lo = low
                hi = min(low + max_gap, high - remain * min_gap)
            elif i == n-1:
                # 最后一趟：必须 ≥ high-max_gap
                lo = max(prev + min_gap, high - max_gap)
                hi = min(prev + max_gap, high)
            else:
                # 中间趟：普通头尾剪枝
                lo = prev + min_gap
                hi = min(prev + max_gap, high - remain * min_gap)

            if lo > hi:
                ok = False
                break

            t = random.randint(lo, hi)
            times.append(t)
            prev = t

        if ok:
            return times

    # 失败则返回 None
    return None


def sample_bus_times(
    n=6,
    t_min=t_range[0],
    t_max=t_range[-1],
    min_gap=5,
    HS=(49, 59),
    min_before=25,
    max_before=45,
    max_tries=10000
):
    # 预计算每个 HS 的允许区间
    windows = [(hs - max_before, hs - min_before) for hs in HS]
    all_times = list(range(t_min, t_max + 1))

    for _ in range(max_tries):
        # 1) 从 [t_min, t_max] 随机选 n 个不重复的时刻
        times = sorted(random.sample(all_times, n))

        # 2) 检查最小间隔
        ok = all(times[i+1] - times[i] >= min_gap for i in range(n-1))
        if not ok:
            continue

        # 3) 检查每个 HS 的窗口约束
        ok = True
        for lo, hi in windows:
            if not any(lo <= t <= hi for t in times):
                ok = False
                break
        if not ok:
            continue

        # 全部通过
        return times

    return None


#  第二步：生成初始解（即同时定义、初始化决策变量）
def init_solution():
    sol = {
        'x': {},  # x[(d,k,t,n)] ∈ {0,1}
        'y': {}   # y[t] ∈ {0,1}
    }
    # 对决策变量x的随机选择，对每个 (d,k) 随机选一个 (t,n)
    times_f = sample_times()
    times_b = sample_times()
    units = [[], []]
    for d in d_range:
        for k in k_art_range:
            t, n = random.choice(allow_range[k]), random.choice(n_choice_range)
            units[d - 1].append(n)

    for d in d_range:
        for k in k_art_range:
            if d == 1:
                t = times_f[k-1]
            else:
                t = times_b[k-1]
            n = units[d-1][k-1]
            for tt in t_range:
                for nn in n_range:
                    sol['x'][(d, k, tt, nn)] = 1 if (tt == t and nn == n) else 0

    # 对决策变量y的随机选择
    times = sample_bus_times(n=6, HS=HS, min_before=h_hsr_min, max_before=h_hsr_max)
    for t in t_range:
        sol['y'][t] = 1 if t in times else 0

    return sol


# 第二步，评价函数，这里是所有约束的放置和计算目标函数值的部分
def evaluate(sol):
    x, y = sol['x'], sol['y']  # 读取决策变量x和y

    # 定义ART行程时间集合
    D_art = {}  # 定义art的出发时间集合:d,k,s
    for d in d_range:
        for k in k_art_range:
            for s in s_art_range:
                D_art[(d, k, s)] = 0
    A_art = {}  # 定义art的到达时间集合:d,k,s
    for d in d_range:
        for k in k_art_range:
            for s in s_art_range:
                A_art[(d, k, s)] = 0

    for d in d_range:
        for k in k_art_range:
            D_art[d, k, 1] = sum(t * x[d, k, t, n] for t in t_range for n in n_range)  # 确定art首站出发的时间（约束4）
            for s in s_art_range:
                if s != 1:
                    if d == 1:
                        A_art[d, k, s] = D_art[d, k, s-1] + t_s_art[s-2]
                    else:
                        A_art[d, k, s] = D_art[d, k, s-1] + t_s_art[14-s]  # 定义各站到达时间（约束6）
                    D_art[d, k, s] = A_art[d, k, s] + dw_art  # 定义各站出发时间（约束5)

    # 定义bus行程时间集合
    D_bus = {}
    k_v = 0  # 虚拟行程编号
    for t in t_range:
        if y[t] == 1:
            k_v += 1  # 行程编号+1
            D_bus[(k_v, 1)] = t  # 定义行程k从第一站出发的时间，并赋予虚拟行程编号
    k_bus_range = range(1, k_v+1)  # 确定bus编号范围

    A_bus = {}
    for k in k_bus_range:
        for s in s_bus_range:
            A_bus[(k, s)] = 0

    for k in k_bus_range:
        for s in s_bus_range:
            if s != 1:
                A_bus[(k, s)] = D_bus[(k, s-1)] + t_s_bus[s-2]
                D_bus[(k, s)] = A_bus[(k, s)] + dw_bus

    r_w = {}  # 定义art各行程各站的候车人数
    for d in d_range:
        for k in k_art_range:
            for s in s_art_range:
                r_w[(d, k, s)] = 0

    r_a = {}  # 定义art下车人数
    for d in d_range:
        for k in k_art_range:
            for s in s_art_range:
                r_a[(d, k, s)] = 0

    r_b = {}  # 定义art上车人数
    for d in d_range:
        for k in k_art_range:
            for s in s_art_range:
                r_b[(d, k, s)] = 0

    r_on = {}  # 定义art在车上的人数
    for d in d_range:
        for k in k_art_range:
            for s in s_art_range:
                r_on[(d, k, s)] = 0

    j_w = {}  # 定义各行程的候车人数
    for k in k_bus_range:
        for s in range(1, 3):
            j_w[(k, s)] = 0

    j_b = {}  # 定义各行程的上车人数
    for k in k_bus_range:
        for s in range(1, 3):
            j_b[(k, s)] = 0

    j_a = {}  # 定义各行程的下车人数
    for k in k_bus_range:
        for s in range(2, 3):
            j_a[(k, s)] = 0

    # 定义bus各行程在第一站的候车、上车人数（约束31、33），在第二站的下车人数（约束32）
    for k in k_bus_range:
        if k == 1:
            j_w[k, 1] = sum(arrival_rate_bus[t] for t in range(1, D_bus[k, 1]+1))
            j_b[k, 1] = min(c_bus, j_w[k, 1])
        else:
            j_w[k, 1] = j_w[k-1, 1] - j_b[k-1, 1] + sum(arrival_rate_bus[t] for t in range(D_bus[k-1, 1]+1, D_bus[k, 1]+1))
            j_b[k, 1] = min(c_bus, j_w[k, 1])
        j_a[k, 2] = j_b[k, 1]

    # 定义ART各行程在各站的候车、上车和下车人数（约束18-22，约束34-35）
    for d in d_range:
        for k in k_art_range:
            for s in s_art_range:
                if k == 1:  # 首先生成行程1的情况
                    if s == 1:
                        r_w[d, k, s] = sum(arrival_rate[d, s, t] for t in range(1, D_art[d, k, s] + 1))
                        r_b[d, k, s] = min(r_w[d, k, s], sum(n * x[d, k, t, n] for t in t_range for n in n_range) * c_art)
                        r_on[d, k, s] = r_b[d, k, s]
                    elif s != s_d_trans[d-1]:
                        # 非换乘站的情况
                        r_w[d, k, s] = sum(arrival_rate[d, s, t] for t in range(1, D_art[d, k, s] + 1))
                        if d == 1:
                            r_a[d, k, s] = sum(eta_f[i, s] * r_b[d, k, i] for i in range(1, s))
                        else:
                            r_a[d, k, s] = sum(eta_b[i, s] * r_b[d, k, i] for i in range(1, s))  # 先下车
                        r_b[d, k, s] = min(r_w[d, k, s], sum(n * x[d, k, t, n] for t in t_range for n in n_range) * c_art - r_on[d, k, s-1] + r_a[d, k, s])  # 后上车
                        r_on[d, k, s] = r_on[d, k, s-1] - r_a[d, k, s] + r_b[d, k, s]  # 最后更新车上情况
                    else:
                        # 换乘站的情况（考虑bus到来的乘客）
                        for k_1 in k_bus_range:
                            if 1 <= A_bus[k_1, 2] <= D_art[d, k, s]:
                                # 如果在第k趟art出发前有公交到达，则下车乘客计入art候车乘客序列
                                r_w[d, k, s] += j_a[k_1, 2] * n_d_trans[d-1]
                        r_w[d, k, s] += sum(arrival_rate[d, s, t] for t in range(1, D_art[d, k, s] + 1))  # 增加常规到达乘客
                        if d == 1:
                            r_a[d, k, s] = sum(eta_f[i, s] * r_b[d, k, i] for i in range(1, s))
                        else:
                            r_a[d, k, s] = sum(eta_b[i, s] * r_b[d, k, i] for i in range(1, s))  # 先下车
                        r_b[d, k, s] = min(r_w[d, k, s], sum(n * x[d, k, t, n] for t in t_range for n in n_range) * c_art - r_on[d, k, s - 1] + r_a[d, k, s])  # 后上车
                        r_on[d, k, s] = r_on[d, k, s - 1] - r_a[d, k, s] + r_b[d, k, s]  # 最后更新车上情况
                else:
                    # 除了第一个行程外的其他art行程
                    if s == 1:
                        r_w[d, k, s] = r_w[d, k-1, s] - r_b[d, k-1, s] + sum(arrival_rate[d, s, t] for t in range(D_art[d, k-1, s]+1, D_art[d, k, s]+1))
                        r_b[d, k, s] = min(r_w[d, k, s], sum(n * x[d, k, t, n] for t in t_range for n in n_range) * c_art)
                        r_on[d, k, s] = r_b[d, k, s]
                    elif s != s_d_trans[d - 1]:
                        # 非换乘站的情况
                        r_w[d, k, s] = r_w[d, k-1, s] - r_b[d, k-1, s] + sum(arrival_rate[d, s, t] for t in range(D_art[d, k-1, s]+1, D_art[d, k, s]+1))
                        if d == 1:
                            r_a[d, k, s] = sum(eta_f[i, s] * r_b[d, k, i] for i in range(1, s))
                        else:
                            r_a[d, k, s] = sum(eta_b[i, s] * r_b[d, k, i] for i in range(1, s))  # 先下车
                        r_b[d, k, s] = min(r_w[d, k, s], sum(n * x[d, k, t, n] for t in t_range for n in n_range) * c_art - r_on[d, k, s - 1] + r_a[d, k, s])  # 后上车
                        r_on[d, k, s] = r_on[d, k, s - 1] - r_a[d, k, s] + r_b[d, k, s]  # 最后更新车上情况
                    else:
                        # 换乘站的情况（考虑bus到来的乘客）
                        rt_value = 0  # 计算来自bus的换乘乘客数量
                        for k_1 in k_bus_range:
                            if D_art[d, k-1, s]+1 <= A_bus[k_1, 2] <= D_art[d, k, s]:
                                # 如果在第k趟art出发前有公交到达，则下车乘客计入art候车乘客序列
                                rt_value += j_a[k_1, 2] * n_d_trans[d-1]
                        r_w[d, k, s] = r_w[d, k-1, s] - r_b[d, k-1, s] + sum(arrival_rate[d, s, t] for t in range(D_art[d, k-1, s]+1, D_art[d, k, s]+1)) + rt_value
                        if d == 1:
                            r_a[d, k, s] = sum(eta_f[i, s] * r_b[d, k, i] for i in range(1, s))
                        else:
                            r_a[d, k, s] = sum(eta_b[i, s] * r_b[d, k, i] for i in range(1, s))  # 先下车
                        r_b[d, k, s] = min(r_w[d, k, s], sum(n * x[d, k, t, n] for t in t_range for n in n_range) * c_art - r_on[d, k, s - 1] + r_a[d, k, s])  # 后上车
                        r_on[d, k, s] = r_on[d, k, s - 1] - r_a[d, k, s] + r_b[d, k, s]  # 最后更新车上情况

    # 定义bus各行程在第二站候车和上车的人数（约束33，38），考虑来自art的换乘乘客数量
    for k in k_bus_range:
        if k == 1:
            jt = 0  # 计算累计从art换乘bus的乘客数量
            for d in d_range:
                for k_1 in k_art_range:
                    if 1 <= A_art[d, k_1, s_d_trans[d - 1]] <= D_bus[k, 2]:
                        section = math.floor((A_art[d, k_1, s_d_trans[d - 1]] - 1) / 10)
                        jt += r_a[d, k_1, s_d_trans[d - 1]] * proportion_a_to_b[section]
            j_w[k, 2] = jt  # 候车人数
            j_b[k, 2] = min(c_bus, j_w[k, 2])  # 上车人数
        else:
            # 除了第一个行程外的其他行程
            jt = 0  # 计算累计从art换乘bus的乘客数量
            for d in d_range:
                for k_1 in k_art_range:
                    if D_bus[k-1, 2]+1 <= A_art[d, k_1, s_d_trans[d - 1]] <= D_bus[k, 2]:
                        section = math.floor((A_art[d, k_1, s_d_trans[d - 1]] - 1) / 10)
                        jt += r_a[d, k_1, s_d_trans[d - 1]] * proportion_a_to_b[section]
            j_w[k, 2] = j_w[k-1, 2] - j_b[k-1, 2] + jt
            j_b[k, 2] = min(c_bus, j_w[k, 2])  # 上车人数

    # 目标函数的构建
    # Z1为ART乘客的候车时间，分为非换乘站和换乘站两部分
    a_1 = 1
    a_2 = 2
    Z1_1 = sum(arrival_rate[d, s, t] * (t_m_range[-1] - t) for t in t_m_range for d in d_range for s in s_art_range if s != s_d_trans[d - 1])
    Z1_1 -= sum(r_b[d, k, s] * (t_m_range[-1] - D_art[d, k, s]) for d in d_range for s in s_art_range if s != s_d_trans[d - 1] for k in k_art_range)

    Z1_2 = sum(arrival_rate[d, s_d_trans[d - 1], t] * (t_m_range[-1] - t) for t in t_m_range for d in d_range)
    Z1_2 += sum(j_a[k, 2] * (t_m_range[-1] - A_bus[k, 2]) for k in k_bus_range)  # 换乘乘客的候车时间
    Z1_2 -= sum(r_b[d, k, s_d_trans[d - 1]] * (t_m_range[-1] - D_art[d, k, s_d_trans[d - 1]]) for d in d_range for k in k_art_range)

    Z1 = a_1 * Z1_1 + a_2 * Z1_2
    # Z2为BUS乘客的候车时间，分为站点1和站点2两部分
    Z2_1 = sum(arrival_rate_bus[t] * (t_range[-1] - t) for t in t_range)
    Z2_1 -= sum(j_b[k, 1] * (t_range[-1] - D_bus[k, 1]) for k in k_bus_range)
    Z2_2 = 0
    for d in d_range:
        for k in k_art_range:
            section = math.floor((A_art[d, k, s_d_trans[d - 1]] - 1) / 10)
            Z2_2 += r_a[d, k, s_d_trans[d - 1]] * proportion_a_to_b[section] * (t_m_range[-1] - A_art[d, k, s_d_trans[d - 1]])
    Z2_2 -= sum(j_b[k, 2] * (t_m_range[-1] - D_bus[k, 2]) for k in k_bus_range)

    Z2 = a_1 * Z2_1 + a_2 * Z2_2

    # Z3为ART的运营成本，包括unit的总使用成本和行程固定行驶成本
    pc_art = 500  # art单元的采购成本
    tao = 10
    Z3 = sum(n * x[d, k, t, n] for d in d_range for k in k_art_range for t in t_range for n in n_range) * pc_art  # 购置成本
    Z3 += tao * sum(t_art_total * n * x[d, k, t, n] for d in d_range for k in k_art_range for t in t_range for n in n_range)  # 运行成本

    # Z4为BUS的运营成本，包括车辆的总使用成本和行程固定行驶成本
    pc_bus = 100
    myu = 10
    Z4 = sum(y[t] for t in t_range) * pc_bus
    Z4 += myu * sum(t_bus_total * y[t] for t in t_range)

    # 综合目标
    beita_1 = 0.5
    beita_2 = 0.5
    Z = beita_1 * (Z1 + Z2) + beita_2 * (Z3 + Z4)
    return Z


# 第三步，领域生成
def get_neighbors(sol, N=40):
    neighs = []
    for _ in range(N):
        nb = {'x': sol['x'].copy(), 'y': sol['y'].copy()}
        # 随机在一个 (d,k) 上换发车时刻或编组
        if random.random() < 0.7:  # 随机生成0-1的浮点数
            # 选择改变ART的行程
            d, k = random.choice(d_range), random.choice(k_art_range)  # 随机选择需要改变的行程
            for t in t_range:
                for n in n_range:
                    if nb['x'][(d, k, t, n)] == 1:
                        nb['x'][(d, k, t, n)] = 0  # 清除该行程的时间t和单位n
            # 随机选取新的t和n
            while True:
                t2, n2 = random.choice(allow_range[k]), random.choice(n_choice_range)  # 在行程k可行的时间t范围内进行选择

                # 判别是否满足最小、最大间隔，否则循环
                if k != 1 and k != k_art_range[-1]:
                    # 读取前一行程和后一行程的时间
                    de_k_1 = sum(t_1 * nb['x'][d, k-1, t_1, n_1] for t_1 in t_range for n_1 in n_range)  # 前一行程
                    de_k_2 = sum(t_2 * nb['x'][d, k+1, t_2, n_2] for t_2 in t_range for n_2 in n_range)  # 后一行程
                    if h_art_min <= t2 - de_k_1 <= h_art_max and h_art_min <= de_k_2 - t2 <= h_art_max:
                        nb['x'][(d, k, t2, n2)] = 1
                        break
                elif k == 1:
                    de_k_2 = sum(t_2 * nb['x'][d, k + 1, t_2, n_2] for t_2 in t_range for n_2 in n_range)  # 后一行程
                    if h_art_min <= de_k_2 - t2 <= h_art_max:
                        nb['x'][(d, k, t2, n2)] = 1
                        break
                else:
                    de_k_1 = sum(t_1 * nb['x'][d, k - 1, t_1, n_1] for t_1 in t_range for n_1 in n_range)  # 前一行程
                    if h_art_min <= t2 - de_k_1 <= h_art_max:
                        nb['x'][(d, k, t2, n2)] = 1
                        break
        else:
            # 小概率改一个y[t]
            y_1_list = []  # 装y=1的集合
            for t in t_range:
                if nb['y'][t] == 1:
                    y_1_list.append(t)
            while True:
                t0 = random.choice(t_range)  # 随机选一个时间
                if nb['y'][t0] == 1:
                    pb = False  # 判别y是否在HS覆盖范围内，不在则为False
                    pb_num = 0
                    for t_hs in HS:
                        if t_hs-h_hsr_max <= t0 < t_hs-h_hsr_min:
                            pb = True
                            pb_num = t_hs
                    if pb is True:
                        if sum(nb['y'][t] for t in range(pb_num-h_hsr_max, pb_num-h_hsr_min)) > 1:
                            nb['y'][t0] = 0  # 删除要保证不违反约束24
                            break
                    else:
                        nb['y'][t0] = 0  # 删除要保证不违反约束24
                        break

                else:
                    prev_time = max([t for t in y_1_list if t < t0], default=None)
                    next_time = min([t for t in y_1_list if t > t0], default=None)
                    if prev_time is not None and next_time is not None:
                        if t0 - prev_time >= h_bus_min and next_time - t0 >= h_bus_min:
                            nb['y'][t0] = 1
                            break
                    elif prev_time is None:
                        if next_time - t0 >= h_bus_min:
                            nb['y'][t0] = 1
                            break
                    else:
                        if t0 - prev_time >= h_bus_min:
                            nb['y'][t0] = 1
                            break
        neighs.append(nb)  # 添加至领域集合
    return neighs


def sol_signature(sol):
    """
    为解生成简洁签名：
      - ART 每趟发车时间列表 [(d,k,t)],
      - Bus 发车时刻列表 [t,...]
    """
    # 提取ART发车时刻
    art_times = []
    for (d,k,t,n), v in sol['x'].items():
        if v == 1:
            art_times.append((d,k,t,n))
    art_times.sort()
    # 提取Bus发车时刻
    bus_times = [t for t,v in sol['y'].items() if v == 1]
    bus_times.sort()
    return (tuple(art_times), tuple(bus_times))


# 禁忌搜索主流程
def tabu_search(tenure=50, max_iter=1000, stagnation_limit=100):
    # 1) 初始化
    best = init_solution()
    best_val = evaluate(best)
    curr, curr_val = best, best_val

    # 2) 禁忌表存签名
    tabu = deque(maxlen=tenure)
    tabu.append(sol_signature(curr))
    stagnant = 0  # 判断停滞更新的代数

    for it in range(max_iter):
        neighs = get_neighbors(curr)
        # 对邻域按目标值排序
        neighs.sort(key=evaluate)

        accepted = False
        for nb in neighs:
            sig = sol_signature(nb)
            val = evaluate(nb)
            # Aspiration: 如果这个解比 best 还优，即使在禁忌表里也接受
            if val < best_val:
                curr, curr_val = nb, val
                accepted = True
                break
            # 否则，只接受不在禁忌表中的
            if sig not in tabu:
                curr, curr_val = nb, val
                accepted = True
                break
        # 如果一个邻域都没选中，可选择最优候选（破禁忌）
        if not accepted:
            curr = neighs[0]
            curr_val = evaluate(curr)

        # 更新最优
        if curr_val < best_val:
            best, best_val = curr, curr_val
            stagnant = 0
        else:
            stagnant += 1

        # 更新禁忌表
        tabu.append(sol_signature(curr))

        if it % 100 == 0:
            print(f"Iter {it}, best_val={best_val:.2f}")

        # 检查停滞终止
        if stagnant >= stagnation_limit:
            print(f"连续{stagnation_limit}代无改进, 在{it}代结束搜索.")
            break

    return best, best_val


# 使用gurobi进行求解
def gurobi(sol):
    model = Model('op1')
    # 设置约束条件
    # 决策变量定义
    x = model.addVars(d_range, k_art_range, t_range, n_range, vtype=GRB.BINARY, name="x")
    y = model.addVars(t_range, vtype=GRB.BINARY, name="y")
    p_in = model.addVars(T_art_depot_in, g_range, n_range, vtype=GRB.BINARY, name="p_in")
    p_out = model.addVars(T_art_depot_out, g_range, n_range, vtype=GRB.BINARY, name="p_out")
    q_in = model.addVars(T_art_turn_in, g_range, n_range, vtype=GRB.BINARY, name="q_in")
    q_out = model.addVars(T_art_turn_out, g_range, n_range, vtype=GRB.BINARY, name="q_out")
    v_in = model.addVars(T_bus_depot_in, vtype=GRB.BINARY, name="v_in")
    v_out = model.addVars(T_bus_depot_out, vtype=GRB.BINARY, name="v_out")
    z_in = model.addVars(T_bus_turn_in, vtype=GRB.BINARY, name="z_in")
    z_out = model.addVars(T_bus_turn_out, vtype=GRB.BINARY, name="z_out")
    c_g = model.addVars(g_range, lb=5, ub=12, vtype=GRB.INTEGER, name="c_g")
    e_bus = model.addVar(lb=2, ub=4, vtype=GRB.INTEGER, name="e_bus")

    # 中间变量定义
    D_art = model.addVars(d_range, k_art_range, s_art_range, lb=1, vtype=GRB.CONTINUOUS, name="D_art")
    A_art = model.addVars(d_range, k_art_range, s_art_range, lb=1, vtype=GRB.CONTINUOUS, name="A_art")
    m = model.addVars(d_range, k_art_range, s_art_range, t_m_range, vtype=GRB.BINARY, name="m")
    m_union = model.addVars(d_range, k_art_range, s_art_range, t_m_range, vtype=GRB.CONTINUOUS, name="m_union")
    r_w = model.addVars(d_range, k_art_range, s_art_range, lb=0, vtype=GRB.CONTINUOUS, name="r_w")
    r_a = model.addVars(d_range, k_art_range, s_art_range, lb=0, vtype=GRB.CONTINUOUS, name="r_a")
    r_b = model.addVars(d_range, k_art_range, s_art_range, lb=0, vtype=GRB.CONTINUOUS, name="r_b")
    r_on = model.addVars(d_range, k_art_range, s_art_range, lb=0, vtype=GRB.CONTINUOUS, name="r_on")
    j_w = model.addVars(t_range, s_bus_range, lb=0, vtype=GRB.CONTINUOUS, name="j_w")
    j_a = model.addVars(t_range, s_bus_range, lb=0, vtype=GRB.CONTINUOUS, name="j_a")
    j_b = model.addVars(t_range, s_bus_range, lb=0, vtype=GRB.CONTINUOUS, name="j_b")
    rt_w = model.addVars(d_range, k_art_range, lb=0, vtype=GRB.CONTINUOUS, name="rt_w")
    gamma = model.addVars(d_range, k_art_range, t_m_range, vtype=GRB.BINARY, name="gamma")
    N = model.addVars(d_range, k_art_range, lb=1, ub=2, vtype=GRB.CONTINUOUS, name="N")

    # 目标函数类的中间变量
    Z_1 = model.addVar(lb=0, vtype=GRB.CONTINUOUS, name="Z_1")
    Z_1_1 = model.addVar(lb=0, vtype=GRB.CONTINUOUS, name="Z_1_1")
    Z_1_2 = model.addVar(lb=0, vtype=GRB.CONTINUOUS, name="Z_1_2")
    Z_2 = model.addVar(lb=0, vtype=GRB.CONTINUOUS, name="Z_2")
    Z_2_1 = model.addVar(lb=0, vtype=GRB.CONTINUOUS, name="Z_2_1")
    Z_2_2 = model.addVar(lb=0, vtype=GRB.CONTINUOUS, name="Z_2_2")
    Z_3 = model.addVar(lb=0, vtype=GRB.CONTINUOUS, name="Z_3")
    Z_4 = model.addVar(lb=0, vtype=GRB.CONTINUOUS, name="Z_4")
    # 辅助变量
    kappa = model.addVars(t_range, lb=0, vtype=GRB.CONTINUOUS, name="kappa")
    zb = model.addVars(t_range, [1, 2], lb=0, vtype=GRB.CONTINUOUS, name="zb")
    zb_1 = model.addVars(t_range, [1, 2], lb=0, vtype=GRB.CONTINUOUS, name="zb_1")
    za = model.addVars(d_range, k_art_range, s_art_range, lb=0, vtype=GRB.CONTINUOUS, name="za")
    nu = model.addVars(d_range, k_art_range, t_1_range, lb=0, vtype=GRB.CONTINUOUS, name="nu")
    omg = model.addVars(d_range, k_art_range, t_range, lb=0, vtype=GRB.CONTINUOUS, name="omg")
    eps = model.addVars(t_range, lb=0, vtype=GRB.CONTINUOUS, name="eps")
    rou = model.addVars(d_range, k_art_range, s_art_range, t_m_range, lb=0, vtype=GRB.CONTINUOUS, name="rou")
    rou_1 = model.addVars(d_range, k_art_range, t_m_range, lb=0, vtype=GRB.CONTINUOUS, name="rou_1")
    zeta = model.addVars(d_range, k_art_range, s_art_range, vtype=GRB.BINARY, name="zeta")
    zeta_1 = model.addVars(t_range, [1, 2], vtype=GRB.BINARY, name="zeta_1")

    # 根据tabu结果的预固定
    for d in d_range:
        for k in k_art_range:
            for t in t_range:
                for n in n_range:
                    if sol['x'][d, k, t, n] == 1:
                        x[d, k, t, n].lb = 1

    # 约束1
    for d in d_range:
        for k in k_art_range:
            lhs_1 = quicksum(x[d, k, t, n] for t in t_range for n in n_range)
            model.addConstr(lhs_1 == 1, name=f"constraint_1_{d}_{k}")

    # 约束2-3
    for t in t_range:
        for d in d_range:
            if t+h_art_min-1 in t_range:
                lhs_2 = quicksum(x[d, k, t, n] for t in range(t, t+h_art_min) for k in k_art_range for n in n_range)
                model.addConstr(lhs_2 <= 1, name=f"constraint_2_{t}_{d}")
            if t+h_art_max-1 in t_range:
                lhs_3 = quicksum(x[d, k, t, n] for t in range(t, t+h_art_max) for k in k_art_range for n in n_range)
                model.addConstr(lhs_3 >= 1, name=f"constraint_3_{t}_{d}")

    # 约束4
    for d in d_range:
        for k in k_art_range:
            lhs_4 = D_art[d, k, 1]
            rhs_4 = quicksum(t * x[d, k, t, n] for t in t_range for n in n_range)
            model.addConstr(lhs_4 == rhs_4, name=f"constraint_4_{d}_{k}")

    # 约束5
    for d in d_range:
        for k in k_art_range:
            for s in s_art_range:
                if s != 1:
                    rhs_5 = A_art[d, k, s] + dw_art
                    model.addConstr(D_art[d, k, s] == rhs_5, name=f"constraint_5_{d}_{k}_{s}")

    # 约束6
    for d in d_range:
        for k in k_art_range:
            for s in s_art_range:
                if s != 1:
                    if d == 1:
                        rhs_6 = D_art[d, k, s-1] + t_s_art[s-2]
                        model.addConstr(A_art[d, k, s] == rhs_6, name=f"constraint_6_{d}_{k}_{s}")
                    else:
                        rhs_6 = D_art[d, k, s-1] + t_s_art[14-s]
                        model.addConstr(A_art[d, k, s] == rhs_6, name=f"constraint_6_{d}_{k}_{s}")

    # 约束7
    for d in d_range:
        for k in k_art_range:
            for s in s_art_range:
                lhs_7 = quicksum(m[d, k, s, t] for t in t_m_range)
                model.addConstr(lhs_7 == D_art[d, k, s], name=f"constraint_7_{d}_{k}_{s}")

    # 约束8
    for d in d_range:
        for k in k_art_range:
            for s in s_art_range:
                for t in t_m_range:
                    if t != t_m_range[-1]:
                        model.addConstr(m[d, k, s, t] >= m[d, k, s, t+1], name=f"constraint_8_{d}_{k}_{s}_{t}")

    # 约束9
    for d in d_range:
        for k in k_art_range:
            for s in s_art_range:
                for t in t_m_range:
                    if k == 1:
                        model.addConstr(m_union[d, k, s, t] == m[d, k, s, t], name=f"constraint_9_{d}_{k}_{s}_{t}")
                    else:
                        rhs_9 = m[d, k, s, t] - m[d, k-1, s, t]
                        model.addConstr(m_union[d, k, s, t] == rhs_9, name=f"constraint_9_{d}_{k}_{s}_{t}")

    # 约束10
    for t in t_range:
        lhs_10 = quicksum(n * x[1, k, t, n] for k in k_art_range for n in n_range)
        rhs_10 = quicksum(n * p_out[t-t_depot_out, 1, n] for n in n_range) + quicksum(n * q_out[t-t_turn_out, 1, n] for n in n_range)
        model.addConstr(lhs_10 == rhs_10, name=f"constraint_10_{t}")

    # 约束11
    for t in t_range:
        lhs_11 = quicksum(n * x[1, k, t, n] for k in k_art_range for n in n_range)
        rhs_11 = quicksum(n * p_in[t+t_art_total+t_depot_in, 2, n] for n in n_range) + quicksum(n * q_in[t+t_art_total+t_turn_in, 2, n] for n in n_range)
        model.addConstr(lhs_11 == rhs_11, name=f"constraint_11_{t}")

    # 约束12
    for t in t_range:
        lhs_12 = quicksum(n * x[2, k, t, n] for k in k_art_range for n in n_range)
        rhs_12 = quicksum(n * p_out[t-t_depot_out, 2, n] for n in n_range) + quicksum(n * q_out[t-t_turn_out, 2, n] for n in n_range)
        model.addConstr(lhs_12 == rhs_12, name=f"constraint_12_{t}")

    # 约束13
    for t in t_range:
        lhs_13 = quicksum(n * x[2, k, t, n] for k in k_art_range for n in n_range)
        rhs_13 = quicksum(n * p_in[t+t_art_total+t_depot_in, 1, n] for n in n_range) + quicksum(n * q_in[t+t_art_total+t_turn_in, 1, n] for n in n_range)
        model.addConstr(lhs_13 == rhs_13, name=f"constraint_13_{t}")

    # 约束14
    for g in g_range:
        union_set = sorted(set(T_art_depot_in).union(set(T_art_depot_out)))
        t_depot_union = list(union_set)
        for t_total in t_depot_union:
            lhs_14 = (c_g[g] - quicksum(n * p_out[t, g, n] for t in T_art_depot_out if t <= t_total for n in n_range) +
                      quicksum(n * p_in[t, g, n] for t in T_art_depot_in if t <= t_total for n in n_range))
            model.addConstr(lhs_14 >= 0, name=f"constraint_14_{g}_{t_total}")

    # 约束15-16
    for g in g_range:
        union_set = sorted(set(T_art_turn_in).union(set(T_art_turn_out)))
        t_turn_union = list(union_set)
        for t_total in t_turn_union:
            lhs_15 = (quicksum(n * q_in[t, g, n] for t in T_art_turn_in if t <= t_total for n in n_range) -
                      quicksum(n * q_out[t, g, n] for t in T_art_turn_out if t <= t_total for n in n_range))
            model.addConstr(lhs_15 >= 0, name=f"constraint_15_{g}_{t_total}")
            model.addConstr(lhs_15 <= u_g[g], name=f"constraint_16_{g}_{t_total}")

    # 约束17
    for g in g_range:
        lhs_17 = (quicksum(n * q_in[t, g, n] for t in T_art_turn_in for n in n_range) -
                  quicksum(n * q_out[t, g, n] for t in T_art_turn_out for n in n_range))
        model.addConstr(lhs_17 == 0, name=f"constraint_17_{g}")

    # 约束18
    for d in d_range:
        for k in k_art_range:
            for s in s_art_range:
                if s != s_d_trans[d-1]:
                    if k == 1:
                        rhs_18 = quicksum(m_union[d, k, s, t] * arrival_rate[d, s, t] for t in t_m_range)
                        model.addConstr(r_w[d, k, s] == rhs_18, name=f"constraint_18_{d}_{k}_{s}")
                    else:
                        rhs_18 = r_w[d, k-1, s] - r_b[d, k-1, s] + quicksum(m_union[d, k, s, t] * arrival_rate[d, s, t] for t in t_m_range)
                        model.addConstr(r_w[d, k, s] == rhs_18, name=f"constraint_18_{d}_{k}_{s}")

    # 约束19
    for d in d_range:
        for k in k_art_range:
            for s in s_art_range:
                if d == 1:
                    rhs_19 = quicksum(eta_f[i, s] * r_b[d, k, i] for i in range(1, s))
                    model.addConstr(r_a[d, k, s] == rhs_19, name=f"constraint_19_{d}_{k}_{s}")
                else:
                    rhs_19 = quicksum(eta_b[i, s] * r_b[d, k, i] for i in range(1, s))
                    model.addConstr(r_a[d, k, s] == rhs_19, name=f"constraint_19_{d}_{k}_{s}")

    # 约束21
    for d in d_range:
        for k in k_art_range:
            rhs_21 = quicksum(n * x[d, k, t, n] for t in t_range for n in n_range)
            model.addConstr(N[d, k] == rhs_21, name=f"constraint_21_{d}_{k}")

    # 约束60（线性化过程）
    for d in d_range:
        for k in k_art_range:
            for s in s_art_range:
                if s != 1:
                    model.addConstr(za[d, k, s] <= r_w[d, k, s], name=f"constraint_60_1_{d}_{k}_{s}")
                    model.addConstr(za[d, k, s] <= N[d, k] * c_art - r_on[d, k, s-1] + r_a[d, k, s], name=f"constraint_60_2_{d}_{k}_{s}")
                    model.addConstr(za[d, k, s] >= r_w[d, k, s] - M * zeta[d, k, s], name=f"constraint_60_3_{d}_{k}_{s}")
                    model.addConstr(za[d, k, s] >= N[d, k] * c_art - r_on[d, k, s-1] + r_a[d, k, s] - M * (1 - zeta[d, k, s]), name=f"constraint_60_4_{d}_{k}_{s}")
                else:
                    model.addConstr(za[d, k, s] <= r_w[d, k, s], name=f"constraint_60_1_{d}_{k}_{s}")
                    model.addConstr(za[d, k, s] <= N[d, k] * c_art, name=f"constraint_60_2_{d}_{k}_{s}")
                    model.addConstr(za[d, k, s] >= r_w[d, k, s] - M * zeta[d, k, s], name=f"constraint_60_3_{d}_{k}_{s}")
                    model.addConstr(za[d, k, s] >= N[d, k] * c_art - M * (1 - zeta[d, k, s]), name=f"constraint_60_4_{d}_{k}_{s}")

    # 约束61（线性化过程）
    for d in d_range:
        for k in k_art_range:
            for s in s_art_range:
                if s != 1:
                    model.addConstr(r_w[d, k, s] - (N[d, k] * c_art - r_on[d, k, s-1] + r_a[d, k, s]) <= M * zeta[d, k, s], name=f"constraint_61_1_{d}_{k}_{s}")
                    model.addConstr((N[d, k] * c_art - r_on[d, k, s-1] + r_a[d, k, s]) - r_w[d, k, s] <= M * (1 - zeta[d, k, s]), name=f"constraint_61_2_{d}_{k}_{s}")
                else:
                    model.addConstr(r_w[d, k, s] - N[d, k] * c_art <= M * zeta[d, k, s], name=f"constraint_61_1_{d}_{k}_{s}")
                    model.addConstr(N[d, k] * c_art - r_w[d, k, s] <= M * (1 - zeta[d, k, s]), name=f"constraint_61_2_{d}_{k}_{s}")

    # 约束58（20的转化）
    for d in d_range:
        for k in k_art_range:
            for s in s_art_range:
                model.addConstr(r_b[d, k, s] == za[d, k, s], name=f"constraint_58_{d}_{k}_{s}")

    # 约束22
    for d in d_range:
        for k in k_art_range:
            for s in s_art_range:
                if s == 1:
                    model.addConstr(r_on[d, k, s] == r_b[d, k, s], name=f"constraint_22_{d}_{k}_{s}")
                else:
                    rhs_22 = r_on[d, k, s-1] - r_a[d, k, s] + r_b[d, k, s]
                    model.addConstr(r_on[d, k, s] == rhs_22, name=f"constraint_22_{d}_{k}_{s}")

    # 约束23
    for t_1 in t_range:
        if t_1+h_bus_min-1 <= t_range[-1]:
            lhs_23 = quicksum(y[t] for t in range(t_1, t_1+h_bus_min))
            model.addConstr(lhs_23 <= 1, name=f"constraint_23_{t_1}")

    # 约束24
    for t_2 in HS:
        lhs_24 = quicksum(y[t] for t in range(t_2-h_hsr_max, t_2-h_hsr_min))
        model.addConstr(lhs_24 >= 1, name=f"constraint_24_{t_2}")

    # 约束25
    for t in t_range:
        rhs_25 = v_out[t-u_depot_out] + z_out[t-u_cur_out]
        model.addConstr(y[t] == rhs_25, name=f"constraint_25_{t}")

    # 约束26
    for t in t_range:
        rhs_26 = v_in[t+t_bus_total+u_depot_in] + z_in[t+t_bus_total+u_cur_in]
        model.addConstr(y[t] == rhs_26, name=f"constraint_26_{t}")

    # 约束27
    union_set = sorted(set(T_bus_depot_in).union(set(T_bus_depot_out)))
    t_depot_union = list(union_set)
    for t_total in t_depot_union:
        lhs_27 = (e_bus - quicksum(v_out[t] for t in T_bus_depot_out if t <= t_total) +
                  quicksum(v_in[t] for t in T_bus_depot_in if t <= t_total))
        model.addConstr(lhs_27 >= 0, name=f"constraint_27_{t_total}")

    # 约束28-29
    union_set = sorted(set(T_bus_turn_in).union(set(T_bus_turn_out)))
    t_turn_union = list(union_set)
    for t_total in t_turn_union:
        lhs_28 = ((quicksum(z_in[t] for t in T_bus_turn_in if t <= t_total)) -
                  quicksum(z_out[t] for t in T_bus_turn_out if t <= t_total))
        model.addConstr(lhs_28 >= 0, name=f"constraint_28_{t_total}")
        model.addConstr(lhs_28 <= u_cur, name=f"constraint_29_{t_total}")

    # 约束30
    lhs_30 = quicksum(z_in[t] for t in T_bus_turn_in) - quicksum(z_out[t] for t in T_bus_turn_out)
    model.addConstr(lhs_30 == 0, name="constraint_30")

    # 约束57（31的线性化）
    for t in t_range:
        if t != 1:
            model.addConstr(kappa[t] <= j_w[t-1, 1] + arrival_rate_bus[t] + M * y[t-1], name=f"constraint_57_1_{t}")
            model.addConstr(kappa[t] >= j_w[t-1, 1] + arrival_rate_bus[t] - M * y[t-1], name=f"constraint_57_2_{t}")
            model.addConstr(kappa[t] <= j_w[t-1, 1] - j_b[t-1, 1] + arrival_rate_bus[t] + M * (1 - y[t-1]), name=f"constraint_57_3_{t}")
            model.addConstr(kappa[t] >= j_w[t-1, 1] - j_b[t-1, 1] + arrival_rate_bus[t] - M * (1 - y[t-1]), name=f"constraint_57_4_{t}")

    # 约束56（31的转化）
    for t in t_range:
        if t != 1:
            model.addConstr(j_w[t, 1] == kappa[t], name=f"constraint_56_{t}")
        else:
            model.addConstr(j_w[t, 1] == arrival_rate_bus[t], name=f"constraint_56_{t}")

    # 约束32
    for t in t_range:
        model.addConstr(j_a[t, 2] == j_b[t, 1], name=f"constraint_32_{t}")

    # 约束63（线性化过程）
    for t in t_range:
        for s in [1, 2]:
            model.addConstr(zb[t, s] <= c_bus, name=f"constraint_63_1_{t}_{s}")
            model.addConstr(zb[t, s] <= j_w[t, s], name=f"constraint_63_2_{t}_{s}")
            model.addConstr(zb[t, s] >= c_bus - M * zeta_1[t, s], name=f"constraint_63_3_{t}_{s}")
            model.addConstr(zb[t, s] >= j_w[t, s] - M * (1 - zeta_1[t, s]),  name=f"constraint_63_4_{t}_{s}")

    # 约束64（线性化过程）
    for t in t_range:
        for s in [1, 2]:
            model.addConstr(c_bus-j_w[t, s] <= M * zeta_1[t, s], name=f"constraint_64_1_{t}_{s}")
            model.addConstr(j_w[t, s] - c_bus <= M * (1 - zeta_1[t, s]), name=f"constraint_64_2_{t}_{s}")

    # 约束67（线性化过程）
    for t in t_range:
        for s in [1, 2]:
            model.addConstr(zb_1[t, s] <= M * y[t], name=f"constraint_67_1_{t}_{s}")
            model.addConstr(zb_1[t, s] <= zb[t, s], name=f"constraint_67_2_{t}_{s}")
            model.addConstr(zb_1[t, s] >= zb[t, s] - M * (1 - y[t]), name=f"constraint_67_3_{t}_{s}")
            model.addConstr(zb_1[t, s] >= 0, name=f"constraint_67_4_{t}_{s}")

    # 约束59（33的转化）
    for t in t_range:
        for s in [1, 2]:
            model.addConstr(j_b[t, s] == zb_1[t, s], name=f"constraint_59_{t}_{s}")

    # 约束70（线性化过程）
    for d in d_range:
        for k in k_art_range:
            for t in t_1_range:
                model.addConstr(nu[d, k, t] <= M * m_union[d, k, s_d_trans[d-1], t], name=f"constraint_70_1_{d}_{k}_{t}")
                model.addConstr(nu[d, k, t] <= j_a[t-t_s_bus[0], 2], name=f"constraint_70_2_{d}_{k}_{t}")
                model.addConstr(nu[d, k, t] >= j_a[t-t_s_bus[0], 2] - M * (1 - m_union[d, k, s_d_trans[d-1], t]), name=f"constraint_70_3_{d}_{k}_{t}")
                model.addConstr(nu[d, k, t] >= 0, name=f"constraint_70_4_{d}_{k}_{t}")

    # 约束68（34的转化）
    for d in d_range:
        for k in k_art_range:
            model.addConstr(rt_w[d, k] == quicksum(nu[d, k, t] * n_d_trans[d-1] for t in t_1_range), name=f"constraint_68_{d}_{k}")

    # 约束35
    for d in d_range:
        for k in k_art_range:
            if k == 1:
                rhs_35 = quicksum(m_union[d, k, s_d_trans[d-1], t] * arrival_rate[d, s_d_trans[d-1], t] for t in t_m_range) + rt_w[d, k]
                model.addConstr(r_w[d, k, s_d_trans[d-1]] == rhs_35, name=f"constraint_35_{d}_{k}")
            else:
                rhs_35 = r_w[d, k-1, s_d_trans[d-1]] - r_b[d, k-1, s_d_trans[d-1]] + quicksum(m_union[d, k, s_d_trans[d-1], t] * arrival_rate[d, s_d_trans[d-1], t] for t in t_m_range) + rt_w[d, k]
                model.addConstr(r_w[d, k, s_d_trans[d-1]] == rhs_35, name=f"constraint_35_{d}_{k}")

    # 约束36
    for d in d_range:
        for k in k_art_range:
            rhs_36 = quicksum(t * gamma[d, k, t] for t in t_m_range)
            model.addConstr(A_art[d, k, s_d_trans[d-1]] == rhs_36, name=f"constraint_36_{d}_{k}")

    # 约束37
    for d in d_range:
        for k in k_art_range:
            lhs_37 = quicksum(gamma[d, k, t] for t in t_m_range)
            model.addConstr(lhs_37 == 1, name=f"constraint_37_{d}_{k}")

    # 约束73（线性化过程）
    for d in d_range:
        for k in k_art_range:
            for t in t_range:
                model.addConstr(omg[d, k, t] <= M * gamma[d, k, t+t_s_bus[0]], name=f"constraint_73_1_{d}_{k}_{t}")
                model.addConstr(omg[d, k, t] <= r_a[d, k, s_d_trans[d-1]], name=f"constraint_73_2_{d}_{k}_{t}")
                model.addConstr(omg[d, k, t] >= r_a[d, k, s_d_trans[d-1]] - M * (1 - gamma[d, k, t+t_s_bus[0]]), name=f"constraint_73_3_{d}_{k}_{t}")
                model.addConstr(omg[d, k, t] >= 0, name=f"constraint_73_4_{d}_{k}_{t}")

    # 约束74（线性化过程）
    for t in t_range:
        if t != 1:
            section = math.floor((t+t_s_bus[0]-1) / 10)
            model.addConstr(eps[t] <= j_w[t-1, 2] + quicksum(omg[d, k, t] * proportion_a_to_b[section] for d in d_range for k in k_art_range) + M * y[t-1], name=f"constraint_74_1_{t}")
            model.addConstr(eps[t] >= j_w[t-1, 2] + quicksum(omg[d, k, t] * proportion_a_to_b[section] for d in d_range for k in k_art_range) - M * y[t-1], name=f"constraint_74_2_{t}")
            model.addConstr(eps[t] <= j_w[t-1, 2] - j_b[t-1, 2] + quicksum(omg[d, k, t] * proportion_a_to_b[section] for d in d_range for k in k_art_range) + M * (1 - y[t-1]), name=f"constraint_74_3_{t}")
            model.addConstr(eps[t] >= j_w[t-1, 2] - j_b[t-1, 2] + quicksum(omg[d, k, t] * proportion_a_to_b[section] for d in d_range for k in k_art_range) - M * (1 - y[t-1]), name=f"constraint_74_4_{t}")

    # 约束71（38的转化）
    for t in t_range:
        if t != 1:
            model.addConstr(j_w[t, 2] == eps[t], name=f"constraint_71_{t}")
        else:
            section = math.floor((t + t_s_bus[0] - 1) / 10)
            model.addConstr(j_w[t, 2] == quicksum(omg[d, k, t] * proportion_a_to_b[section] for d in d_range for k in k_art_range), name=f"constraint_71_{t}")

    # 约束77（线性化过程）
    for d in d_range:
        for k in k_art_range:
            for s in s_art_range:
                for t in t_m_range:
                    if s != s_d_trans[d-1]:
                        model.addConstr(rou[d, k, s, t] <= r_b[d, k, s], name=f"constraint_77_1_{d}_{k}_{s}_{t}")
                        model.addConstr(rou[d, k, s, t] <= M * (1 - m[d, k, s, t]), name=f"constraint_77_2_{d}_{k}_{s}_{t}")
                        model.addConstr(rou[d, k, s, t] >= r_b[d, k, s] - M * m[d, k, s, t], name=f"constraint_77_3_{d}_{k}_{s}_{t}")
                        model.addConstr(rou[d, k, s, t] >= 0, name=f"constraint_77_4_{d}_{k}_{s}_{t}")

    # 约束79（线性化过程）
    for d in d_range:
        for k in k_art_range:
            for t in t_m_range:
                model.addConstr(rou_1[d, k, t] <= r_b[d, k, s_d_trans[d-1]], name=f"constraint_79_1_{d}_{k}_{t}")
                model.addConstr(rou_1[d, k, t] <= M * (1 - m[d, k, s_d_trans[d-1], t]), name=f"constraint_79_2_{d}_{k}_{t}")
                model.addConstr(rou_1[d, k, t] >= r_b[d, k, s_d_trans[d-1]] - M * m[d, k, s_d_trans[d-1], t], name=f"constraint_79_3_{d}_{k}_{t}")
                model.addConstr(rou_1[d, k, t] >= 0, name=f"constraint_79_4_{d}_{k}_{t}")

    # 目标函数75（51的转化）
    a_1 = 1
    a_2 = 2
    rhs_75_1 = a_1 * quicksum((quicksum(arrival_rate[d, s, t] * (t_m_range[-1] - t) for t in t_m_range) - quicksum(rou[d, k, s, t] for k in k_art_range for t in t_m_range)) for d in d_range for s in s_art_range if s != s_d_trans[d-1])
    rhs_75_2 = a_2 * quicksum((quicksum(arrival_rate[d, s_d_trans[d-1], t] * (t_m_range[-1]-t) for t in t_m_range) - quicksum(rou_1[d, k, t] for k in k_art_range for t in t_m_range)) for d in d_range) + a_2 * quicksum(j_a[t, 2] * (t_m_range[-1]-t-t_s_bus[0]) for t in t_range)
    model.addConstr(Z_1_1 == rhs_75_1/a_1, name="objective_75_1")
    model.addConstr(Z_1_2 == rhs_75_2/a_2, name="objective_75_2")
    model.addConstr(Z_1 == rhs_75_1 + rhs_75_2, name="objective_75")

    # 目标函数52
    rhs_52_1 = a_1 * quicksum(j_w[t, 1] for t in t_range)
    rhs_52_2 = a_2 * quicksum(j_w[t, 2] for t in t_range)
    model.addConstr(Z_2_1 == rhs_52_1/a_1, name="objective_52_1")
    model.addConstr(Z_2_2 == rhs_52_2/a_2, name="objective_52_2")
    model.addConstr(Z_2 == rhs_52_1 + rhs_52_2, name="objective_52")

    # 目标函数53
    pc_art = 500  # art单元的采购成本
    tao = 10
    rhs_53 = pc_art * quicksum(c_g[g] for g in g_range) + tao * (quicksum(t_art_total * n * x[d, k, t, n] for d in d_range for k in k_art_range for t in t_range for n in n_range) +
                                                                 quicksum(t_depot_in * n * p_in[t, g, n] for g in g_range for t in T_art_depot_in for n in n_range) +
                                                                 quicksum(t_depot_out * n * p_out[t, g, n] for g in g_range for t in T_art_depot_out for n in n_range) +
                                                                 quicksum(t_turn_in * n * q_in[t, g, n] for g in g_range for t in T_art_turn_in for n in n_range) +
                                                                 quicksum(t_turn_out * n * q_out[t, g, n] for g in g_range for t in T_art_turn_out for n in n_range))
    model.addConstr(Z_3 == rhs_53, name="objective_53")

    # 目标函数54
    pc_bus = 200
    myu = 10
    rhs_54 = pc_bus * e_bus + myu * (quicksum(t_bus_total * y[t] for t in t_range) +
                                     quicksum(u_depot_in * v_in[t] for t in T_bus_depot_in) +
                                     quicksum(u_depot_out * v_out[t] for t in T_bus_depot_out) +
                                     quicksum(u_cur_in * z_in[t] for t in T_bus_turn_in) +
                                     quicksum(u_cur_out * z_out[t] for t in T_bus_turn_out))
    model.addConstr(Z_4 == rhs_54, name="objective_54")

    # 设置总目标
    beita_1 = 0.5
    beita_2 = 0.5
    model.setObjective(beita_1 * (Z_1+Z_2) + beita_2 * (Z_3+Z_4), GRB.MINIMIZE)

    return model, x, y, c_g, e_bus, Z_1, Z_1_1, Z_1_2, Z_2, Z_2_1, Z_2_2, Z_3, Z_4, p_in, p_out, q_in, q_out, z_in, z_out, v_in, v_out, r_w, j_w


def pic_illustrate(x_time_f, x_time_b, p_in, p_out, q_in, q_out, z_in, z_out, v_in, v_out, r_w, j_w):
    # 运行图绘制
    station_name = ['D1', 'T1', 'YBW', 'TXR', 'DWR', 'YBR', 'TB', 'DB', 'SDS',
                    'XSJ', 'JDR', 'CJB', 'XTR',
                    'GMR', 'CCR', 'ZJC', 'SWJ', 'LXR', "CIP", 'T2', 'D2']
    times = np.arange(7.5, 9.5, 1 / 4)  # 时间范围是7-9点，间隔10分钟
    station_index_art_f = (
    2, 2, 3, 3, 4, 4, 8, 8, 9, 9, 10, 10, 11, 11, 12, 12, 13, 13, 14, 14, 15, 15, 16, 16, 17, 17, 18,
    18)  # art的站点顺序（方向f）
    station_index_art_b = (
    18, 18, 17, 17, 16, 16, 15, 15, 14, 14, 13, 13, 12, 12, 11, 11, 10, 10, 9, 9, 8, 8, 4, 4, 3, 3, 2,
    2)  # art的站点顺序（方向b）
    station_index_bus = (5, 5, 4, 4, 5, 5)  # bus的站点顺序（全程）
    depot_index_art_out = {1: (0, 2), 2: (20, 18)}  # art的depot的站点运行顺序
    depot_index_art_in = {1: (2, 0), 2: (18, 20)}  # art的depot的站点运行顺序
    depot_index_bus_out = (7, 5)  # bus的depot的站点运行顺序
    depot_index_bus_in = (5, 7)  # bus的depot的站点运行顺序
    turn_index_art_out = {1: (1, 2), 2: (19, 18)}  # art的turn的站点运行顺序
    turn_index_art_in = {1: (2, 1), 2: (18, 19)}  # art的turn的站点运行顺序
    turn_index_bus_out = (6, 5)  # bus的turn的站点运行顺序
    turn_index_bus_in = (5, 6)  # bus的turn的站点运行顺序
    hsr_arrival_index = (5, 5)  # hsr到达的点位序号
    hsr_departure_index = (5, 5)  # hsr出发的点位序号
    t_index_art_f = (3, 6, 9, 12, 15, 18, 22, 25, 28, 31, 34, 36, 38)  # 距离首站的运行时间
    t_index_art_b = (2, 4, 7, 10, 13, 16, 20, 23, 26, 29, 32, 35, 38)  # 距离首站的运行时间
    t_index_bus = (6, 12)  # 距离首站的运行时间
    linewidth = {1: 0.85, 2: 1.4}  # 根据n决定线条的粗细，n=1时是细的
    bus_linewidth = 1
    plt.figure(figsize=(12, 8))
    plt.rcParams['font.family'] = 'Times New Roman'
    plt.axhspan(4, 7, facecolor='gray', alpha=0.1)

    index = 0
    for t in x_time_f:  # 绘制各站之间的点和线
        train_times = []
        for s in s_art_range:
            if s != 1:
                train_times.append(7.5 + (t + t_index_art_f[s - 2] - 2) / 60)
                train_times.append(7.5 + (t + t_index_art_f[s - 2] - 1) / 60)
            else:
                train_times.append(7.5 + (t - 2) / 60)
                train_times.append(7.5 + (t - 1) / 60)
        plt.plot(train_times, station_index_art_f, color='deepskyblue', marker='o', markersize=1,
                 linewidth=linewidth[n_f[index]], linestyle="-", zorder=2)
        index += 1

    index = 0
    for t in x_time_b:
        train_times = []
        for s in s_art_range:
            if s != 1:
                train_times.append(7.5 + (t + t_index_art_b[s - 2] - 2) / 60)
                train_times.append(7.5 + (t + t_index_art_b[s - 2] - 1) / 60)
            else:
                train_times.append(7.5 + (t - 2) / 60)
                train_times.append(7.5 + (t - 1) / 60)
        plt.plot(train_times, station_index_art_b, color='deepskyblue', marker='o', markersize=1,
                 linewidth=linewidth[n_b[index]], linestyle="-", zorder=2)
        index += 1

    for g in g_range:
        for t in T_art_depot_out:
            for n in n_range:
                train_times = []
                if p_out[t, g, n].x > 0.1:
                    train_times.append(7.5 + (t - 1) / 60)
                    train_times.append(7.5 + (t + t_depot_out - 2) / 60)
                    plt.plot(train_times, depot_index_art_out[g], color='deepskyblue', marker='o', markersize=1,
                             linewidth=linewidth[n], linestyle=":", zorder=2)

        for t in T_art_depot_in:
            for n in n_range:
                train_times = []
                if p_in[t, g, n].x > 0.1:
                    train_times.append(7.5 + (t - t_depot_in - 1) / 60)
                    train_times.append(7.5 + (t - 1) / 60)
                    plt.plot(train_times, depot_index_art_in[g], color='deepskyblue', marker='o', markersize=1,
                             linewidth=linewidth[n], linestyle=":", zorder=2)

        for t in T_art_turn_out:
            for n in n_range:
                train_times = []
                if q_out[t, g, n].x > 0.1:
                    train_times.append(7.5 + (t - 1) / 60)
                    train_times.append(7.5 + (t + t_turn_out - 2) / 60)
                    plt.plot(train_times, turn_index_art_out[g], color='deepskyblue', marker='o', markersize=1,
                             linewidth=linewidth[n], linestyle=":", zorder=2)

        for t in T_art_turn_in:
            for n in n_range:
                train_times = []
                if q_in[t, g, n].x > 0.1:
                    train_times.append(7.5 + (t - t_turn_in - 1) / 60)
                    train_times.append(7.5 + (t - 1) / 60)
                    plt.plot(train_times, turn_index_art_in[g], color='deepskyblue', marker='o', markersize=1,
                             linewidth=linewidth[n], linestyle=":", zorder=2)

    index = 0
    for t in y_time:  # 绘制bus各站之间的点和线
        train_times = []
        for s in s_bus_range:
            if s != 1:
                train_times.append(7.5 + (t + t_index_bus[s - 2] - 2) / 60)
                train_times.append(7.5 + (t + t_index_bus[s - 2] - 1) / 60)
            else:
                train_times.append(7.5 + (t - 2) / 60)
                train_times.append(7.5 + (t - 1) / 60)
        plt.plot(train_times, station_index_bus, color='orangered', marker='o', markersize=0.5, linewidth=bus_linewidth,
                 linestyle="--", zorder=3)
        index += 1

    for t in T_bus_depot_out:
        train_times = []
        if v_out[t].x > 0.1:
            train_times.append(7.5 + (t - 1) / 60)
            train_times.append(7.5 + (t + u_depot_out - 2) / 60)
            plt.plot(train_times, depot_index_bus_out, color='orangered', marker='o', markersize=0.5, linewidth=bus_linewidth,
                     linestyle=":", zorder=3)

    for t in T_bus_depot_in:
        train_times = []
        if v_in[t].x > 0.1:
            train_times.append(7.5 + (t - u_depot_in - 1) / 60)
            train_times.append(7.5 + (t - 1) / 60)
            plt.plot(train_times, depot_index_bus_in, color='orangered', marker='o', markersize=0.5, linewidth=bus_linewidth,
                     linestyle=":", zorder=3)

    for t in T_bus_turn_out:
        train_times = []
        if z_out[t].x > 0.1:
            train_times.append(7.5 + (t - 1) / 60)
            train_times.append(7.5 + (t + u_cur_out - 2) / 60)
            plt.plot(train_times, turn_index_bus_out, color='orangered', marker='o', markersize=0.5, linewidth=bus_linewidth,
                     linestyle=":", zorder=3)

    for t in T_bus_turn_in:
        train_times = []
        if z_in[t].x > 0.1:
            train_times.append(7.5 + (t - u_cur_in - 1) / 60)
            train_times.append(7.5 + (t - 1) / 60)
            plt.plot(train_times, turn_index_bus_in, color='orangered', marker='o', markersize=0.5, linewidth=bus_linewidth,
                     linestyle=":", zorder=3)

    # 补充表示高铁出发、到达时间点
    train_times = []
    for t in HS:
        train_times.append(7.5 + (t - 1) / 60)
    plt.scatter(train_times, hsr_arrival_index, color='green', marker='D', s=1.5, zorder=4)
    train_times = []
    for t in HD:
        train_times.append(7.5 + (t - 1) / 60)
    plt.scatter(train_times, hsr_departure_index, color='orange', marker='D', s=1.5, zorder=4)

    plt.yticks(np.arange(len(station_name)), station_name)
    plt.xticks(np.arange(7.5, 9.5, 1 / 4), [f'{int(t)}:{int((t % 1) * 60):02d}' for t in times])

    # 补充表示出发、到达时间点的label
    for t in HS:
        plt.text(
            7.5 + (t - 1) / 60,  # x 坐标向左偏 0.5
            5.3,  # 与箭头起点同一 y
            'De',
            ha='center',  # 文本右对齐，这样文字会从左向右写
            va='center'  # 垂直居中对齐
        )

    for t in HD:
        plt.text(
            7.5 + (t - 1) / 60,  # x 坐标向左偏 0.5
            5.3,  # 与箭头起点同一 y
            'Ar',
            ha='center',  # 文本右对齐，这样文字会从左向右写
            va='center'  # 垂直居中对齐
        )

    # 绘制公交各时间t在各站的候车情况
    wait_passenger_1 = []  # 第一站各时段等候乘客数量
    wait_passenger_2 = []  # 第二站各时段等候乘客数量
    bus_times_1 = []
    bus_times_2 = []
    for t in t_range:
        wait_passenger_1.append(-1*j_w[t, 1].x / 200)
        wait_passenger_2.append(j_w[t, 2].x / 200)
        bus_times_1.append(7.5 + (t - 1 - 1) / 60)
        bus_times_2.append(7.5 + (t + 5 - 1) / 60)
    plt.bar(bus_times_1, wait_passenger_1, width=0.0167, align='edge', bottom=5, color='darkorange', alpha=0.5, zorder=1)
    plt.bar(bus_times_2, wait_passenger_2, width=0.0167, align='edge', bottom=4, color='darkorange', alpha=0.5, zorder=1)

    # 绘制ART各行程在各站的候车情况
    station_index_art_f = (2, 3, 4, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18)  # art的站点顺序（方向f）
    station_index_art_b = (18, 17, 16, 15, 14, 13, 12, 11, 10, 9, 8, 4, 3, 2)  # art的站点顺序（方向b）
    k = 0
    for t in x_time_f:  # 绘制各站之间的点和线
        k += 1
        train_times_wait_f = []
        wait_passenger_1 = []
        for s in s_art_range:
            if s != 1:
                train_times_wait_f.append(7.5 + (t + t_index_art_f[s - 2] - 2) / 60)
            else:
                train_times_wait_f.append(7.5 + (t - 2) / 60)
            wait_passenger_1.append(r_w[1, k, s].x / 200)
        plt.bar(train_times_wait_f, wait_passenger_1, width=0.0167, align='edge', bottom=station_index_art_f, color='purple', alpha=0.5, zorder=1)

    k = 0
    for t in x_time_b:  # 绘制各站之间的点和线
        k += 1
        train_times_wait_b = []
        wait_passenger_2 = []
        for s in s_art_range:
            if s != 1:
                train_times_wait_b.append(7.5 + (t + t_index_art_b[s - 2] - 2) / 60)
            else:
                train_times_wait_b.append(7.5 + (t - 2) / 60)
            wait_passenger_2.append(-1 * r_w[2, k, s].x / 200)
        plt.bar(train_times_wait_b, wait_passenger_2, width=0.0167, align='edge', bottom=station_index_art_b, color='navy', alpha=0.5, zorder=1)

    plt.xlabel('Time')
    plt.ylabel('Stations')
    plt.grid(True)
    plt.savefig('S1.png',
                dpi=300,  # 分辨率
                bbox_inches='tight'  # 紧凑边距
                )
    plt.show()


# 两阶段求解主程序
if __name__ == "__main__":
    start_time = time.time()
    # 先使用禁忌搜索确定x的值
    print("————————————————")
    print("启动禁忌搜索")
    print("————————————————")
    sol, val = tabu_search(tenure=10, max_iter=500, stagnation_limit=150)
    print("禁忌搜索最优目标值：", val)
    # 使用gurobi求解剩余部分，将x的解作为输入
    print("—————————————————————")
    print("启动Gurobi求解剩余部分")
    print("—————————————————————")
    model, x, y, c_g, e_bus, Z_1, Z_1_1, Z_1_2, Z_2, Z_2_1, Z_2_2, Z_3, Z_4, p_in, p_out, q_in, q_out, z_in, z_out, v_in, v_out, r_w, j_w = gurobi(sol)
    # 改变求解参数
    model.setParam("Cuts", 2)
    model.setParam("MIPFocus", 2)
    # 求解模型
    model.optimize()
    model.write("518.sol")
    if model.status == GRB.INFEASIBLE:
        model.computeIIS()
        model.write("4.22.ilp")

    # 结果输出
    result_x = []  # 存放x的结果
    result_y = []  # 存放y的结果
    for d in d_range:
        for k in k_art_range:
            for t in t_range:
                for n in n_range:
                    if x[d, k, t, n].x > 0.1:
                        result_x.append((d, k, t, n))

    for t in t_range:
        if y[t].x > 0.1:
            result_y.append(t)

    x_time_f = []
    x_time_b = []
    n_f = []
    n_b = []
    y_time = []
    c_g_result = []
    for x in result_x:
        if x[0] == 1:
            x_time_f.append(x[2])
            n_f.append(x[3])
        else:
            x_time_b.append(x[2])
            n_b.append(x[3])
    for y in result_y:
        y_time.append(y)
    print(f"ART方向f的发车时间为：{x_time_f}")
    print(f"ART方向b的发车时间为：{x_time_b}")
    print(f"bus的发车时间为：{y_time}")
    for g in g_range:
        c_g_result.append(c_g[g].x)
    print(f'n_f:{n_f}')
    print(f'n_b:{n_b}')
    print(f"ART两个车库分别有：{c_g_result}个单元")
    print(f"Bus车库有：{e_bus.x}个车")
    print("————————————")
    print(f"乘客等待时间为：{Z_1.x + Z_2.x}")
    print(f"运营商成本为：{Z_3.x + Z_4.x}")
    print(f"Z1_1={Z_1_1.x}")
    print(f"Z1_2={Z_1_2.x}")
    print(f"Z2_1={Z_2_1.x}")
    print(f"Z2_2={Z_2_2.x}")
    end_time = time.time()
    elapsed_time = end_time - start_time
    print("————————————")
    print(f"程序运行时长: {elapsed_time} seconds")
    # 绘图
    pic_illustrate(x_time_f, x_time_b, p_in, p_out, q_in, q_out, z_in, z_out, v_in, v_out, r_w, j_w)