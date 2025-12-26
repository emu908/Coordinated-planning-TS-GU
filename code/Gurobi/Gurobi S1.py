# S1-8-60 Senario 1 Gurobi
from gurobipy import *
import xlrd
import math

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
proportion_a_to_b = [1, 1, 1, 1, 0, 0, 0]  # 在换乘站art乘客选择bus的比例（各时段）
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

# 定义参数值
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
u_g = {1: 4, 2: 2}  # 各折返线可存储的unit数
u_cur = 2
t_art_total = 38
t_bus_total = 12

M = 3000  # 特大的数

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

# gurobi建模过程
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

# 额外约束-x
for d in d_range:
    for n in n_range:
        for t in range(11, 61):
            model.addConstr(x[d, 1, t, n] == 0, name=f"extra_x_1_{d}_{t}_{n}")
        for t in range(1, 6):
            model.addConstr(x[d, 2, t, n] == 0, name=f"extra_x_2_1_{d}_{t}_{n}")
        for t in range(21, 61):
            model.addConstr(x[d, 2, t, n] == 0, name=f"extra_x_2_2_{d}_{t}_{n}")
        for t in range(1, 11):
            model.addConstr(x[d, 3, t, n] == 0, name=f"extra_x_3_1_{d}_{t}_{n}")
        for t in range(31, 61):
            model.addConstr(x[d, 3, t, n] == 0, name=f"extra_x_3_2_{d}_{t}_{n}")
        for t in range(1, 16):
            model.addConstr(x[d, 4, t, n] == 0, name=f"extra_x_4_1_{d}_{t}_{n}")
        for t in range(41, 61):
            model.addConstr(x[d, 4, t, n] == 0, name=f"extra_x_4_2_{d}_{t}_{n}")
        for t in range(1, 21):
            model.addConstr(x[d, 5, t, n] == 0, name=f"extra_x_5_1_{d}_{t}_{n}")
        for t in range(46, 61):
            model.addConstr(x[d, 5, t, n] == 0, name=f"extra_x_5_2_{d}_{t}_{n}")
        for t in range(1, 31):
            model.addConstr(x[d, 6, t, n] == 0, name=f"extra_x_6_1_{d}_{t}_{n}")
        for t in range(51, 61):
            model.addConstr(x[d, 6, t, n] == 0, name=f"extra_x_6_2_{d}_{t}_{n}")
        for t in range(1, 41):
            model.addConstr(x[d, 7, t, n] == 0, name=f"extra_x_7_1_{d}_{t}_{n}")
        for t in range(56, 61):
            model.addConstr(x[d, 7, t, n] == 0, name=f"extra_x_7_2_{d}_{t}_{n}")
        for t in range(1, 50):
            model.addConstr(x[d, 8, t, n] == 0, name=f"extra_x_8_{d}_{t}_{n}")

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

# 改变求解参数
model.setParam("MIPFocus", 2)

# 求解模型
model.optimize()

# 错误调试
if model.status == GRB.INFEASIBLE:
    model.computeIIS()
    model.write("3.ilp")

# gurobi输出文件
model.write("1.24.lp")
model.write("1.24.sol")