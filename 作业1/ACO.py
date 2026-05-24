import random  
import math  
import numpy as np
import matplotlib.pyplot as plt
class ACO(object):  # 定义蚁群算法类（Ant Colony Optimization）
    def __init__(self, num_city, data):  # 构造函数，num_city：城市数量，data：城市坐标数据
        self.m = 50  # 蚂蚁数量，种群规模为50只蚂蚁
        self.alpha = 1  # 信息素重要程度因子，控制信息素对选择概率的影响权重
        self.beta = 5  # 启发函数重要因子，控制距离（能见度）对选择概率的影响权重
        self.rho = 0.1  # 信息素挥发因子，每次迭代后信息素保留的比例为1-rho
        self.Q = 1  # 常量系数，用于计算信息素增量的大小
        self.num_city = num_city  # 城市规模（城市的总数量）
        self.location = data  # 城市坐标数据，形状为(num_city, 2)
        self.Tau = np.zeros([num_city, num_city])  # 信息素矩阵，初始化为全零，形状为(num_city, num_city)
        self.Table = [[0 for _ in range(num_city)] for _ in range(self.m)]  # 蚁群路径表：m只蚂蚁各自的访问顺序，每行对应一只蚂蚁的路径
        self.iter = 1  # 当前迭代次数计数器（初始为1）
        self.iter_max = 500  # 最大迭代次数，算法运行500代后停止
        self.dis_mat = self.compute_dis_mat(num_city, self.location)  # 计算城市之间的距离矩阵，自身到自身的距离设为inf
        self.Eta = 10. / self.dis_mat  # 启发式函数（能见度），取距离的倒数，距离越近启发值越大
        self.paths = None  # 蚁群中每只蚂蚁的路径长度列表，初始为None
        # 存储每个温度下的最终路径，画出收敛图
        self.iter_x = []  # 存储迭代次数（横坐标），用于绘制收敛曲线
        self.iter_y = []  # 存储每次迭代的最优路径长度（纵坐标），用于绘制收敛曲线
        # self.greedy_init(self.dis_mat,100,num_city)
    def greedy_init(self, dis_mat, num_total, num_city):  # 贪心初始化方法：用最近邻算法生成初始信息素分布（当前未调用）
        start_index = 0  # 起始城市的索引，从0号城市开始
        result = []  # 存储所有生成的贪心路径
        for i in range(num_total):  # 循环生成num_total条贪心路径
            rest = [x for x in range(0, num_city)]  # 还未访问的城市列表，初始包含所有城市
            # 所有起始点都已经生成了
            if start_index >= num_city:  # 如果所有城市都已被用作起点
                start_index = np.random.randint(0, num_city)  # 随机选择一个城市作为新的起点
                result.append(result[start_index].copy())  # 复制之前生成的一条路径作为新的路径
                continue  # 跳过本次循环的剩余部分
            current = start_index  # 将当前城市设为起始城市
            rest.remove(current)  # 从未访问列表中移除起始城市
            # 找到一条最近邻路径
            result_one = [current]  # 初始化当前路径，从起始城市开始
            while len(rest) != 0:  # 当还有未访问的城市时继续循环
                tmp_min = math.inf  # 初始化最小距离为正无穷大
                tmp_choose = -1  # 记录距离最近的候选城市索引
                for x in rest:  # 遍历所有未访问的城市
                    if dis_mat[current][x] < tmp_min:  # 如果当前城市到城市x的距离小于已知最小距离
                        tmp_min = dis_mat[current][x]  # 更新最小距离
                        tmp_choose = x  # 记录这个最近的城市索引
                current = tmp_choose  # 移动到最近的城市
                result_one.append(tmp_choose)  # 将选中的城市加入当前路径
                rest.remove(tmp_choose)  # 从未访问列表中移除选中的城市
            result.append(result_one)  # 将生成的贪心路径加入结果列表
            start_index += 1  # 起始索引加1，下一次从下一个城市开始
        pathlens = self.compute_paths(result)  # 计算所有贪心路径的长度
        sortindex = np.argsort(pathlens)  # 按路径长度从小到大排序，返回排序后的索引
        index = sortindex[0]  # 取出最短路径的索引
        result = result[index]  # 获取最短的那条贪心路径
        for i in range(len(result)-1):  # 遍历最短路径上的相邻城市对
            s = result[i]  # 当前城市
            s2 = result[i+1]  # 下一个城市
            self.Tau[s][s2]=1  # 在信息素矩阵中标记这条边，初始信息素浓度为1
        self.Tau[result[-1]][result[0]] = 1  # 首尾相连（回到起点），信息素浓度设为1
        # for i in range(num_city):
        #     for j in range(num_city):
        # return result

    # 轮盘赌选择
    def rand_choose(self, p):  # 轮盘赌选择算法，根据概率分布p随机选择一个城市
        x = np.random.rand()  # 生成一个[0,1)区间的均匀随机数
        for i, t in enumerate(p):  # 遍历概率列表，i为索引，t为对应概率值
            x -= t  # 用x依次减去各选项的概率值
            if x <= 0:  # 当累积概率超过随机数x时
                break  # 停止循环，此时i即为被选中的选项索引
        return i  # 返回被选中选项的索引

    # 生成蚁群
    def get_ants(self, num_city):  # 生成蚁群：让每只蚂蚁完成一次完整的城市遍历，构建完整路径
        for i in range(self.m):  # 遍历每一只蚂蚁（共m只）
            start = np.random.randint(num_city - 1)  # 随机选择一个城市作为蚂蚁i的出发点
            self.Table[i][0] = start  # 将起点记录在路径表的第一列
            unvisit = list([x for x in range(num_city) if x != start])  # 未访问城市列表（排除起点）
            current = start  # 当前所在城市初始化为起点
            j = 1  # 路径索引，从1开始（0已经是起点）
            while len(unvisit) != 0:  # 当还有未访问的城市时循环
                P = []  # 初始化转移概率列表
                # 通过信息素计算城市之间的转移概率
                for v in unvisit:  # 遍历每一个未访问的城市
                    P.append(self.Tau[current][v] ** self.alpha * self.Eta[current][v] ** self.beta)  # 计算转移概率的分子：信息素^alpha * 启发值^beta
                P_sum = sum(P)  # 计算所有未访问城市转移概率分子之和（归一化分母）
                P = [x / P_sum for x in P]  # 归一化处理，使转移概率之和为1
                # 轮盘赌选择一个城市
                index = self.rand_choose(P)  # 根据轮盘赌算法选择下一个要访问的城市在unvisit中的索引
                current = unvisit[index]  # 获取被选中城市的实际编号
                self.Table[i][j] = current  # 将选中的城市记录到蚂蚁i的路径表中
                unvisit.remove(current)  # 从未访问列表中移除已访问的城市
                j += 1  # 路径索引加1，准备记录下一个访问的城市

    # 计算不同城市之间的距离
    def compute_dis_mat(self, num_city, location):  # 计算城市间距离矩阵，使用欧几里得距离
        dis_mat = np.zeros((num_city, num_city))  # 初始化距离矩阵为全零
        for i in range(num_city):  # 遍历所有城市作为起点
            for j in range(num_city):  # 遍历所有城市作为终点
                if i == j:  # 如果是同一个城市
                    dis_mat[i][j] = np.inf  # 自身到自身的距离设为无穷大，避免蚂蚁留在原地
                    continue  # 跳过本次内层循环
                a = location[i]  # 获取城市i的坐标(x,y)
                b = location[j]  # 获取城市j的坐标(x,y)
                tmp = np.sqrt(sum([(x[0] - x[1]) ** 2 for x in zip(a, b)]))  # 计算欧几里得距离：sqrt((x1-x2)^2 + (y1-y2)^2)
                dis_mat[i][j] = tmp  # 将计算出的距离存入距离矩阵
        return dis_mat  # 返回完整的距离矩阵

    # 计算一条路径的长度
    def compute_pathlen(self, path, dis_mat):  # 计算单条路径的总长度（包含回到起点的距离）
        a = path[0]  # 路径的第一个城市
        b = path[-1]  # 路径的最后一个城市
        result = dis_mat[a][b]  # 初始化为从最后一个城市回到第一个城市的距离（形成回路）
        for i in range(len(path) - 1):  # 遍历路径中相邻城市对
            a = path[i]  # 当前城市
            b = path[i + 1]  # 下一个城市
            result += dis_mat[a][b]  # 累加相邻城市间的距离
        return result  # 返回路径总长度

    # 计算一个群体的长度
    def compute_paths(self, paths):  # 计算蚁群中所有蚂蚁路径的长度
        result = []  # 初始化结果列表
        for one in paths:  # 遍历每只蚂蚁的路径
            length = self.compute_pathlen(one, self.dis_mat)  # 计算该蚂蚁路径的总长度
            result.append(length)  # 将长度加入结果列表
        return result  # 返回所有路径长度的列表

    # 更新信息素
    def update_Tau(self):  # 更新信息素矩阵：挥发旧信息素 + 增加新信息素
        delta_tau = np.zeros([self.num_city, self.num_city])  # 初始化信息素增量矩阵为全零
        paths = self.compute_paths(self.Table)  # 计算当前蚁群中所有蚂蚁的路径长度
        for i in range(self.m):  # 遍历每只蚂蚁
            for j in range(self.num_city - 1):  # 遍历蚂蚁路径上相邻城市对（除首尾外）
                a = self.Table[i][j]  # 路径上的当前城市
                b = self.Table[i][j + 1]  # 路径上的下一个城市
                delta_tau[a][b] = delta_tau[a][b] + self.Q / paths[i]  # 信息素增量 = Q / 路径长度（路径越短，增量越大）
            a = self.Table[i][0]  # 路径的第一个城市
            b = self.Table[i][-1]  # 路径的最后一个城市
            delta_tau[a][b] = delta_tau[a][b] + self.Q / paths[i]  # 首尾相连的边也增加信息素（闭合回路）
        self.Tau = (1 - self.rho) * self.Tau + delta_tau  # 信息素更新公式：挥发后的旧信息素 + 新增信息素

    def aco(self):  # 蚁群算法主循环
        best_lenth = math.inf  # 初始化最优路径长度为无穷大
        best_path = None  # 初始化最优路径为None
        for cnt in range(self.iter_max):  # 主迭代循环，共执行iter_max次
            # 生成新的蚁群
            self.get_ants(self.num_city)  # 让所有蚂蚁构建完整路径，结果存储在self.Table中
            self.paths = self.compute_paths(self.Table)  # 计算每只蚂蚁的路径长度
            # 取该蚁群的最优解
            tmp_lenth = min(self.paths)  # 找出当前迭代中蚂蚁的最短路径长度
            tmp_path = self.Table[self.paths.index(tmp_lenth)]  # 找到对应最短路径的那只蚂蚁的访问序列
            # 可视化初始的路径
            if cnt == 0:  # 如果是第一次迭代
                init_show = self.location[tmp_path]  # 获取初始路径对应的城市坐标
                init_show = np.vstack([init_show, init_show[0]])  # 在末尾补上起点坐标，形成闭合回路（用于后续可视化，但此处未使用）
            # 更新最优解
            if tmp_lenth < best_lenth:  # 如果当前迭代的最短路径比历史最优更短
                best_lenth = tmp_lenth  # 更新全局最优路径长度
                best_path = tmp_path  # 更新全局最优路径序列
            # 更新信息素
            self.update_Tau()  # 根据本轮所有蚂蚁的路径更新信息素矩阵

            # 保存结果
            self.iter_x.append(cnt)  # 记录当前迭代次数到收敛曲线横坐标
            self.iter_y.append(best_lenth)  # 记录当前全局最优路径长度到收敛曲线纵坐标
            print(cnt,best_lenth)  # 打印当前迭代次数和全局最优路径长度，用于监控算法进展
        return best_lenth, best_path  # 返回找到的最优路径长度和最优路径序列

    def run(self):  # 对外接口：运行蚁群算法并返回结果
        best_length, best_path = self.aco()  # 调用aco()方法执行蚁群算法主循环
        return self.location[best_path], best_length  # 返回最优路径的坐标序列和路径长度


# 读取数据
def read_tsp(path):  # 读取TSP标准数据集文件（.tsp格式）
    lines = open(path, 'r').readlines()  # 按行读取文件全部内容
    assert 'NODE_COORD_SECTION\n' in lines  # 断言文件中包含节点坐标段标记，确保文件格式正确
    index = lines.index('NODE_COORD_SECTION\n')  # 找到节点坐标段的起始行索引
    data = lines[index + 1:-1]  # 截取坐标数据行（从标记行的下一行到倒数第二行）
    tmp = []  # 临时存储解析后的坐标数据
    for line in data:  # 遍历每一行坐标数据
        line = line.strip().split(' ')  # 去除首尾空白后按空格分割
        if line[0] == 'EOF':  # 如果遇到EOF（文件结束标记）
            continue  # 跳过该行
        tmpline = []  # 存储当前行解析出的数值
        for x in line:  # 遍历分割后的每个元素
            if x == '':  # 如果元素为空字符串（多个连续空格导致）
                continue  # 跳过空元素
            else:
                tmpline.append(float(x))  # 将有效数据转换为浮点数并加入临时行列表
        if tmpline == []:  # 如果当前行解析结果为空
            continue  # 跳过空行
        tmp.append(tmpline)  # 将解析出的坐标行加入结果列表
    data = tmp  # 将解析结果赋给data
    return data  # 返回坐标数据列表


data = read_tsp('./搜索算法与机器学习/homework/作业1/TSP_collection/data/st70.tsp')  # 读取st70标准TSP数据集（70个城市）

data = np.array(data)  # 将数据转换为numpy数组方便后续处理
data = data[:, 1:]  # 去掉第一列（城市编号），只保留x和y坐标
# 加上一行因为会回到起点
show_data = np.vstack([data, data[0]])  # 在数据末尾添加第一行坐标，用于可视化闭合路径

aco = ACO(num_city=data.shape[0], data=data.copy())  # 创建ACO对象，传入城市数量和坐标数据
Best_path, Best = aco.run()  # 运行蚁群算法，获取最优路径坐标和最优路径长度
print(Best)  # 打印最优路径的总长度
Best_path = np.vstack([Best_path, Best_path[0]])  # 在最优路径末尾添加起点坐标，形成闭合回路
plt.plot(Best_path[:, 0], Best_path[:, 1])  # 绘制最优路径：横坐标为x，纵坐标为y
plt.title('st70:蚁群算法规划结果')  # 设置图表标题
plt.show()  # 显示绘制的路径图
