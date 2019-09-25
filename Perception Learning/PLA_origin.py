# array()就是数组，与列表和元组存储地址不同，array直接存储数据

from numpy import *


def create_separable_data(weights, b, num_lines):
    w = array(weights)
    #array()相当于是列表的嵌套
    num_features = len(weights)
    data_set = zeros((num_lines, num_features + 1))
    # 多的1个元素用于存储是否为正确隔离的标志

    for i in range(num_lines):
        x = random.rand(1, num_features) * 20 - 10
        # 生成-10到10之间的随机数，一个num_features维的列表
        inner_product = sum(w * x) + b
        # w*x会返回一个 array()，内部存有列表，各个元素为两个列表元素之积
        if inner_product <= 0:
            data_set[i] = append(x, -1)
            # 除去最后一个元素为-1，其余元素与x相等
        else:
            data_set[i] = append(x, 1)

    return data_set
    # 完成对具有线性可分性数据生成


def data_plot(data_set):
    import matplotlib.pyplot as plt
    fig = plt.figure()
    ax = fig.add_subplot(111)
    # 1x1网格,第一子图
    ax.set_title('Linear separable data set')
    plt.xlabel('X')
    plt.ylabel('Y')
    labels = array(data_set[:, 2])
    idx_1 = where(data_set[:, 2] == 1)
    # 输出相等位置的索引（以位置序号表示）以及相应的数据类型
    p1 = ax.scatter(data_set[idx_1, 0], data_set[idx_1, 1], marker='o', color='g', label=1, s=20)
    # scatter函数，用于取点
    idx_2 = where(data_set[:, 2] == -1)
    p2 = ax.scatter(data_set[idx_2, 0], data_set[idx_2, 1], marker='x', color='r', label=2, s=20)
    plt.legend(loc='upper right')
    plt.show()


def train(data_set, plot=False):
    num_lines = data_set.shape[0]
    # 获取data_set的行数和列数
    num_features = data_set.shape[1]
    w = zeros((1, num_features - 1))
    b = 0  # 初始化分割线
    separated = False

    i = 0
    while not separated and i < num_lines:
        # 不分和i<两条件同时满足时运行
        if data_set[i][-1] * (sum(w * data_set[i, 0:-1]) + b) <= 0:
            # data_set[i,0:-1]返回一个拥有某行前两列数据的array[]
            w += data_set[i][-1] * data_set[i, 0:-1]
            b = b + data_set[i][-1]
            separated = False
            i = 0
            # 一旦有一个分类出错则全部重新来过
        else:
            i += 1
    if plot:
        import matplotlib.pyplot as plt
        from matplotlib.lines import Line2D
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        # 将ax作为子图的对象，便于后续调用
        ax.set_title('Linear separable data set')
        plt.xlabel('X')
        plt.ylabel('Y')
        idx_1 = where(data_set[:, 2] == 1)
        p1 = ax.scatter(data_set[idx_1, 0], data_set[idx_1, 1], marker='o', color='g', label=1, s=20)
        idx_2 = where(data_set[:, 2] == -1)
        p1 = ax.scatter(data_set[idx_2, 0], data_set[idx_2, 1], marker='x', color='r', label=1, s=20)
        x = w[0][0] / abs(w[0][0]) * 10
        y = w[0][1] / abs(w[0][0]) * 10
        ann = ax.annotate(u"", xy=(x, y), xytext=(0, 0), size=10, arrowprops=dict(arrowstyle="-|>"))
        # 用于对图上的注释，此处用来画法向量，注释内容以此为注释文本，注释点坐标
        # 注释文本坐标，注释标志（箭头）大小，注释样式
        ys = ((-12 * (-w[0][0])-b) / w[0][1], (12 * (-w[0][0])-b) / w[0][1])
        # 构建出线上点坐标
        ax.add_line(Line2D((-12, 12), ys, linewidth=1, color='blue'))
        # 用两个线上点的xy坐标作图
        plt.legend(loc='upper right')
        # 放置图例
        plt.show()

    return w


data= create_separable_data([4, 3], 0, 100)
w = train(data, True)

