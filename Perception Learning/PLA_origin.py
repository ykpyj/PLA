# array()就是数组，与列表和元组存储地址不同，array直接存储数据

from numpy import *


def create_separable_data(weights, b, num_lines):
    w = array(weights)
    num_features = len(weights)
    data_set = zeros((num_lines, num_features + 1))

    for i in range(num_lines):
        x = random.rand(1, num_features) * 20 - 10
        # 生成-10到10之间的随机数，一个num_features维的列表
        inner_product = sum(w * x) + b
        # w*x会返回一个 array()，内部存有列表，各个元素为两个列表元素之积
        if inner_product <= 0:
            data_set[i] = append(x, -1)
            # 除去最后一个元素为-1，其余元素于x相等
        else:
            data_set[i] = append(x, 1)

    return data_set
    # 完成对具有线性可分性的数据


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
    p1 = ax.scatter(data_set[idx_1, 0], data_set[idx_1, 1], marker='o', color='g', label=1, s=20)
    # scatter函数，用于取点
    idx_2 = where(data_set[:, 2] == -1)
    p2 = ax.scatter(data_set[idx_2, 0], data_set[idx_2, 1], marker='x', color='r', label=2, s=20)
    plt.legend(loc='upper right')
    plt.show()


def train(data_set, plot=False):
    num_lines = data_set.shape[0]
    # 获取data_set的行数和每行的维数
    num_features = data_set.shape[1]
    w = zeros((1, num_features - 1))
    b = 0  # 初始化分割线
    separated = False

    i = 0
    while not separated and i < num_lines:
        if data_set[i][-1] * (sum(w * data_set[i, 0:-1]) + b) <= 0:
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
        ax.set_title('Linear separable data set')
        plt.xlabel('X')
        plt.ylabel('Y')
        idx_1 = where(data_set[:, 2] == 1)
        p1 = ax.scatter(data_set[idx_1, 0], data_set[idx_1, 1], marker='o', color='g', label=1, s=20)
        idx_2 = where(data_set[:, 2] == -1)
        p1 = ax.scatter(data_set[idx_2, 0], data_set[idx_2, 1], marker='x', color='r', label=1, s=20)
        x = w[0][0] / abs(w[0][0]) * 10
        y = w[0][1] / abs(w[0][0]) * 10
        ann = ax.annotate(u"", xy=(x, y), xytext=(0, 0), size=20, arrowprops=dict(arrowstyle="-|>"))
        ys = (-12 * (-w[0][0]) / w[0][1], 12 * (-w[0][0]) / w[0][1])
        ax.add_line(Line2D((-12, 12), ys, linewidth=1, color='blue'))
        plt.legend(loc='upper right')
        plt.show()

    return w


data = create_separable_data([4, 3], 0, 100)
w = train(data, True)
