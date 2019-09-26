from numpy import *


def create_separable_data(weights, b, num_lines):
    w = array(weights)
    num_features = len(weights)
    data = zeros((num_lines, num_features + 1))
    for i in range(num_lines):  # 从0开始到（num_lines-1）结束
        x = random.rand(1, num_features) * 20 - 10
        inner_product = sum(w * x) + b
        if inner_product <= 0:
            data[i] = append(x, -1)
        else:
            data[i] = append(x, 1)

    return data


def train_dual(data):
    num_lines = data.shape[0]
    num_features = data.shape[1]
    alpha = zeros((1, num_lines))
    w = zeros((1, num_features - 1))
    b = 0
    argument = zeros((1, num_features))
    i = 0
    separated = False
    while not separated and i < num_lines:
        for j in range(num_lines):
            w += alpha[0][j] * data[j][-1] * data[j, 0:-1]

        judge = data[i][-1] * (sum(w * data[i][0:-1]) + b)
        if judge <= 0:
            alpha[0][i] += 0.05
            b += 0.05 * data[i][-1]
            i = 0
            w = 0
            # 每次更新权重先让w=0，否则一直累加算法出错
            separated = False
        else:
            i += 1
    argument[0] = append(w, b)
    print(argument)
    return argument


def plot(data, argument):
    w = argument[0][0:-1]
    b = argument[0][-1]
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.set_title('Linear separable data set')
    plt.xlabel('x')
    plt.ylabel('y')
    idx_1 = where(data[:, -1] == 1)
    # 对于where函数来说，仅仅代表位置的array对外可见，数据的type不可见
    p1 = ax.scatter(data[idx_1, 0], data[idx_1, 1], marker='o', color='g', label=1, s=20)
    idx_2 = where(data[:, -1] == -1)
    p2 = ax.scatter(data[idx_2, 0], data[idx_2, 1], marker='x', color='r', label=2, s=20)
    x = w[0] / abs(w[0]) * 10
    y = w[1] / abs(w[1]) * 10
    ann = ax.annotate(u"", xy=(x, y), xytext=(0, 0), size=10, arrowprops=dict(arrowstyle="-|>"))
    # 用于对图上的注释，此处用来画法向量，注释内容以此为注释文本，注释点坐标
    # 注释文本坐标，注释标志（箭头）大小，注释样式
    ys = ((-12 * (-w[0]) - b) / w[1], (12 * (-w[0]) - b) / w[1])
    # 构建出线上点坐标
    ax.add_line(Line2D((-12, 12), ys, linewidth=1, color='blue'))
    # 用两个线上点的xy坐标作图，Line2D前者为点的横坐标，后者为点的纵坐标
    plt.legend(loc='upper right')
    # 放置图例
    plt.show()


data = create_separable_data([4, 3], 0, 100)
arg = train_dual(data)
plot(data, arg)
