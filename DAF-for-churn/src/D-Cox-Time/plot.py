import re
import matplotlib.pyplot as plt


if __name__ == '__main__':

    f = open('./logs/D-Cox-Time/log')
    epoch = []
    x1 = range(0, 500)
    x2 = range(0, 500)
    train_loss = []
    val_loss = []
    for line in f.readlines():
        # 4:	[0s / 1s],		train_loss: 1.9679,	val_loss: 1.7287
        res = re.findall(r"\d+\.?\d*", line)
        train_loss.append(float(res[len(res) - 2]))
        val_loss.append(float(res[len(res) - 1]))
        # epoch.append((int(res[0]), float(res[len(res) - 2]), float(res[len(res) - 1])))
    x1 = range(0, len(train_loss))
    x2 = range(0, len(val_loss))
    y1 = train_loss
    y2 = val_loss
    plt.plot(x1, y1, 'r-', label='train_loss')
    plt.title('Loss vs. epoches')
    plt.plot(x2, y2, 'b-', label='val_loss')
    plt.xlabel('D-Cox-Time Epochs(Adams, No Early Stop)')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()
