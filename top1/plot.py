import re
import matplotlib.pyplot as plt


if __name__ == '__main__':
    loss = []
    le = 0
    for i in [1]:
        f = open('./data/{}.txt'.format(i))
        for line in f.readlines():
            # 2021-12-01 04:20:06.014 [INFO] 59064 m7_lstm_3inputs_age epoch: 1 batch: 9300 loss: 1.1561301946640015
            if 'batch' in str(line):
                res = re.findall(r"\d+\.?\d*", line)
                # batch.append(float(res[len(res) - 2]))
                loss.append(float(res[len(res) - 1]))
        x = [i for i in range(len(loss))]
        plt.plot(x, loss, color='lightcoral', label='Input Dimension = {}'.format(6))
        le = len(loss)
    loss = []
    for i in [0]:
        f = open('./data/{}.txt'.format(i))
        for line in f.readlines():
            # 2021-12-01 04:20:06.014 [INFO] 59064 m7_lstm_3inputs_age epoch: 1 batch: 9300 loss: 1.1561301946640015
            if 'batch' in str(line):
                res = re.findall(r"\d+\.?\d*", line)
                loss.append(float(res[len(res) - 1]))
                if len(loss) > le:
                    break
        x = [i for i in range(len(loss))]
        plt.plot(x, loss, color='skyblue', label='Input Dimension = {}'.format(3))


    plt.title('Loss vs. Steps')
    plt.xlabel('LSTM-Transformers Dimensions(Rangers)')
    plt.ylabel('Training Loss (In Total)')
    plt.legend()
    plt.show()
