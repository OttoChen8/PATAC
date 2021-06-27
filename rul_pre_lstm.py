import numpy as np
import torch
from torch import nn
import pandas as pd
import os
import matplotlib.pyplot as plt
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

"""
Github: Yonv1943 Zen4 Jia1 hao2
https://github.com/Yonv1943/DL_RL_Zoo/blob/master/RNN
The source of training data 
https://github.com/L1aoXingyu/
code-of-learn-deep-learning-with-pytorch/blob/master/
chapter5_RNN/time-series/lstm-time-series.ipynb
"""
# length of SOH sequence
car1_size = 50
car2_size = 63
car3_size = 92
car4_size = 110
car5_size = 112
car6_size = 44
car7_size = 130
car8_size = 99
car9_size = 128


def run_train_lstm():
    '''load data'''
    carno = '9'
    loaded_data = load_soh_data(carNo=carno)
    # data = load_data()
    ## normalization
    train_size = int(len(loaded_data[:car9_size]))-1
    data = (loaded_data - loaded_data[:train_size, :].mean(axis=0)) / loaded_data[:train_size, :].std(axis=0)

    inp_dim = loaded_data.shape[1]
    out_dim = 1
    mid_dim = 12
    mid_layers = 1
    # batch_size = 48
    mod_dir = '.'


    data_x = data[:-1, :]
    data_y = data[+1:, 0]
    assert data_x.shape[1] == inp_dim

    # train_size = int(len(data_x) * 0.75)
    # train_size = data_x[:car1_size]
    batch_size = int(train_size * 0.8)

    train_x = data_x[:train_size]
    train_y = data_y[:train_size]
    train_x = train_x.reshape((train_size, inp_dim))
    train_y = train_y.reshape((train_size, out_dim))

    '''build model'''
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = RegLSTM(inp_dim, out_dim, mid_dim, mid_layers).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-2)

    '''train'''
    var_x = torch.tensor(train_x, dtype=torch.float32, device=device)
    var_y = torch.tensor(train_y, dtype=torch.float32, device=device)

    batch_var_x = list()
    batch_var_y = list()

    for i in range(batch_size):
        j = train_size - i
        batch_var_x.append(var_x[j:])
        batch_var_y.append(var_y[j:])

    from torch.nn.utils.rnn import pad_sequence
    batch_var_x = pad_sequence(batch_var_x)
    batch_var_y = pad_sequence(batch_var_y)

    with torch.no_grad():
        weights = np.tanh(np.arange(len(train_y)) * (np.e / len(train_y)))
        weights = torch.tensor(weights, dtype=torch.float32, device=device)

    print("Training Start")
    for e in range(768):
        out = net(batch_var_x)

        # loss = criterion(out, batch_var_y)
        loss = (out - batch_var_y) ** 2 * weights
        loss = loss.mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if e % 64 == 0:
            print('Epoch: {:4}, Loss: {:.5f}'.format(e, loss.item()))
    torch.save(net.state_dict(), '{}/net.pth'.format(mod_dir))
    print("Save in:", '{}/net.pth'.format(mod_dir))

    '''eval'''
    net.load_state_dict(torch.load('{}/net.pth'.format(mod_dir), map_location=lambda storage, loc: storage))
    net = net.eval()

    test_x = data_x.copy()
    test_x[(train_size+1):, 0] = 0
    test_x = test_x[:, np.newaxis, :]
    test_x = torch.tensor(test_x, dtype=torch.float32, device=device)

    '''simple way but no elegant'''
    # for i in range(train_size, len(data) - 2):
    #     test_y = net(test_x[:i])
    #     test_x[i, 0, 0] = test_y[-1]

    '''elegant way but slightly complicated'''
    eval_size = 1
    zero_ten = torch.zeros((mid_layers, eval_size, mid_dim), dtype=torch.float32, device=device)
    test_y, hc = net.output_y_hc(test_x[:train_size], (zero_ten, zero_ten))
    test_x[train_size + 1, 0, 0] = test_y[-1]
    for i in range(train_size + 1, len(data) - 2):
        test_y, hc = net.output_y_hc(test_x[i:i + 1], hc)
        test_x[i + 1, 0, 0] = test_y[-1]

    pred_y = test_x[1:, 0, 0]
    pred_y = pred_y.cpu().data.numpy()

    #变换回去
    pred_y = pred_y * loaded_data[:train_size, 0].std() + loaded_data[:train_size, 0].mean()
    data_y = data_y * loaded_data[:train_size, 0].std() + loaded_data[:train_size, 0].mean()

    # diff_y = (pred_y[train_size:] - data_y[train_size:-1])/data_y[train_size:-1]
    # l1_loss = np.mean(np.abs(diff_y))
    # l2_loss = np.mean(diff_y ** 2)
    # print("L1: {:.5f}    L2: {:.5f}".format(l1_loss, l2_loss))

    plt.figure()
    plt.scatter(train_size + 36, pred_y[train_size + 36], marker='*', s=100, zorder=30)
    print('after 180days:', pred_y[train_size + 36])
    plt.scatter(train_size + 71, pred_y[train_size + 71], marker='*', s=100, zorder=30)
    print('after 365days:', pred_y[train_size + 71])
    plt.plot(np.arange(train_size, pred_y.shape[0]), pred_y[train_size:], 'r', label='pred', linewidth=3, zorder=20)
    plt.plot(data_y[:train_size], 'b', label='real', alpha=0.3, linewidth=3)
    plt.plot([train_size, train_size], [data_y[train_size-1]-0.008, data_y[train_size-1]+0.008], linestyle='--', color='k', label='train | pred')
    plt.title('Car_{}'.format(carno), fontsize=15)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.xlabel('Timestamp', fontsize=15)
    plt.ylabel('RUL', fontsize=15)
    plt.legend(loc='best')
    # plt.savefig('lstm_reg.png')
    # plt.pause(4)
    plt.show()
    return


class RegLSTM(nn.Module):
    def __init__(self, inp_dim, out_dim, mid_dim, mid_layers):
        super(RegLSTM, self).__init__()

        self.rnn = nn.LSTM(inp_dim, mid_dim, mid_layers)  # rnn
        self.reg = nn.Sequential(
            nn.Linear(mid_dim, mid_dim),
            nn.Tanh(),
            nn.Linear(mid_dim, out_dim),
        )  # regression

    def forward(self, x):
        y = self.rnn(x)[0]  # y, (h, c) = self.rnn(x)

        seq_len, batch_size, hid_dim = y.shape
        y = y.view(-1, hid_dim)
        y = self.reg(y)
        y = y.view(seq_len, batch_size, -1)
        return y

    """
    PyCharm Crtl+click nn.LSTM() jump to code of PyTorch:
    Examples::
        >>> rnn = nn.LSTM(10, 20, 2)
        >>> input = torch.randn(5, 3, 10)
        >>> h0 = torch.randn(2, 3, 20)
        >>> c0 = torch.randn(2, 3, 20)
        >>> output, (hn, cn) = rnn(input, (h0, c0))
    """

    def output_y_hc(self, x, hc):
        y, hc = self.rnn(x, hc)  # y, (h, c) = self.rnn(x)

        seq_len, batch_size, hid_dim = y.size()
        y = y.view(-1, hid_dim)
        y = self.reg(y)
        y = y.view(seq_len, batch_size, -1)
        return y, hc


def load_data():
    data = pd.read_csv(r'D:\003.Projects\NCBDC2020\决赛\data\feature_1202\feature_car1.csv'.replace('\\', '/'))
    soh_date = pd.read_csv(
        r'D:\003.Projects\NCBDC2020\决赛\data\feature_days_from_InitialDate_car1.csv'.replace('\\', '/'))
    soh_seq = soh_date['0']
    datetime_seq = soh_date['datetime']
    import time
    datetime_float_seq = np.array([])
    for datetime in datetime_seq:
        # print(type(datetime))
        datetime_float = time.mktime(time.strptime(datetime, "%Y/%m/%d"))
        datetime_float_seq = np.append(datetime_float_seq, datetime_float)
    data.iloc[:, 0] = soh_seq
    # data['datetime'] = (datetime_float_seq - datetime_float_seq.mean())/datetime_float_seq.std()
    data['datetime'] = datetime_float_seq
    # data['0'] = data['0']/200000

    return data.values


def load_soh_data(carNo):
    import time
    from datetime import datetime
    soh_date = pd.read_csv(
        r'D:\003.Projects\NCBDC2020\决赛\data\add_one_year\add_one_year\new_feature_days_from_InitialDate_car{}.csv'.replace('\\', '/').format(carNo))
    soh_seq = soh_date['soh3'].to_frame(name='soh')  # 提取soh序列,有四种结果，分别是soh1，soh2，soh3，soh4四个字段
    datetime_seq = soh_date['datetime']  # 提取datetime序列，类型str
    year_seq = np.array([])
    month_seq = np.array([])
    day_seq = np.array([])
    datetime_float_seq = np.array([])
    for dt in datetime_seq:
        year = float(datetime.strptime(dt, "%Y/%m/%d").strftime('%Y'))
        month = float(datetime.strptime(dt, "%Y/%m/%d").strftime('%m'))
        day = float(datetime.strptime(dt, "%Y/%m/%d").strftime('%d'))
        datetime_float = time.mktime(time.strptime(dt, "%Y/%m/%d"))
        year_seq = np.append(year_seq, year)
        month_seq = np.append(month_seq, month)
        day_seq = np.append(day_seq, day)
        datetime_float_seq = np.append(datetime_float_seq, datetime_float)
    soh_seq['timestamp'] = datetime_float_seq
    soh_seq['day'] = day_seq
    soh_seq['month'] = month_seq
    soh_seq['year'] = year_seq
    return soh_seq.values[:, :2]


if __name__ == '__main__':
    run_train_lstm()
    # run_train_gru()
    # run_origin()
