import os
import time
import random

import h5py
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.optim as optim

from cnn_models import *


def to_categorical(train_labels):
    ret = np.zeros((len(train_labels), train_labels.max() + 1))
    ret[np.arange(len(ret)), train_labels] = 1
    return ret


def clip_by_tensor(t, t_min, t_max):
    result = (t >= t_min).float() * t + (t < t_min).float() * t_min
    result = (result <= t_max).float() * result + (result >
                                                   t_max).float() * t_max
    return result


def loss_mul(y_hat, label):
    # Cross-entropy loss
    y_hat = torch.softmax(y_hat, dim=1)
    log_prob = torch.log(clip_by_tensor(y_hat, 1e-10, 1))
    loss = -torch.sum(log_prob * label, dim=1)
    return loss


def data_loader(sub_dir, dataset, error_rate):
    print('Start loading the dataset...')
    if dataset == 'imagewoof':
        classes = [
            'Shih-Tzu', 'Rhodesian ridgeback', 'Australian terrier', 'Samoyed',
            'Dingo'
        ]
    elif dataset == 'Flowers':
        classes = ['daisy', 'dandelion', 'rose', 'sunflower', 'tulip']

    train_data = h5py.File(
        os.path.join(sub_dir, dataset, f'{dataset}_train.h5'), 'r')
    test_data = h5py.File(os.path.join(sub_dir, dataset, f'{dataset}_test.h5'),
                          'r')
    val_data = h5py.File(os.path.join(sub_dir, dataset, f'{dataset}_val.h5'),
                         'r')
    X_train = train_data['x_train'][:]
    Y_train = train_data['y_train'][:]
    X_test = test_data['x_test'][:]
    Y_test = test_data['y_test'][:]
    X_val = val_data['x_val'][:]
    Y_val = val_data['y_val'][:]
    y_train_e = Y_train.copy()
    x_train = np.swapaxes(np.swapaxes(X_train, 1, 3), 2, 3)
    y_train = to_categorical(Y_train)
    x_test = np.swapaxes(np.swapaxes(X_test, 1, 3), 2, 3)
    y_test = to_categorical(Y_test)
    x_val = np.swapaxes(np.swapaxes(X_val, 1, 3), 2, 3)
    y_val = to_categorical(Y_val)
    n_classes = y_test.shape[1]

    x_train /= 255  # Normalization
    x_val /= 255
    x_test /= 255
    #####################################################
    sign_num = int(len(x_train) * error_rate)  # error rate
    print(f'signed {sign_num} error samples')
    if os.path.exists(
            os.path.join(sub_dir, dataset,
                         f'sign_error_matrix_{error_rate}.txt')):
        error_matrix = np.loadtxt(
            os.path.join(sub_dir, dataset,
                         f'sign_error_matrix_{error_rate}.txt')).astype(int)
    else:
        error_idx = random.sample(range(0, len(x_train)), sign_num)
        error_matrix = np.vstack((error_idx, np.zeros_like(error_idx)))
        for i, j in enumerate(error_idx):
            ran = random.randint(0, n_classes - 1)
            while ran == y_train[j].argmax():
                ran = random.randint(0, n_classes - 1)
            y_train_e[j] = ran
            error_matrix[1, i] = ran
        np.savetxt(os.path.join(sub_dir, dataset,
                                f'sign_error_matrix_{error_rate}.txt'),
                   error_matrix,
                   fmt="%d")
    for i, j in enumerate(error_matrix[0, :]):
        y_train_e[j] = error_matrix[1, i]
    y_train_e = to_categorical(y_train_e)
    ######################################################
    print(f'Train：{len(x_train)} Validation: {len(x_val)} Test：{len(x_test)}')
    print('load dataset finished !!!!')

    return x_train, y_train_e, x_test, y_test, x_val, y_val, error_matrix, n_classes, classes


def manual_judgment(y_dot, x_train, y_train_e, error_matrix, classes):
    uncertain_idx = []

    num = 0
    for i in range(len(y_dot)):
        if not (np.equal(y_train_e[i], y_dot[i]).all()):
            uncertain_idx.append(i)
            num += 1
    print('The initial number of uncertain samples:', num)

    y_dot_human = y_dot.copy()
    img = np.swapaxes(np.swapaxes(x_train, 1, 2), 2, 3)
    i = 1
    uncertain_idx_sub = list(
        set(uncertain_idx).difference(set(error_matrix[0, :])))
    for idx in uncertain_idx_sub:
        img0 = img[idx]
        plt.figure(figsize=(2.5, 3.5))
        plt.title(
            f'{i}/{len(uncertain_idx_sub)}  imprecise\n {np.argmax(y_dot,axis=1)[idx]}-{classes[np.argmax(y_dot,axis=1)[idx]]}'
        )
        plt.imshow(img0)
        plt.show()

        try:
            a = eval(input())
        except:
            print("input error")
            a = eval(input())

        if int(a) == 0:
            y_dot_human[idx, :] = y_train_e[idx, :]
        i += 1

    num = 0
    for i in range(len(y_dot_human)):
        if not (np.equal(y_train_e[i], y_dot_human[i]).all()):
            num += 1
    print('The number of uncertain samples after manual judgment:', num)

    return y_dot_human


def validation():
    net.eval()
    clip_num = val_clip_num
    bpoint = np.linspace(0, len(x_val), clip_num + 1).astype(int)
    avg_acc = 0
    avg_loss = 0
    for i in range(clip_num):
        images = torch.from_numpy(
            x_val[bpoint[i]:bpoint[i + 1]]).to(device).float()
        y = torch.from_numpy(y_val[bpoint[i]:bpoint[i + 1]]).to(device)
        y_hat = net(images)

        loss = loss_mul(y_hat, y).sum().detach().cpu().item()
        loss = float(loss) / len(images)
        avg_loss += loss
        acc = torch.eq(y_hat.argmax(dim=1),
                       y.argmax(dim=1)).sum().detach().cpu().numpy()
        acc = float(acc) / len(images)
        avg_acc += acc
    avg_acc /= clip_num
    avg_loss /= clip_num

    return avg_loss, acc


def test():
    net.eval()
    clip_num = test_clip_num
    bpoint = np.linspace(0, len(x_test), clip_num + 1).astype(int)
    av_acc = 0
    for i in range(clip_num):
        images = torch.from_numpy(
            x_test[bpoint[i]:bpoint[i + 1]]).to(device).float()
        y = torch.from_numpy(y_test[bpoint[i]:bpoint[i + 1]]).to(device)
        y_hat = net(images)
        train_acc = torch.eq(y_hat.argmax(dim=1),
                             y.argmax(dim=1)).sum().detach().cpu().numpy()
        acc = float(train_acc) / len(images)
        av_acc += acc
    av_acc /= clip_num

    return av_acc


def main():
    print('train start........')
    steps = x_train.shape[0]
    remaining = steps % batch_size
    fre = 0
    last_loss = 100
    for e in range(1, epoch + 1):
        num1 = 0
        num2 = 0
        y_dot = y_train_e.copy()
        train_acc = 0
        train_loss = 0
        val_loss_ = 0
        val_acc_ = 0
        start_time = time.time()
        st = 1 / (1 + np.exp(-3 / (beta - 1) * (e - 1))) - 0.5  #0.4525
        for batch in range(0, steps - remaining, batch_size):
            print('\r' + str(round(batch * 100 / (steps - remaining), 2)),
                  '% ',
                  end='',
                  flush=True)

            batch_x = x_train[batch:batch + batch_size]
            batch_y = y_train_e[batch:batch + batch_size]
            images = torch.from_numpy(batch_x).to(device).float()

            net.train()
            optimizer.zero_grad()
            y_hat = net(images)

            ############################################## update y_dot
            y_hat_np = y_hat.detach().cpu().numpy()
            y_np = batch_y.copy()

            y_max_idx = np.argsort(y_np, axis=1)[:, -1]
            y_hat_max1_idx = np.argsort(y_hat_np, axis=1)[:, -1]
            y_hat_max2_idx = np.argsort(y_hat_np, axis=1)[:, -2]
            y_max = []
            y_hat_max1 = []
            y_hat_max2 = []
            for i in range(batch_size):
                y_max.append(y_hat_np[i][y_max_idx[i]])
                y_hat_max1.append(y_hat_np[i][y_hat_max1_idx[i]])
                y_hat_max2.append(y_hat_np[i][y_hat_max2_idx[i]])
            y_max = np.array(y_max)
            y_hat_max1 = np.array(y_hat_max1)
            y_hat_max2 = np.array(y_hat_max2)

            sigama = (y_hat_max1 - y_max)
            deta = (y_hat_max1 - y_hat_max2)

            for t in range(batch_size):
                if y_hat_max1_idx[t] != y_max_idx[t]:
                    mat = np.zeros(n_classes)
                    mat[y_hat_max1_idx[t]] = 1
                    y_dot[batch + t, :] = mat
                    num1 += 1
                elif deta[t] <= alpha:
                    mat = np.zeros(n_classes)
                    mat[y_hat_max2_idx[t]] = 1
                    y_dot[batch + t, :] = mat
                    num2 += 1

            ############################################## Calculate loss
            batch_y = torch.from_numpy(batch_y).to(device)
            batch_y_dot = torch.from_numpy(y_dot[batch:batch +
                                                 batch_size]).to(device)
            sigama = torch.from_numpy(sigama.astype(np.float32)).to(device)

            loss = loss_mul(y_hat, batch_y)
            loss_dot = loss_mul(y_hat, batch_y_dot)

            condition1 = (y_max_idx != y_hat_max1_idx).astype(int)
            condition2 = np.ones_like(deta) - np.heaviside(deta - alpha, 0)
            condition1 = torch.from_numpy(condition1).to(device)
            condition2 = torch.from_numpy(condition2).to(device)
            deta = torch.from_numpy(deta.astype(np.float32)).to(device)
            total_loss = loss - (condition1 * torch.exp(-sigama) +
                                 (condition2 *
                                  (torch.ones_like(condition1) - condition1) *
                                  torch.exp(-deta / alpha))) * st * loss_dot

            av_loss = total_loss.sum() / batch_size
            av_loss.backward()

            optimizer.step()
            av_loss = av_loss.detach().cpu().numpy()
            train_loss += float(av_loss)
            acc = torch.eq(y_hat.argmax(dim=1),
                           batch_y.argmax(dim=1)).sum().detach().cpu().numpy()
            acc = float(acc) / batch_size
            train_acc += acc

            # validation
            val_loss, val_acc = validation()
            val_loss_ += val_loss
            val_acc_ += val_acc
        average_acc = train_acc / round((steps - remaining) / batch_size)
        average_loss = train_loss / round((steps - remaining) / batch_size)
        f_val_loss = val_loss_ / round((steps - remaining) / batch_size)
        f_val_acc = val_acc_ / round((steps - remaining) / batch_size)
        # test
        average_test_acc = test()

        end_time = time.time()
        print(
            f'Epoch {e} Train Loss={average_loss:.4f} Train Accuracy={average_acc:.4f}  Val Loss={f_val_loss:.4f}  Val Accuracy={f_val_acc:.4f}  {end_time-start_time:.2f}s  Test Accuracy={average_test_acc:.4f}  Multi-label sample: {num1}+{num2}={num1+num2}'
        )

        if last_loss <= f_val_loss + 0.0005:
            fre += 1
        else:
            fre = 0
        #if fre >= 3 and e >= 20:
        if e >= 25:
            print('train finished')
            break
        last_loss = f_val_loss

    return y_dot


if __name__ == '__main__':
    ########################################## initialization
    sub_dir = 'model_dataset'
    dataset = 'imagewoof'
    #imagewoof  Flowers
    model = 'GoogLeNet'
    #GoogLeNet MobileNetV2 DenseNet169 VGG16 ResNet101 EfficientNetB0 ShuffleNetV2

    alpha = 0.01  # Threshold
    error_rate = 0.01

    epoch = 30
    batch_size = 32
    beta = 20  # stable
    val_clip_num = 50
    test_clip_num = 50
    initial_lrate = 0.001

    ########################################## load data
    x_train, y_train_e, x_test, y_test, x_val, y_val, error_matrix, n_classes, classes = data_loader(
        sub_dir, dataset, error_rate)

    ########################################## Training model and parameters
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'Training on {dataset} dataset using {device} by {model}')

    if model == 'ResNet101':
        net = ResNet101(num_classes=n_classes)
    elif model == 'VGG16':
        net = vgg(model_name="VGG16", num_classes=n_classes, init_weights=True)
    elif model == 'MobileNetV2':
        net = MobileNetV2(num_classes=n_classes)
    elif model == 'GoogLeNet':
        net = GoogLeNet(num_classes=n_classes)
    elif model == 'DenseNet169':
        net = DenseNet169(num_classes=n_classes)
    elif model == 'EfficientNetB0':
        net = EfficientNetB0(num_classes=n_classes)
    elif model == 'ShuffleNetV2':
        net = ShuffleNetV2(net_size=1, num_classes=n_classes)

    net = net.to(device)
    optimizer = optim.Adam(net.parameters(), lr=initial_lrate)

    ################ obtain the initial label with uncertain samples
    y_dot = main()
    np.savetxt(os.path.join(sub_dir, dataset, model,
                            f'y_dot_alpha-{alpha}_DCNN-1.txt'),
               y_dot,
               fmt="%d")

    ################ obtain the label with uncertain samples after manual judgment
    y_dot_human = manual_judgment(y_dot, x_train, y_train_e, error_matrix,
                                  classes)
    # Not uncertain samples input 0 ,otherwise input other integer
    np.savetxt(os.path.join(sub_dir, dataset, model,
                            f'y_dot_alpha-{alpha}_DCNN-2.txt'),
               y_dot_human,
               fmt="%d")
