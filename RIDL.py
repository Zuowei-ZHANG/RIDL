import os
import time

import h5py
import numpy as np
import torch
import torch.optim as optim

from cnn_models import *
from evaluation import evaluate


def to_categorical(train_labels):
    ret = np.zeros((len(train_labels), train_labels.max() + 1))
    ret[np.arange(len(ret)), train_labels] = 1
    return ret


def uncertain_label(y_label1, y_label2, classes, n):

    transfer_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i == j:
                transfer_matrix[i, j] = i
    s = n
    for t in range(n):
        transfer_matrix[t, t + 1:n] = np.arange(s, s + n - t - 1)
        transfer_matrix[t + 1:n, t] = np.arange(s, s + n - t - 1)
        s += n - t - 1

    y1 = np.argmax(y_label1, axis=1)
    y2 = np.argmax(y_label2, axis=1)
    y = np.zeros((len(y1), int(n * (n + 1) / 2)))
    for k in range(len(y1)):
        y[k][int(transfer_matrix[y1[k], y2[k]])] = 1

    uncertain_classes = []
    for m in range(int(n * (n + 1) / 2)):
        if m < n:
            uncertain_classes.append(f'{classes[m]}')
        else:
            idx = np.argwhere(transfer_matrix == m)
            uncertain_classes.append(
                f'{classes[idx[0][0]]} or {classes[idx[0][1]]}')

    return transfer_matrix, y, uncertain_classes


def clip_by_tensor(t, t_min, t_max):
    result = (t >= t_min).float() * t + (t < t_min).float() * t_min
    result = (result <= t_max).float() * result + (result >
                                                   t_max).float() * t_max
    return result


def loss_mul(y_hat, label):
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

    sign_num = int(len(x_train) * error_rate)  # error rate
    print(f'signed {sign_num} error samples')

    error_matrix = np.loadtxt(
        os.path.join(sub_dir, dataset,
                     f'sign_error_matrix_{error_rate}.txt')).astype(int)

    for i, j in enumerate(error_matrix[0, :]):
        y_train_e[j] = error_matrix[1, i]
    y_train_e = to_categorical(y_train_e)

    print(f'Train：{len(x_train)} Validation: {len(x_val)} Test：{len(x_test)}')
    print('load dataset finished !!!!')

    return x_train, y_train_e, x_test, y_test, x_val, y_val, n_classes, classes


def train(batch_x, batch_y):
    net.train()

    images = torch.from_numpy(batch_x).to(device)
    labels = torch.from_numpy(batch_y).to(device)
    optimizer.zero_grad()
    output = net(images).to(device)
    #    loss
    loss = loss_mul(output, labels)

    bk_loss = loss.sum() / batch_size
    bk_loss.backward()
    #loss.backward()
    optimizer.step()

    total_loss = float(bk_loss.detach().cpu().numpy())
    acc = torch.eq(output.argmax(dim=1),
                   labels.argmax(dim=1)).sum().detach().cpu().numpy()
    acc = float(acc) / batch_size

    return total_loss, acc


def validation(x_val, y_val):
    net.eval()
    clip_num = val_clip_num
    cla = y_val.shape[1]
    pad = np.zeros((y_val.shape[0], int(cla * (cla + 1) / 2 - cla)))
    y_val = np.hstack((y_val, pad))
    bpoint = np.linspace(0, len(x_val), clip_num + 1).astype(int)
    avg_acc = 0
    avg_loss = 0
    for i in range(clip_num):
        images = torch.from_numpy(
            x_val[bpoint[i]:bpoint[i + 1]]).to(device).float()
        y_val_clip = torch.from_numpy(y_val[bpoint[i]:bpoint[i +
                                                             1]]).to(device)
        y_hat = net(images)

        loss = loss_mul(y_hat, y_val_clip).sum().detach().cpu().item()
        loss = float(loss) / len(images)
        avg_loss += loss
        acc = torch.eq(y_hat.argmax(dim=1),
                       y_val_clip.argmax(dim=1)).sum().detach().cpu().numpy()
        acc = float(acc) / len(images)
        avg_acc += acc
    avg_acc /= clip_num
    avg_loss /= clip_num

    return avg_loss, avg_acc


def test(x_test, y_test):
    net.eval()
    total_correct = 0
    imprecise = 0
    error_rate = 0
    clip_num = test_clip_num
    bpoint = np.linspace(0, len(x_test), clip_num + 1).astype(int)
    for i in range(clip_num):
        images = torch.from_numpy(
            x_test[bpoint[i]:bpoint[i + 1]]).to(device).float()
        y_hat = net(images)

        y_argmax = np.argmax(y_hat.detach().cpu().numpy(), axis=1)
        y_test_argmax = np.argmax(y_test[bpoint[i]:bpoint[i + 1]], axis=1)
        for i in range(len(y_argmax)):
            if y_argmax[i] >= n_classes:
                imprecise += 1
            elif y_test_argmax[i] == y_argmax[i]:
                total_correct += 1
            else:
                error_rate += 1

    acc = float(total_correct) / len(x_test)
    er = float(error_rate) / len(x_test)
    im = float(imprecise) / len(x_test)

    return acc, er, im


def classification(y_uncertain, x_train, y_train, x_test, y_test, x_val, y_val,
                   classes, case, alpha):

    global transfer_matrix, y, uncertain_classes, stop_num, lr_decay
    transfer_matrix, y, uncertain_classes = uncertain_label(y_train,
                                                            y_uncertain,
                                                            classes,
                                                            n=len(classes))
    steps = x_train.shape[0]
    remaining = steps % batch_size
    last_loss = 100
    fre = 0
    for e in range(1, epoch + 1):
        start_time = time.time()
        train_acc = 0
        train_loss = 0
        val_loss_ = 0
        val_acc_ = 0
        for batch in range(0, steps - remaining, batch_size):
            print('\r' + str(round(batch * 100 / (steps - remaining), 2)),
                  '%',
                  end='',
                  flush=True)

            batch_x = x_train[batch:batch + batch_size]
            batch_y = y[batch:batch + batch_size]
            total_loss, acc = train(batch_x, batch_y)
            train_loss += total_loss
            train_acc += acc

            val_loss, val_acc = validation(x_val, y_val)
            val_loss_ += val_loss
            val_acc_ += val_acc

        average_test_acc, test_er, test_im = test(x_test, y_test)
        scheduler.step()

        f_val_loss = val_loss_ / round((steps - remaining) / batch_size)
        f_val_acc = val_acc_ / round((steps - remaining) / batch_size)
        average_acc = train_acc / round((steps - remaining) / batch_size)
        average_loss = train_loss / round((steps - remaining) / batch_size)
        end_time = time.time()
        print(
            f' {case} Epoch {e} Train Loss={average_loss:.4f} Train Accuracy={average_acc:.4f}  Val Loss={f_val_loss:.4f}  Val Accuracy= {f_val_acc:.4f}  {end_time-start_time:.2f}s Test Accuracy={average_test_acc:.4f} Test error_rate={test_er:.4f} Test imprecise={test_im:.4f}'
        )

        if last_loss <= f_val_loss + 0.0005:
            fre += 1
        else:
            fre = 0
        if fre >= stop_num and e > lr_decay:
            print('train finished')
            break
        last_loss = f_val_loss

        torch.save(
            net.state_dict(),
            os.path.join(
                sub_dir, dataset, model,
                f'model_saved\\alpha-{alpha}_{case}_epoch-{e}_params.pkl'))

    net.eval()
    y_true = np.argmax(y_test, axis=1)
    y_pred = np.zeros_like(y_true)
    clipnum = test_clip_num
    clippoint = np.linspace(0, len(x_test), clipnum + 1).astype(int)
    for i in range(clipnum):
        test_block = torch.from_numpy(
            x_test[clippoint[i]:clippoint[i + 1]]).to(device).float()
        y_p = net(test_block)
        y_p_m = np.argmax(y_p.detach().cpu().numpy(), axis=1)
        y_pred[clippoint[i]:clippoint[i + 1]] = y_p_m
    Re, Ri, P, R, F1, bt = evaluate(y_pred, y_true, n_classes=n_classes)


def evaluate_model(x_test, y_test):
    net.eval()
    y_true = np.argmax(y_test, axis=1)
    y_pred = np.zeros_like(y_true)
    clipnum = test_clip_num
    clippoint = np.linspace(0, len(x_test), clipnum + 1).astype(int)
    for i in range(clipnum):
        test_block = torch.from_numpy(
            x_test[clippoint[i]:clippoint[i + 1]]).to(device).float()
        y_p = net(test_block)
        y_p_m = np.argmax(y_p.detach().cpu().numpy(), axis=1)
        y_pred[clippoint[i]:clippoint[i + 1]] = y_p_m
    Re, Ri, P, R, F1, bt = evaluate(y_pred, y_true, n_classes=n_classes)


if __name__ == '__main__':
    ########################################## initialization
    sub_dir = 'model_dataset'
    dataset = 'imagewoof'
    #imagewoof  Flowers
    model = 'GoogLeNet'
    #GoogLeNet MobileNetV2 DenseNet169 VGG16 ResNet101 EfficientNetB0 ShuffleNetV2
    case = 'DCNN-1'  # DCNN-1, DCNN-2
    
    alpha = 0.01
    error_rate = 0.01
    switch = 1  # Are pre-trained parameters used?   1:yes  0: no

    epoch = 30
    batch_size = 32
    val_clip_num = 50
    test_clip_num = 50
    stop_num = 5
    lr_decay = 20
    lrate = 0.001
    ########################################## load data

    x_train, y_train_e, x_test, y_test, x_val, y_val, n_classes, classes = data_loader(
        sub_dir, dataset, error_rate)
    new_n_classes = int(n_classes * (n_classes + 1) / 2)

    ##########################################  training model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'Training on {dataset} dataset using {device} by {model} ___{case}')

    if model == 'ResNet101':
        net = ResNet101(num_classes=new_n_classes)
    elif model == 'VGG16':
        net = vgg(model_name="VGG16",
                  num_classes=new_n_classes,
                  init_weights=True)
    elif model == 'MobileNetV2':
        net = MobileNetV2(num_classes=new_n_classes)
    elif model == 'GoogLeNet':
        net = GoogLeNet(num_classes=new_n_classes)
    elif model == 'DenseNet169':
        net = DenseNet169(num_classes=new_n_classes)
    elif model == 'EfficientNetB0':
        net = EfficientNetB0(num_classes=new_n_classes)
    elif model == 'ShuffleNetV2':
        net = ShuffleNetV2(net_size=1, num_classes=new_n_classes)

    ##################################
    if switch == 1:
        net.load_state_dict(
            torch.load(os.path.join(sub_dir, dataset, model,
                                    f'model_saved\\{model}_{case}_params.pkl'),
                       map_location=torch.device(device)))
        evaluate_model(x_test, y_test)
    elif switch == 0:
        if case == 'DCNN-1':
            y_uncertain = np.loadtxt(
                os.path.join(sub_dir, dataset, model,
                            f'y_dot_alpha-{alpha}_DCNN-1.txt')).astype(int)
        elif case == 'DCNN-2':
            y_uncertain = np.loadtxt(
                os.path.join(sub_dir, dataset, model,
                            f'y_dot_alpha-{alpha}_DCNN-2.txt')).astype(int) 
        net = net.to(device)
        optimizer = optim.Adam(net.parameters(), lr=lrate)
        #optimizer = optim.SGD(net.parameters(), lr=1e-3, momentum=0.9)
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, [lr_decay],
                                                   gamma=0.1)
        classification(y_uncertain, x_train, y_train_e, x_test, y_test, x_val,
                       y_val, classes, case, alpha)
