import numpy as np
from sklearn import metrics


def uncertain_label(n):
    # Constructing the transfer matrix, n is the number of classes
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

    return transfer_matrix


def evaluate(y_pred, y_true, n_classes, gamma=0.8):
    A = 2  # meta class
    benefit_value = 0
    Re = 0
    Ri = 0
    transfer_matrix = uncertain_label(n_classes)
    #print(transfer_matrix)
    for i in range(len(y_pred)):
        if y_pred[i] < n_classes:
            if y_pred[i] == y_true[i]:
                benefit_value += 1
            else:
                Re += 1
        if y_pred[i] >= n_classes:
            if y_true[i] in np.argwhere(transfer_matrix == y_pred[i])[0, :]:
                benefit_value += (1 / A)**gamma
                Ri += 1
            else:
                Re += 1
    bt = round(benefit_value / len(y_pred), 4)
    Re = round(Re / len(y_pred), 4)
    Ri = round(Ri / len(y_pred), 4)

    matrix = metrics.confusion_matrix(y_true.reshape(-1), y_pred.reshape(-1))
    print(matrix)

    precision = 0
    recall = 0
    f_measure1 = 0
    fp = 0
    fn = 0
    for i in range(n_classes):
        TP = matrix[i, i]
        FP = matrix[:, i].sum() - matrix[i, i]
        FN = matrix[i, :].sum() - matrix[i, i]
        TN = len(y_pred) - TP - FP - FN

        fp += FP
        fn += FN

        if TP == 0 and FP == 0:
            P = 0
        else:
            P = TP / (TP + FP)
        if TP == 0 and FN == 0:
            R = 0
        else:
            R = TP / (TP + FN)
        if P == 0 and R == 0:
            F1 = 0
        else:
            F1 = 2 * P * R / (P + R)

        precision += P
        recall += R
        f_measure1 += F1
    e_precision = round(precision / n_classes, 4)
    e_recall = round(recall / n_classes, 4)
    e_f_measure1 = round(f_measure1 / n_classes, 4)

    if max(y_pred) < n_classes:

        print(
            f'Re:{Re} Ri:{Ri} Ra:{round(1-Re-Ri, 4)} P:{e_precision} R:{e_recall} F1:{e_f_measure1} Bt:{bt}'
        )
    else:
        print(
            f'Re:{Re} Ri:{Ri} Ra:{round(1-Re-Ri, 4)} E-P:{e_precision} E-R:{e_recall} E-F1:{e_f_measure1} Bt:{bt}'
        )

    return Re, Ri, e_precision, e_recall, e_f_measure1, bt
