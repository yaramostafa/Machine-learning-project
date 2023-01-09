from sklearn.neural_network import MLPClassifier
import warnings
import matplotlib.pyplot as plt
from oldSC import *
import warnings

warnings.filterwarnings('ignore')


def MLP(t_x, t_y, cv_x, cv_y, tst_x, tst_y):
    flat_t_x = t_x.reshape(t_x.shape[0], t_x.shape[1] * t_x.shape[2])
    flat_cv_x = cv_x.reshape(cv_x.shape[0], cv_x.shape[1] * cv_x.shape[2])
    flat_tst_x = tst_x.reshape(tst_x.shape[0], tst_x.shape[1] * tst_x.shape[2])

    # List to store accuracy scores in
    MLP_score = []
    for i in [0.000001, 0.00001, 0.0001, 0.001, 0.01, 0.1, 1]:
        MLP = MLPClassifier(alpha=i, activation='relu', solver='adam')
        MLP.fit(flat_t_x, t_y)
        MLP_score.append(MLP.score(flat_cv_x, cv_y))

    highest_MLP = MLP_score.index(max(MLP_score))

    MLP = MLPClassifier(alpha=highest_MLP, activation='relu', solver='adam')
    MLP.fit(flat_t_x, t_y)
    test_score = MLP.score(flat_tst_x, tst_y)
    return MLP_score, test_score


def results(face_or_digit):
    if face_or_digit == "digit":
        tr_x, tr_y = digit_train()
        cv_x, cv_y = digit_valid()
        tst_x, tst_y = digit_test()
    else:
        tr_x, tr_y = face_train()
        cv_x, cv_y = face_valid()
        tst_x, tst_y = face_test()

    # fit
    # score_digit= SVM(tr_x, tr_y,cv_x,cv_y,tst_x,tst_y)
    # print(score_digit*100, '%')

    score_digit, tst_score_digit = MLP(tr_x, tr_y, cv_x, cv_y, tst_x, tst_y)
    highest_MLP = score_digit.index(max(score_digit))

    plt.figure(figsize=(10, 6))
    x = [0.000001, 0.00001, 0.0001, 0.001, 0.01, 0.1, 1]
    plt.plot(range(len(x)), score_digit, color='blue', linestyle='dashed', marker='o',
             markerfacecolor='purple', markersize=10)
    plt.xticks(range(len(x)), x)
    plt.title(f'Score vs. Alpha for {face_or_digit}')
    plt.xlabel('A')
    plt.ylabel('Score')
    print(f"Maximum score for {face_or_digit} : ", score_digit[highest_MLP] * 100, "at alpha =", x[highest_MLP])
    plt.show()

    print(f"Training accuracy for {face_or_digit}: ", tst_score_digit * 100, '%')


results("face")
results("digit")
