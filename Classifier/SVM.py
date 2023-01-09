import matplotlib.pyplot as plt
from sklearn.svm import SVC
from oldSC import *


def SVM(t_x, t_y, cv_x, cv_y, tst_x, tst_y):
    flat_t_x = t_x.reshape(t_x.shape[0], t_x.shape[1] * t_x.shape[2])
    flat_cv_x = cv_x.reshape(cv_x.shape[0], cv_x.shape[1] * cv_x.shape[2])
    flat_tst_x = tst_x.reshape(tst_x.shape[0], tst_x.shape[1] * tst_x.shape[2])

    # List to store accuracy scores in
    SVM_Scores = []

    # an initial SVM model with rbf kernel and different values for c (Regularization parameter)
    for i in [0.01, 0.1, 1, 10, 100, 1000]:
        svm_c = SVC(kernel='poly', C=i, degree=2)
        svm_c.fit(flat_t_x, t_y)
        SVM_Scores.append(svm_c.score(flat_cv_x, cv_y))

    highest_SVM = SVM_Scores.index(max(SVM_Scores))

    svm_c = SVC(kernel='rbf', C=highest_SVM)
    svm_c.fit(flat_t_x, t_y)
    test_score = svm_c.score(flat_tst_x, tst_y)
    return SVM_Scores, test_score


# predictions = svm_linear.predict(flat_tst_x)
# confusion = metrics.confusion_matrix(y_true = tst_y, y_pred = predictions)
# accuracyScore = metrics.accuracy_score(y_true=tst_y, y_pred=predictions)
# return accuracyScore
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
    score_digit, tst_score_digit = SVM(tr_x, tr_y, cv_x, cv_y, tst_x, tst_y)
    highest_SVM = score_digit.index(max(score_digit))

    plt.figure(figsize=(10, 6))
    x = [0.01, 0.1, 1, 10, 100, 1000]
    plt.plot(range(len(x)), score_digit, color='blue', linestyle='dashed', marker='o',
             markerfacecolor='purple', markersize=10)
    plt.xticks(range(len(x)), x)
    plt.title(f'Score vs. cost factor(s) for {face_or_digit}')
    plt.xlabel('C')
    plt.ylabel('Score')
    print(f"Maximum score for {face_or_digit} : ", score_digit[highest_SVM] * 100, "at c", x[highest_SVM])

    plt.show()

    print(f"Training accuracy for {face_or_digit}: ", tst_score_digit * 100, '%')


results("face")
results("digit")
