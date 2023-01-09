import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from oldSC import *


def DecisionTree(t_x, t_y, cv_x, cv_y, tst_x, tst_y):
    flat_t_x = t_x.reshape(t_x.shape[0], t_x.shape[1] * t_x.shape[2])
    flat_cv_x = cv_x.reshape(cv_x.shape[0], cv_x.shape[1] * cv_x.shape[2])
    flat_tst_x = tst_x.reshape(tst_x.shape[0], tst_x.shape[1] * tst_x.shape[2])

    DT_Scores = []

    for i in range(1, 21):
        dt = DecisionTreeClassifier(criterion="gini", max_depth=i, min_samples_leaf=1, random_state=42)
        dt.fit(flat_t_x, t_y)
        DT_Scores.append(dt.score(flat_cv_x, cv_y))

    highest_DT = DT_Scores.index(max(DT_Scores))

    dt = DecisionTreeClassifier(criterion="gini", max_depth=10, min_samples_leaf=1, random_state=42)

    dt.fit(flat_t_x, t_y)
    test_score = dt.score(flat_tst_x, tst_y)
    return DT_Scores, test_score


def results(face_or_digit):
    if face_or_digit == "digit":
        tr_x, tr_y = digit_train()
        cv_x, cv_y = digit_valid()
        tst_x, tst_y = digit_test()
    else:
        tr_x, tr_y = face_train()
        cv_x, cv_y = face_valid()
        tst_x, tst_y = face_test()

    score_digit, tst_score_digit = DecisionTree(tr_x, tr_y, cv_x, cv_y, tst_x, tst_y)
    highest_DT = score_digit.index(max(score_digit))

    plt.figure(figsize=(10, 6))

    plt.plot(range(1, 21), score_digit, color='blue', linestyle='dashed', marker='o',
             markerfacecolor='purple', markersize=10)

    plt.title(f'Score vs. minimum number of samples required to be at a leaf node for {face_or_digit}')
    plt.xlabel('max_depth')
    plt.ylabel('Score')
    print(f"Maximum score for {face_or_digit} : ", score_digit[highest_DT] * 100, "at max_depth", highest_DT + 1)

    plt.show()

    print(f"Training accuracy for {face_or_digit}: ", tst_score_digit * 100, '%')


results("face")
results("digit")
