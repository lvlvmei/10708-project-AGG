import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import svm
pd.options.mode.chained_assignment = None


def train_test_split_balanced(dataset, selected_features, test_size= 0.333, random_state= 42, **kind):
    selected_kind = kind['kind']
    kinds= dataset[selected_kind].value_counts().index.tolist()
    X_train = []
    X_test = []
    Y_train = []
    Y_test = []

    for num in range(len(kinds)):
        this_kind = kinds[num]
        sub_dataset = dataset[dataset[selected_kind] == this_kind]
        x = sub_dataset.loc[:, selected_features]
        y = sub_dataset[selected_kind]
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size, random_state=random_state)
        X_train.append(x_train)
        X_test.append(x_test)
        Y_train.append(y_train)
        Y_test.append(y_test)

    return pd.concat(X_train,axis=0), pd.concat(X_test, axis=0), pd.concat(Y_train, axis=0), pd.concat(Y_test, axis=0)


blue_5nn = pd.read_csv("data/blue_inn_5-o.txt", sep=',')
blue_5nn = blue_5nn.drop(['NAN'], axis=1)
blue_6nn = pd.read_csv("data/blue_inn_6-o.txt", sep=',')
blue_6nn = blue_6nn.drop(['NAN'], axis=1)
blue_7nn = pd.read_csv("data/blue_inn_7-o.txt", sep=',')
blue_7nn = blue_7nn.drop(['NAN'], axis=1)

blue_5_7_nn = pd.concat((blue_5nn, blue_7nn, blue_6nn),sort=True)
blue_5_7_nn['BIDIS'] =np.where(blue_5_7_nn['DIS'] >= 240, 1, 0)

def model_SVM_567(dataset, selected_features):
    X_train, X_test, y_train, y_test = train_test_split_balanced(dataset, selected_features, kind = 'BIDIS')
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    clf = svm.SVC(kernel='linear', C=1)
    clf.fit(X_train, y_train)
    return clf, scaler

BLUE_567_model = model_SVM_567(blue_5_7_nn, ['NRF', 'IRS', 'INN', 'irs1', 'rcp1', 'typ1'])

def BLUE_567_check(NRF, IRS, INN, irs1, rcp1, typ1):
    predict_set = [[NRF, IRS, INN, irs1, rcp1[0], typ1]]
    clf, scaler = BLUE_567_model
    predict_set = scaler.transform(predict_set)
    return int(float(clf.predict(predict_set)[0]))


check_for_red_dis = pd.read_table('/Users/lvmeizhong/Desktop/hippolyta-pro/data/all_check_red_disappear.txt', header=None, sep=' ')
check_for_red_dis.columns = ['SSN','GSN','IRS','INN','LF0','LF1','TP1','LF2','TP2','STP']
TYPE3 = check_for_red_dis[(check_for_red_dis.TP1 == 1) & (check_for_red_dis.TP2 == 1)]
cut_lable1 = [0, 1]
cut_bins1 =[0, 50, 251]
TYPE3.loc[:,'STP_L'] = pd.cut(TYPE3['STP'], bins=cut_bins1, labels = cut_lable1)
TYPE3.loc[:,'LFS'] = TYPE3.apply(lambda x: min(x['LF1'], x['LF2']), axis=1)
TYPE3.loc[:,'LFL'] = TYPE3.apply(lambda x: max(x['LF1'], x['LF2']), axis=1)

def model_SVM_T3(dataset, selected_features):
    X_train, X_test, y_train, y_test = train_test_split_balanced(dataset, selected_features, kind = 'STP_L', random_state=101)
    scaler = StandardScaler().fit(X_train)
    X_train = scaler.transform(X_train)
    clf = svm.SVC(kernel='rbf', C=1000, gamma=0.001)
    clf.fit(X_train, y_train)
    return clf, scaler

TYPE3_sub1_model = model_SVM_T3(TYPE3, ['LF0', 'IRS', 'LFS', 'LFL', 'INN'])
#
def TYPE3_check(LF0, IRS, LFS, LFL, INN):
    predict_set = [[LF0, IRS, LFS, LFL, INN]]
    #p1 LF0, p2 IRS
    clf, scaler = TYPE3_sub1_model
    predict_set = scaler.transform(predict_set)
    # 0 for disappear, 1 for not disappear
    return int(float(clf.predict(predict_set)[0]))

