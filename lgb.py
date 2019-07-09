import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import log_loss
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle


def train(filename):
    train = pd.read_csv(filename)
    '''
    train = train.loc[((train.answer == 1) & (train.day_mean4 < 200)) | ((train.answer == 2) & (train.day_mean4 < 300))|
              ((train.answer == 3) & (train.day_mean4 < 120)) | ((train.answer == 4) & (train.day_mean4 < 120)) |
              ((train.answer == 5) & (train.day_mean4 < 60)) | ((train.answer == 6) & (train.day_mean4 < 160)) |
              ((train.answer == 7) & (train.day_mean4 < 500)) | ((train.answer == 8) & (train.day_mean4 < 220)) |
              ((train.answer == 9) & (train.day_mean4 < 500))]
              '''
    print(train.shape)

    train = shuffle(train)
    train = shuffle(train)
    train['answer'] = train['answer'] - 1
    answer = train[['answer']]

    del train['answer']
    del train['area_id']
    x_train, x_test, y_train, y_test = train_test_split(train,answer, test_size = 0.2, random_state = 100)

    lgb_train = lgb.Dataset(x_train, y_train, free_raw_data=False)
    lgb_eval = lgb.Dataset(x_test, y_test, reference=lgb_train, free_raw_data=False)
    lgb_all = lgb.Dataset(train,answer,free_raw_data=False)
    params = {'boosting_type': 'gbdt',
              'objective': 'multiclass',
              'metrics': 'multi_logloss',
              'nthread': 10,   #线程数
              'num_class': 9,  #类别数
              'learning_rate': 0.02,
              'num_leaves': 150,
              'max_depth': 16,
              'max_bin': 200, #将feature存入bin的最大值，越大越准，最大255,默认值255
              'subsample_for_bin': 50000, #用于构建直方图数据的数量，默认值为20000,越大训练效果越好，但速度会越慢
              'subsample': 0.8, #子采样，为了防止过拟合
              'subsample_freq': 1,  #重采样频率,如果为正整数，表示每隔多少次迭代进行bagging
              'colsample_bytree': 0.8, #每棵随机采样的列数的占比,一般取0.5-1
              'reg_alpha': 0.2, #L1正则化项，越大越保守
              'reg_lambda': 0, #L2正则化项，越大越保守
              'min_split_gain': 0.0,
              'min_child_weight': 1, #默认值为1,越大越能避免过拟合，建议使用CV调整
              'min_child_samples': 10, #alias：min_data_in_leaf 越大越能避免树过深，避免过拟合，但是可能欠拟合 需要CV调整
              'scale_pos_weight': 1, # 类别不均衡时设定,
              }
    num_round = 20000
    
    model_train = lgb.train(params,
                    lgb_train,
                    num_round,
                    categorical_feature=['top1_0','top2_0','top3_0','top1_1','top2_1','top3_1',
                                         'top1_2','top2_2','top3_2','top1_3','top2_3','top3_3',
                                         'top1_4','top2_4','top3_4','top1_5','top2_5','top3_5',
                                         'top1_6','top2_6','top3_6','top1_7','top2_7','top3_7',
                                         'day_month_rank1','day_month_rank2','day_month_rank3',
                                         'day_month_rank4','day_month_rank5','day_month_rank6',
                                         'day_cat_rank0','day_cat_rank1','day_cat_rank2',
                                         'day_cat_rank3','day_cat_rank4','day_cat_rank5','day_cat_rank6','day_cat_rank7'],
                    valid_sets=lgb_eval, early_stopping_rounds = 100)
    #线下验证正确率
    pred =model_train.predict(x_test)
    pred = [list(x).index(max(x)) for x in pred]
    actually = list(y_test['answer'])
    print(len(pred),len(actually))
    print(type(pred),type(actually))
    observe = pd.DataFrame()
    observe['pred'] = pred
    observe['actually'] = actually
    observe['equal'] = observe['pred'] - observe['actually']
    right = observe[observe['equal'] == 0].shape[0]
    all = observe.shape[0]
    print(right/all)
    print(model_train.best_iteration)
    model = lgb.train(params, lgb_all, model_train.best_iteration,
                     categorical_feature=['top1_0','top2_0','top3_0','top1_1','top2_1','top3_1',
                                         'top1_2','top2_2','top3_2','top1_3','top2_3','top3_3',
                                         'top1_4','top2_4','top3_4','top1_5','top2_5','top3_5',
                                         'top1_6','top2_6','top3_6','top1_7','top2_7','top3_7',
                                         'day_month_rank1','day_month_rank2','day_month_rank3',
                                         'day_month_rank4','day_month_rank5','day_month_rank6',
                                         'day_cat_rank0','day_cat_rank1','day_cat_rank2',
                                         'day_cat_rank3','day_cat_rank4','day_cat_rank5','day_cat_rank6','day_cat_rank7'])
    model.save_model('lgb1.model') # 用于存储训练出的模型
    dfFeature = pd.DataFrame()
    dfFeature['featureName'] = model.feature_name()
    dfFeature['score'] = model.feature_importance()
    dfFeature.to_csv('featureImportance1.csv')

def predict():
    model = lgb.Booster(model_file = 'lgb1.model') #init model
    test = pd.read_csv('test_feature6.csv')
    test.sort_values('area_id',inplace=True)

    submit = test[['area_id']]
    del test['area_id']
    pred = model.predict(test)

    #得到预测的排序结果，保存下来，用于做结果层面的特征融合
    lgb_sort = pd.DataFrame(pred,columns=['1','2','3','4','5','6','7','8','9'])
    lgb_sort.to_csv('lgb_sort.csv',index=False)

    #选出概率最大的一个，拿到对应label，作为类标号
    pred = [list(x).index(max(x)) for x in pred]
    submit['answer'] = pred

    #按照提交格式，处理数据后保存
    submit['answer'] = submit['answer'] + 1
    submit['area_id'] = submit['area_id'].astype('str')
    submit['answer'] = submit['answer'].astype('str')

    def fill_0(x):
        tt = (6 - len(x.area_id))* '0'
        x.area_id = tt + x.area_id
        dd = (3 - len(x.answer)) * '0'
        x.answer = dd + x.answer
        return x

    submit = submit.apply(fill_0,axis=1)  #填0
    submit.to_csv('observe.csv',index=False)  #为了便于查看答案的分布
    submit.to_csv('submit.txt',sep='\t',index=None,header=None)


if __name__ == '__main__':
    train('train_feature6_.csv')
    #predict()



'''
'day_month_rank1','day_month_rank2','day_month_rank3',
'day_month_rank4','day_month_rank5','day_month_rank6',
'day_cat_rank0','day_cat_rank1','day_cat_rank2',
'day_cat_rank3','day_cat_rank4','day_cat_rank5','day_cat_rank6','day_cat_rank7'
'''
