import pandas as pd

def make_new_train():
    train = pd.read_csv('train_feature1.csv')

    train['meanBA0'] = train['meanB0'] / (train['meanA0']+0.0001)  #这里0.0001是为了避免分母为0
    train['meanCA0'] = train['meanC0'] / (train['meanA0']+0.0001)
    train['meanDA0'] = train['meanD0'] / (train['meanA0']+0.0001)
    train['meanEA0'] = train['meanE0'] / (train['meanA0']+0.0001)

    train['meanBA1'] = train['meanB1'] / (train['meanA1']+0.0001)  #这里0.0001是为了避免分母为0
    train['meanCA1'] = train['meanC1'] / (train['meanA1']+0.0001)
    train['meanDA1'] = train['meanD1'] / (train['meanA1']+0.0001)
    train['meanEA1'] = train['meanE1'] / (train['meanA1']+0.0001)

    train['meanBA2'] = train['meanB2'] / (train['meanA2']+0.0001)  #这里0.0001是为了避免分母为0
    train['meanCA2'] = train['meanC2'] / (train['meanA2']+0.0001)
    train['meanDA2'] = train['meanD2'] / (train['meanA2']+0.0001)
    train['meanEA2'] = train['meanE2'] / (train['meanA2']+0.0001)

    train['meanBA3'] = train['meanB3'] / (train['meanA3']+0.0001)  #这里0.0001是为了避免分母为0
    train['meanCA3'] = train['meanC3'] / (train['meanA3']+0.0001)
    train['meanDA3'] = train['meanD3'] / (train['meanA3']+0.0001)
    train['meanEA3'] = train['meanE3'] / (train['meanA3']+0.0001)

    train['meanBA5'] = train['meanB5'] / (train['meanA5']+0.0001)  #这里0.0001是为了避免分母为0
    train['meanCA5'] = train['meanC5'] / (train['meanA5']+0.0001)
    train['meanDA5'] = train['meanD5'] / (train['meanA5']+0.0001)
    train['meanEA5'] = train['meanE5'] / (train['meanA5']+0.0001)

    train['meanBA6'] = train['meanB6'] / (train['meanA6']+0.0001)  #这里0.0001是为了避免分母为0
    train['meanCA6'] = train['meanC6'] / (train['meanA6']+0.0001)
    train['meanDA6'] = train['meanD6'] / (train['meanA6']+0.0001)
    train['meanEA6'] = train['meanE6'] / (train['meanA6']+0.0001)

    train['meanBA7'] = train['meanB7'] / (train['meanA7']+0.0001)  #这里0.0001是为了避免分母为0
    train['meanCA7'] = train['meanC7'] / (train['meanA7']+0.0001)
    train['meanDA7'] = train['meanD7'] / (train['meanA7']+0.0001)
    train['meanEA7'] = train['meanE7'] / (train['meanA7']+0.0001)

    train['percent_date_times_0'] = train['date_times_0'] / 116
    train['percent_date_times_1'] = train['date_times_0'] / 46
    train['percent_date_times_2'] = train['date_times_0'] / 17
    train['percent_date_times_4'] = train['date_times_0'] / 182

    train['meanA01'] = train['meanA0'] / (train['meanA1']+0.0001)
    train['meanB01'] = train['meanB0'] / (train['meanB1']+0.0001)
    train['meanC01'] = train['meanC0'] / (train['meanC1']+0.0001)
    train['meanD01'] = train['meanD0'] / (train['meanD1']+0.0001)
    train['meanE01'] = train['meanE0'] / (train['meanE1']+0.0001)

    train['meanA02'] = train['meanA0'] / (train['meanA2']+0.0001)
    train['meanB02'] = train['meanB0'] / (train['meanB2']+0.0001)
    train['meanC02'] = train['meanC0'] / (train['meanC2']+0.0001)
    train['meanD02'] = train['meanD0'] / (train['meanD2']+0.0001)
    train['meanE02'] = train['meanE0'] / (train['meanE2']+0.0001)

    train['meanA03'] = train['meanA0'] / (train['meanA3']+0.0001)
    train['meanB03'] = train['meanB0'] / (train['meanB3']+0.0001)
    train['meanC03'] = train['meanC0'] / (train['meanC3']+0.0001)
    train['meanD03'] = train['meanD0'] / (train['meanD3']+0.0001)
    train['meanE03'] = train['meanE0'] / (train['meanE3']+0.0001)

    train['meanA13'] = train['meanA1'] / (train['meanA3']+0.0001)
    train['meanB13'] = train['meanB1'] / (train['meanB3']+0.0001)
    train['meanC13'] = train['meanC1'] / (train['meanC3']+0.0001)
    train['meanD13'] = train['meanD1'] / (train['meanD3']+0.0001)
    train['meanE13'] = train['meanE1'] / (train['meanE3']+0.0001)

    train['meanA05'] = train['meanA0'] / (train['meanA5']+0.0001)
    train['meanB05'] = train['meanB0'] / (train['meanB5']+0.0001)
    train['meanC05'] = train['meanC0'] / (train['meanC5']+0.0001)
    train['meanD05'] = train['meanD0'] / (train['meanD5']+0.0001)
    train['meanE05'] = train['meanE0'] / (train['meanE5']+0.0001)

    train['meanA06'] = train['meanA0'] / (train['meanA6']+0.0001)
    train['meanB06'] = train['meanB0'] / (train['meanB6']+0.0001)
    train['meanC06'] = train['meanC0'] / (train['meanC6']+0.0001)
    train['meanD06'] = train['meanD0'] / (train['meanD6']+0.0001)
    train['meanE06'] = train['meanE0'] / (train['meanE6']+0.0001)

    train['meanA07'] = train['meanA0'] / (train['meanA7']+0.0001)
    train['meanB07'] = train['meanB0'] / (train['meanB7']+0.0001)
    train['meanC07'] = train['meanC0'] / (train['meanC7']+0.0001)
    train['meanD07'] = train['meanD0'] / (train['meanD7']+0.0001)
    train['meanE07'] = train['meanE0'] / (train['meanE7']+0.0001)

    train.to_csv('train_feature2.csv',index=False)

make_new_train()




