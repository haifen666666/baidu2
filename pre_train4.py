import pandas as pd
import numpy as np
from datetime import datetime

#对访问时间贴上时间标签，区分平时、周末、法定假日;上午、下午、晚上、深夜
def put_time_tag():
    visit = pd.read_hdf('visit_quarter.h5')
    visit = visit.iloc[100000000:150000000,:] #分两次操作  前5000万条  和后面的
    visit['date'] = visit['year'] * 10000 + visit['month'] * 100 + visit['day']
    visit['date'] = visit['date'].astype('str')
    print(1)
    visit['week'] = visit['date'].apply(lambda x: datetime.strftime(datetime.strptime(x, '%Y%m%d'), '%w'))#需要大概10分钟
    print(2)
    visit['week'] = visit['week'].astype('int')
    print(3)
    visit['date'] = visit['date'].astype('int')
    visit.rename(columns={'state':'time_frame'},inplace=True)
    print(4)

    visit['date_cate'] = [0] * visit.shape[0]
    festival = [20181001,20181002,20181003,20181004,20181005,20181006,20181007,
                20181230,20181231,20190101,20190204,20190205,20190206,20190207,
                20190208,20190209,20190210]
    festival2 = [20181225,20190128,20190219] #分别是圣诞节，小年，元宵节 不放假的节日
    print(5)
    visit.loc[(visit['week']==0) | (visit['week']==6),'date_cate'] = 1
    print(6)
    visit.loc[visit['date'].isin(festival),'date_cate'] = 2
    print(7)
    visit.loc[visit['date'] == 20181229,'date_cate'] = 0 #这天虽然周六 但是调休
    visit.loc[visit['date'].isin(festival2),'date_cate'] = 3
    print(8)

    del visit['year']
    del visit['month']
    del visit['day']
    visit.to_hdf('visit_quarter_new3.h5',mode='w',key='tt',complevel=6,complib='blosc') #这里需要区分两次的文件，然后在终端合并

#获取统计特征，返回list
def get_statistic_features(df,theory_max) ->list:
    nums_list = list(df['nums'])
    length = len(nums_list)
    #如果理论上应该有7天的数据，实际采样只有5天有，则相差的两天手动补0
    if length != theory_max:
        differ = theory_max - length
        nums_list.extend([0]*differ)
    nums_array = np.array(nums_list)

    max = np.max(nums_array)
    min = np.min(nums_array)
    mean = np.mean(nums_array)
    median = np.median(nums_array)
    std = np.std(nums_array)
    var = np.var(nums_array)
    return [max,min,mean,median,std,var]

#按天统计特征,将每个小时人数相加
def get_day_info(df):
    df2 = df.groupby('date',as_index=False)['numbers'].agg({'nums':'sum'})
    return df2

def get_morning_info(df):
    df2 = df[df['time_frame'] == 1]
    df3 = df2.groupby('date',as_index=False)['numbers'].agg({'nums':'sum'})
    return df3

def get_afternoon_info(df):
    df2 = df[df['time_frame'] == 2]
    df3 = df2.groupby('date',as_index=False)['numbers'].agg({'nums':'sum'})
    return df3

def get_evening_info(df):
    df2 = df[df['time_frame'] == 3]
    df3 = df2.groupby('date',as_index=False)['numbers'].agg({'nums':'sum'})
    return df3

def get_night_info(df):
    df2 = df[df['time_frame'] == 4]
    df3 = df2.groupby('date',as_index=False)['numbers'].agg({'nums':'sum'})
    return df3


#全天、上午、下午...不同时段的统计特征 type为日子类型 周末、节假日等
#平时0 周末1 节假日2 无假期节假日3  不区分4  年前一周5  年中一周6  年后一周7
def get_dif_cate_features(df,cate):
    full_record = [116,46,17,3,182,7,7,7]  #每种类别理论上应该出现的次数 比如平时总共有116天
    theory_max = full_record[cate] #当前类型理论上的最大值

    morning = get_morning_info(df)
    afternoon = get_afternoon_info(df)
    evening = get_evening_info(df)
    night = get_night_info(df)

    morning_features = get_statistic_features(morning,theory_max)
    afternoon_features = get_statistic_features(afternoon,theory_max)
    evening_features = get_statistic_features(evening,theory_max)
    night_features = get_statistic_features(night,theory_max)
    return morning_features+afternoon_features+evening_features+night_features

#获取所有训练集特征
def get_train_features():
    visit = pd.read_hdf('visit_quarter_new.h5')
    visit_area = visit.groupby('area_id') #按照area_id进行分组
    length = len(visit_area)  #统计共有多少分组
    index = 0  #记录当前处理的是第多少个分组
    print(length)
    features = []
    for area_id,area in visit_area:
        print(length,index)  #显示总文件数和已处理文件数
        index += 1
        #获得该区域平时、周末、节假日、所有天的统计信息
        common = area[area['date_cate'] == 0]
        common_features = get_dif_cate_features(common, 0)
        weekend = area[area['date_cate'] == 1]
        weekend_features = get_dif_cate_features(weekend, 1)
        festival = area[area['date_cate'] == 2]
        festival_features = get_dif_cate_features(festival, 2)
        festival2 = area[area['date_cate'] == 3]
        festival2_features = get_dif_cate_features(festival2, 3)
        allday_features = get_dif_cate_features(area, 4) #这里不区分平时、节假日，全部在一起统计,类型计为4
        #获取该区域在年前、年中、年后的统计信息
        spring_before = area[(area['date']>=20190128) & (area['date']<=20190203)]
        spring_ing = area[(area['date']>=20190204) & (area['date']<=20190210)]
        spring_after = area[(area['date']>=20190211) & (area['date']<=20190217)]
        spring_before_features = get_dif_cate_features(spring_before, 5) #年前.年中.年后分别标记为5,6,7
        spring_ing_features = get_dif_cate_features(spring_ing, 6)
        spring_after_features = get_dif_cate_features(spring_after, 7)

        cur_features = common_features+weekend_features+festival_features+festival2_features+\
                       allday_features+spring_before_features + spring_ing_features + spring_after_features

        cur_features.append(area_id)
        features.append(cur_features)
    #column是所有的列名 0-7分别代表8种date_cate  平时、周末、节假日等，ABCDE分别代表全天、上午、下午、晚上、夜里, +全天活跃时、共计出现天数;
    column = ['q_maxB0','q_minB0','q_meanB0','q_medianB0','q_stdB0','q_varB0',
              'q_maxC0','q_minC0','q_meanC0','q_medianC0','q_stdC0','q_varC0',
              'q_maxD0','q_minD0','q_meanD0','q_medianD0','q_stdD0','q_varD0',
              'q_maxE0','q_minE0','q_meanE0','q_medianE0','q_stdE0','q_varE0',

              'q_maxB1','q_minB1','q_meanB1','q_medianB1','q_stdB1','q_varB1',
              'q_maxC1','q_minC1','q_meanC1','q_medianC1','q_stdC1','q_varC1',
              'q_maxD1','q_minD1','q_meanD1','q_medianD1','q_stdD1','q_varD1',
              'q_maxE1','q_minE1','q_meanE1','q_medianE1','q_stdE1','q_varE1',

              'q_maxB2','q_minB2','q_meanB2','q_medianB2','q_stdB2','q_varB2',
              'q_maxC2','q_minC2','q_meanC2','q_medianC2','q_stdC2','q_varC2',
              'q_maxD2','q_minD2','q_meanD2','q_medianD2','q_stdD2','q_varD2',
              'q_maxE2','q_minE2','q_meanE2','q_medianE2','q_stdE2','q_varE2',

              'q_maxB3','q_minB3','q_meanB3','q_medianB3','q_stdB3','q_varB3',
              'q_maxC3','q_minC3','q_meanC3','q_medianC3','q_stdC3','q_varC3',
              'q_maxD3','q_minD3','q_meanD3','q_medianD3','q_stdD3','q_varD3',
              'q_maxE3','q_minE3','q_meanE3','q_medianE3','q_stdE3','q_varE3',

              'q_maxB4','q_minB4','q_meanB4','q_medianB4','q_stdB4','q_varB4',
              'q_maxC4','q_minC4','q_meanC4','q_medianC4','q_stdC4','q_varC4',
              'q_maxD4','q_minD4','q_meanD4','q_medianD4','q_stdD4','q_varD4',
              'q_maxE4','q_minE4','q_meanE4','q_medianE4','q_stdE4','q_varE4',

              'q_maxB5','q_minB5','q_meanB5','q_medianB5','q_stdB5','q_varB5',
              'q_maxC5','q_minC5','q_meanC5','q_medianC5','q_stdC5','q_varC5',
              'q_maxD5','q_minD5','q_meanD5','q_medianD5','q_stdD5','q_varD5',
              'q_maxE5','q_minE5','q_meanE5','q_medianE5','q_stdE5','q_varE5',

              'q_maxB6','q_minB6','q_meanB6','q_medianB6','q_stdB6','q_varB6',
              'q_maxC6','q_minC6','q_meanC6','q_medianC6','q_stdC6','q_varC6',
              'q_maxD6','q_minD6','q_meanD6','q_medianD6','q_stdD6','q_varD6',
              'q_maxE6','q_minE6','q_meanE6','q_medianE6','q_stdE6','q_varE6',

              'q_maxB7','q_minB7','q_meanB7','q_medianB7','q_stdB7','q_varB7',
              'q_maxC7','q_minC7','q_meanC7','q_medianC7','q_stdC7','q_varC7',
              'q_maxD7','q_minD7','q_meanD7','q_medianD7','q_stdD7','q_varD7',
              'q_maxE7','q_minE7','q_meanE7','q_medianE7','q_stdE7','q_varE7',

              'area_id']

    features = pd.DataFrame(features,columns = column)
    features.to_csv('train_feature4.csv',index=False)


if __name__ == '__main__':
    #put_time_tag()
    get_train_features()


