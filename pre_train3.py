import pandas as pd
import numpy as np
from datetime import datetime

#对访问时间贴上时间标签，区分平时、周末、法定假日
def put_time_tag():
    visit = pd.read_hdf('visit_day.h5')
    visit['date'] = visit['year'] * 10000 + visit['month'] * 100 + visit['day']
    visit['date'] = visit['date'].astype('str')
    visit['week'] = visit['date'].apply(lambda x: datetime.strftime(datetime.strptime(x, '%Y%m%d'), '%w'))#需要大概10分钟
    visit['week'] = visit['week'].astype('int')
    visit['date'] = visit['date'].astype('int')

    visit['date_cate'] = [0] * visit.shape[0]
    festival = [20181001,20181002,20181003,20181004,20181005,20181006,20181007,
                20181230,20181231,20190101,20190204,20190205,20190206,20190207,
                20190208,20190209,20190210]
    festival2 = [20181225,20190128,20190219] #分别是圣诞节，小年，元宵节 不放假的节日
    visit.loc[(visit['week']==0) | (visit['week']==6),'date_cate'] = 1
    visit.loc[visit['date'].isin(festival),'date_cate'] = 2
    visit.loc[visit['date'] == 20181229,'date_cate'] = 0 #这天虽然周六 但是调休
    visit.loc[visit['date'].isin(festival2),'date_cate'] = 3

    del visit['year']
    del visit['month']
    del visit['day']
    visit.to_hdf('visit_day_new.h5',mode='w',key='tt',complevel=6,complib='blosc')

#获取统计特征，返回list
def get_statistic_features(df,theory_max) ->list:
    nums_list = list(df['numbers'])
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

#平时0 周末1 节假日2 无假期节假日3  不区分4  年前一周5  年中一周6  年后一周7
def get_dif_cate_features(df,cate):
    full_record = [116,46,17,3,182,7,7,7]  #每种类别理论上应该出现的次数 比如平时总共有116天
    theory_max = full_record[cate] #当前类型理论上的最大值
    day_features = get_statistic_features(df,theory_max) #获取一天的统计特征
    return day_features

#获取所有训练集特征
def get_train_features():
    visit = pd.read_hdf('visit_day_new.h5')
    visit_area = visit.groupby('area_id') #按照area_id进行分组
    length = len(visit_area)  #统计共有多少分组
    index = 0  #记录当前处理的是第多少个分组
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
    #column是所有的列名 0-7分别代表8种date_cate  平时、周末、节假日等
    column = ['day_max0','day_min0','day_mean0','day_median0','day_std0','day_var0',
              'day_max1','day_min1','day_mean1','day_median1','day_std1','day_var1',
              'day_max2','day_min2','day_mean2','day_median2','day_std2','day_var2',
              'day_max3','day_min3','day_mean3','day_median3','day_std3','day_var3',
              'day_max4','day_min4','day_mean4','day_median4','day_std4','day_var4',
              'day_max5','day_min5','day_mean5','day_median5','day_std5','day_var5',
              'day_max6','day_min6','day_mean6','day_median6','day_std6','day_var6',
              'day_max7','day_min7','day_mean7','day_median7','day_std7','day_var7',
              'area_id']

    features = pd.DataFrame(features,columns = column)
    features.to_csv('train_feature3.csv',index=False)


if __name__ == '__main__':
    #put_time_tag()
    get_train_features()


