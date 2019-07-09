import pandas as pd
import numpy as np


#获取统计特征，返回list
def get_statistic_features(df,theory_max) ->list:
    nums_list = list(df['numbers'])
    length = len(nums_list)
    #如果理论上应该有7天的数据，实际采样只有5天有，则相差的两天手动补0
    if length != theory_max:
        differ = theory_max - length
        nums_list.extend([0]*differ)
    nums_array = np.array(nums_list)

    sum = np.sum(nums_array)
    max = np.max(nums_array)
    min = np.min(nums_array)
    mean = np.mean(nums_array)
    median = np.median(nums_array)
    std = np.std(nums_array)
    var = np.var(nums_array)
    return [max,min,mean,median,std,var,sum]

#平时0 周末1 节假日2 无假期节假日3  不区分4  年前一周5  年中一周6  年后一周7
def get_dif_cate_features(df,cate):
    full_record = [116,46,17,3,182,7,7,7]  #每种类别理论上应该出现的次数 比如平时总共有116天
    theory_max = full_record[cate] #当前类型理论上的最大值
    day_features = get_statistic_features(df,theory_max) #获取一天的统计特征
    return day_features

#添加不同日期种类均值（day_mean0 1 2 ...）的统计特征 及其各个类的排名(比如节假日均值排第一 周末均值排第二)
def get_difcat_features(ser):
    cat_list = [ser.day_mean0,ser.day_mean1,ser.day_mean2,ser.day_mean3,
                ser.day_mean4,ser.day_mean5,ser.day_mean6,ser.day_mean7]
    #获取月份均值的统计特征
    cat_array = np.array(cat_list)
    ser.day_cat_max = np.max(cat_array)
    ser.day_cat_min = np.min(cat_array)
    ser.day_cat_mean = np.mean(cat_array)
    ser.day_cat_median = np.median(cat_array)
    ser.day_cat_std = np.std(cat_array)
    ser.day_cat_var = np.var(cat_array)
    cat_list_sort = np.argsort(cat_list)
    ser.day_cat_rank7,ser.day_cat_rank6,ser.day_cat_rank5,ser.day_cat_rank4,\
    ser.day_cat_rank3,ser.day_cat_rank2,ser.day_cat_rank1,ser.day_cat_rank0 = cat_list_sort
    return ser

#获取所有训练集特征
def get_train_features():
    visit = pd.read_hdf('visit_day_test_new.h5')
    visit_area = visit.groupby('area_id') #按照area_id进行分组
    length = len(visit_area)  #统计共有多少分组
    index = 0  #记录当前处理的是第多少个分组
    features = []
    for area_id,area in visit_area:

        area['month'] = area['date'] % 10000 // 100
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

        #统计各个月份的均值 均值的一阶差分(顺序是10 11 12 1 2 3）  月份按照均值排序的排名 比如10月份均值最高  对应rank_month1 = 10
        month_day = {10:31,11:30,12:31,1:31,2:28,3:31}
        month_feature = area.groupby('month',as_index=False)['numbers'].agg({'month_sum':'sum'})

        #有些地区不是每个月都有人  找到这些月份并补充month_sum为0
        month = set(month_feature['month'].unique())
        month_dif = {10,11,12,1,2,3} - month
        if month_dif:
            sup = pd.DataFrame()
            sup['month'] = list(month_dif)
            sup['month_sum'] = [0] * sup.shape[0]
            month_feature = pd.concat([month_feature,sup],ignore_index=True)

        month_feature['days'] = month_feature['month'].apply(lambda x:month_day[x])
        month_feature['mean'] = month_feature['month_sum'] / month_feature['days']
        month_feature.sort_values(by='month_sum',ascending=False,inplace=True)
        month_feature['ranks'] = range(1,7)
        #获取各个月份的均值 和 一阶差分
        day_month10 = float(month_feature.loc[month_feature.month==10,'mean'].values)
        day_month11 = float(month_feature.loc[month_feature.month==11,'mean'].values)
        day_month12 = float(month_feature.loc[month_feature.month==12,'mean'].values)
        day_month1 = float(month_feature.loc[month_feature.month==1,'mean'].values)
        day_month2 = float(month_feature.loc[month_feature.month==2,'mean'].values)
        day_month3 = float(month_feature.loc[month_feature.month==3,'mean'].values)
        day_month10_11 = day_month11 - day_month10
        day_month11_12 = day_month12 - day_month11
        day_month12_1 = day_month1 - day_month12
        day_month1_2 = day_month2 - day_month1
        day_month2_3 = day_month3 - day_month2
        #获取各个月份均值的排名
        day_month_rank1 = int(month_feature.loc[month_feature.ranks==1,'month'].values)
        day_month_rank2 = int(month_feature.loc[month_feature.ranks==2,'month'].values)
        day_month_rank3 = int(month_feature.loc[month_feature.ranks==3,'month'].values)
        day_month_rank4 = int(month_feature.loc[month_feature.ranks==4,'month'].values)
        day_month_rank5 = int(month_feature.loc[month_feature.ranks==5,'month'].values)
        day_month_rank6 = int(month_feature.loc[month_feature.ranks==6,'month'].values)
        #获取月份均值的统计特征
        month_list = list(month_feature['mean'])
        month_array = np.array(month_list)
        day_month_max = np.max(month_array)
        day_month_min = np.min(month_array)
        day_month_mean = np.mean(month_array)
        day_month_median = np.median(month_array)
        day_month_std = np.std(month_array)
        day_month_var = np.var(month_array)
        new_features = [day_month10,day_month11,day_month12,day_month1,day_month2,day_month3,day_month10_11,
                        day_month11_12,day_month12_1,day_month1_2,day_month2_3,day_month_rank1,day_month_rank2,
                        day_month_rank3,day_month_rank4,day_month_rank5,day_month_rank6,day_month_max,day_month_min,
                        day_month_mean,day_month_median,day_month_std,day_month_var]
        cur_features = cur_features + new_features
        cur_features.append(area_id)
        features.append(cur_features)
    #column是所有的列名 0-7分别代表8种date_cate  平时、周末、节假日等
    column = ['day_max0','day_min0','day_mean0','day_median0','day_std0','day_var0','day_sum0',
              'day_max1','day_min1','day_mean1','day_median1','day_std1','day_var1','day_sum1',
              'day_max2','day_min2','day_mean2','day_median2','day_std2','day_var2','day_sum2',
              'day_max3','day_min3','day_mean3','day_median3','day_std3','day_var3','day_sum3',
              'day_max4','day_min4','day_mean4','day_median4','day_std4','day_var4','day_sum4',
              'day_max5','day_min5','day_mean5','day_median5','day_std5','day_var5','day_sum5',
              'day_max6','day_min6','day_mean6','day_median6','day_std6','day_var6','day_sum6',
              'day_max7','day_min7','day_mean7','day_median7','day_std7','day_var7','day_sum7',
              'day_month10','day_month11','day_month12','day_month1','day_month2','day_month3','day_month10_11',
              'day_month11_12','day_month12_1','day_month1_2','day_month2_3','day_month_rank1','day_month_rank2',
              'day_month_rank3','day_month_rank4','day_month_rank5','day_month_rank6','day_month_max','day_month_min',
              'day_month_mean','day_month_median','day_month_std','day_month_var',
              'area_id']


    features = pd.DataFrame(features,columns = column)
    #添加不同日期种类均值（day_mean0 1 2 ...）的统计特征 及其各个类的排名(比如节假日均值排第一 周末均值排第二)
    features = pd.concat([features,pd.DataFrame(columns=['day_cat_max','day_cat_min','day_cat_mean','day_cat_median','day_cat_std',
                                                         'day_cat_var','day_cat_rank0','day_cat_rank1','day_cat_rank2',
                                                         'day_cat_rank3','day_cat_rank4','day_cat_rank5','day_cat_rank6',
                                                         'day_cat_rank7'])],sort=False)
    features = features.apply(get_difcat_features,axis=1)
    features.to_csv('test_feature3_.csv',index=False)


if __name__ == '__main__':
    get_train_features()


