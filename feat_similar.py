import csv
import numpy as np
from scipy.spatial.distance import euclidean

def calculation(test_vector):
    # 犯罪地点
    dis_crime = [euclidean(test_vector, crime_fv) for crime_fv in crime_fv_list]
    min_crime_idx = np.argsort(dis_crime)[0]  # 距离最短的索引
    min_dis_crime_fv = crime_fv_list[min_crime_idx]  # 距离最短的向量
    norm1 = np.linalg.norm(min_dis_crime_fv)
    min_dis_crime_fv /= norm1

    # 非犯罪地点
    dis_non_crime = [euclidean(test_vector, non_crime_fv) for non_crime_fv in non_crime_fv_list]
    min_non_crime_idx = np.argsort(dis_non_crime)[0]  # 距离最短的索引
    min_dis_non_crime_fv = non_crime_fv_list[min_non_crime_idx]  # 距离最短的向量
    norm2 = np.linalg.norm(min_dis_non_crime_fv)
    min_dis_non_crime_fv /= norm2

    norm = np.linalg.norm(test_vector)
    test_vector /= norm
    score = np.dot(test_vector, min_dis_crime_fv) - np.dot(test_vector, min_dis_non_crime_fv)
    print(str(score))
    return score

# crime_fv_list存放所有犯罪地点特征向量
crime_fv_list=[]
crime_fv_file=open('crime_features_meg_img.csv', 'r', encoding='utf-8')
csv_reader = csv.reader(crime_fv_file)
for row in csv_reader:
    numeric_row = [float(value) for value in row]  # 将每个元素转换为浮点数
    crime_fv_list.append(np.array(numeric_row))

# # crime_name_list存放所有犯罪地点图像文件名
# crime_name_list=[]
# crime_name_file=open('crime_name_meg_img.csv', 'r', encoding='utf-8')
# csv_reader1 = csv.reader(crime_name_file)
# for row in csv_reader1:
#     crime_name_list.append(row)

# non_crime_fv_list存放7760个非犯罪地点特征向量
non_crime_fv_list=[]
non_crime_fv_file=open('non_crime_features_meg_img_7760.csv', 'r', encoding='utf-8')
csv_reader2 = csv.reader(non_crime_fv_file)
for row in csv_reader2:
    numeric_row = [float(value) for value in row]  # 将每个元素转换为浮点数
    non_crime_fv_list.append(np.array(numeric_row))

# # non_crime_name_list存放7760个非犯罪地点图像文件名
# non_crime_name_list=[]
# non_crime_name_file=open('non_crime_name_meg_img_7760.csv', 'r', encoding='utf-8')
# csv_reader3 = csv.reader(non_crime_name_file)
# for row in csv_reader3:
#     non_crime_name_list.append(row)

id_list=[]
id_file=open('non_crime_name_meg_img_7760.csv', 'r', encoding='utf-8')
csv_reader_id = csv.reader(id_file)
for row in csv_reader_id:
    id_list.append(row)

vec_list=[]
vec_file=open('non_crime_features_meg_img_7760.csv', 'r', encoding='utf-8')
csv_reader_vec = csv.reader(vec_file)
for row in csv_reader_vec:
    numeric_row = [float(value) for value in row]  # 将每个元素转换为浮点数
    vec_list.append(np.array(numeric_row))

rslt_file=open('risk_score_non_crime_7760.csv', 'w', encoding='utf-8') #输出文件

for idx,test_vector in enumerate(vec_list):
    print(str(idx))
    rslt_file.write(id_list[idx][0]+',')
    score=calculation(test_vector)
    rslt_file.write(str(score)+'\n')



    




