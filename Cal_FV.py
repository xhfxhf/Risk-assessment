import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import os
import csv

# 加载预训练的ResNet50模型
resnet50 = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
# 移除最后的全连接层
resnet50 = torch.nn.Sequential(*(list(resnet50.children())[:-1]))
# 将模型设置为评估模式
resnet50.eval()

# 图像预处理
def preprocess_image(image_path):
    image = Image.open(image_path)
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        # transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    input_tensor = preprocess(image)
    # return input_tensor
    # 在 batch 维度上增加一维
    input_batch = input_tensor.unsqueeze(0)
    return input_batch

out_file=open('features_onePics.csv','w',encoding='UTF-8')   # 输出文件（特征向量）
csv_writer=csv.writer(out_file)
img_name_file=open('id_onePics.csv','w',encoding='UTF-8')   # 输出文件（图片文件名）

# out_features = []   # 存储图像特征向量
filenames = os.listdir('./onePics/')
for idx, image_name in enumerate(filenames):
    print(str(idx))
    image_fullname = './onePics/'+image_name
    # 计算图像特征向量
    input_tensor = preprocess_image(image_fullname)
    with torch.no_grad():
        features = resnet50(input_tensor)
    # out_features.append(features.numpy().squeeze())
    print(features.numpy().squeeze().shape)   # 打印特征向量的形状
    csv_writer.writerow(features.numpy().squeeze())

    index=(image_name.find(".jpg"))
    file_id=image_name[0:index]
    img_name_file.write(file_id+'\n')
# print(len(out_features))
# for i in out_features:
#     print(i)
