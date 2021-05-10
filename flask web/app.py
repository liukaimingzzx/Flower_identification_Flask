from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
from datetime import timedelta
import os
app = Flask(__name__)

from PIL import Image
import torchvision.transforms as transforms
from torchvision import models
from torch.autograd import Variable
import torch
from torch import nn

#数据预处理
data_transform = transforms.Compose([
    transforms.Resize((224,224), 2),                           #对图像大小统一
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[    #图像归一化
                             0.229, 0.224, 0.225])
         ])

#类别
data_classes = ['樱花(cerasus)', '康乃馨(dianthus)', '毛地黄(igitalis_purpurea)', '花菱草(eschscholtzia)',
                '勋章菊(gazania)', '茉莉花(jasminum)', '紫罗兰(matthiola)', '水仙花(narcissus)', '睡莲(nymphaea)',
                '牵牛花(pharbitis)', '杜鹃花(rhododendron)', '月季花(rosa)', '肿柄菊(tithonia)', '旱金莲(tropaeolum_majus)',
                '雏菊(daisy)', '蒲公英(dandelion)', '桃花(peach_blossom)', '玫瑰花(rose)', '向日葵(sunflowers)', '郁金香(tulips)']
baidu_urls = ['https://baike.baidu.com/item/%E6%A8%B1%E8%8A%B1/70387',
              'https://baike.baidu.com/item/%E5%BA%B7%E4%B9%83%E9%A6%A8/34029',
              'https://baike.baidu.com/item/%E6%AF%9B%E5%9C%B0%E9%BB%84',
              'https://baike.baidu.com/item/%E8%8A%B1%E8%8F%B1%E8%8D%89',
              'https://baike.baidu.com/item/%E5%8B%8B%E7%AB%A0%E8%8F%8A',
              'https://baike.baidu.com/item/%E8%8C%89%E8%8E%89%E8%8A%B1/4951',
              'https://baike.baidu.com/item/%E7%B4%AB%E7%BD%97%E5%85%B0/5033',
              'https://baike.baidu.com/item/%E6%B0%B4%E4%BB%99/6410',
              'https://baike.baidu.com/item/%E7%9D%A1%E8%8E%B2/11999986',
              'https://baike.baidu.com/item/%E7%89%B5%E7%89%9B/79184',
              'https://baike.baidu.com/item/%E6%9D%9C%E9%B9%83/18876?fromtitle=%E6%9D%9C%E9%B9%83%E8%8A%B1&fromid=159279',
              'https://baike.baidu.com/item/%E6%9C%88%E5%AD%A3%E8%8A%B1/14505544?fromtitle=%E6%9C%88%E5%AD%A3&fromid=32865',
              'https://baike.baidu.com/item/%E8%82%BF%E6%9F%84%E8%8F%8A',
              'https://baike.baidu.com/item/%E6%97%B1%E9%87%91%E8%8E%B2/892401',
              'https://baike.baidu.com/item/%E9%9B%8F%E8%8F%8A/11015634',
              'https://baike.baidu.com/item/%E8%92%B2%E5%85%AC%E8%8B%B1/17854',
              'https://baike.baidu.com/item/%E6%A1%83%E8%8A%B1/6172',
              'https://baike.baidu.com/item/%E7%8E%AB%E7%91%B0/63206',
              'https://baike.baidu.com/item/%E5%90%91%E6%97%A5%E8%91%B5/6106',
              'https://baike.baidu.com/item/%E9%83%81%E9%87%91%E9%A6%99/13506']

#选择CPU还是GPU的操作
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#加载模型
net = models.vgg16()
net.classifier = nn.Sequential(nn.Linear(25088, 4096),      #vgg16
                                 nn.ReLU(),
                                 nn.Dropout(p=0.5),
                                 nn.Linear(4096, 4096),
                                 nn.ReLU(),
                                 nn.Dropout(p=0.5),
                                 nn.Linear(4096, 20))

net.load_state_dict(torch.load("D:\PyCharm 2020.2\PycharmProjects/flower\save\VGG16_flower_200.pkl",map_location=torch.device('cpu')))
net.eval()
net.to(device)

# 设置允许的文件格式
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'JPG', 'PNG', 'bmp'])
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS
# 设置静态文件缓存过期时间
app.send_file_max_age_default = timedelta(seconds=1)
@app.route('/upload',methods=['POST', 'GET'])
def upload():

    if request.method == 'POST':
        # 通过file标签获取文件
        f = request.files['file']
        if not (f and allowed_file(f.filename)):
            return jsonify({"error": 1001, "msg": "图片类型：png、PNG、jpg、JPG、bmp"})
        # 当前文件所在路径
        basepath = os.path.dirname(__file__)
        # 一定要先创建该文件夹，不然会提示没有该路径
        upload_path = os.path.join(basepath, 'images', secure_filename(f.filename))
        # 保存文件
        f.save(upload_path)

    #读取数据
    img = Image.open(upload_path)
    img=data_transform(img)#这里经过转换后输出的input格式是[C,H,W],网络输入还需要增加一维批量大小B
    img = img.unsqueeze(0)#增加一维，输出的img格式为[1,C,H,W]

    img = Variable(img)
    score = net(img)# 将图片输入网络得到输出
    probability = nn.functional.softmax(score,dim=1)#计算softmax，即该图片属于各类的概率
    max_value,index = torch.max(probability,1)#找到最大概率对应的索引号，该图片即为该索引号对应的类别

    #return str(data_classes[index.item()])
    #print()
    return str(data_classes[index.item()]+","+format(max_value.item())+","+baidu_urls[index.item()])


if __name__ == '__main__':
    app.run(host='xxx.xxx.xxx.xxx')

