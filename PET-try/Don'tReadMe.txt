1.根据RadMe.markdown中的指引加载预训练模型（github无法上传超过100Mb的单个文件，所以无法上传权重文件，将vgg权重放在./pretrained下，SHA权重放在./resume下）
2.加载需要预测的图像至.\data\MyData\test_data\images
3.执行temp.py
4.用bash执行$ sh pred.sh ，在根目录下results生成可视化结果，test_json文件下生成相应标注