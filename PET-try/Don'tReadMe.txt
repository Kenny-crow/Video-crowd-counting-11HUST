1.根据RadMe.markdown中的指引加载预训练模型（github无法上传超过100Mb的单个文件）
2.加载需要预测的图像至.\data\MyData\test_data\images
3.执行temp.py生成相应的假GT文件
4.用bash执行$ sh eval.sh ，在根目录下results生成可视化结果，test_json文件下生成相应标注