from matplotlib import pyplot as plt
import os

def plot_confusion(confusion,model,name2label,lr):
    plt.imshow(confusion, cmap=plt.cm.Blues)
    # ticks 坐标轴的坐标点
    # label 坐标轴标签说明
    indices = range(len(confusion))
   
    #plt.xticks(indices, [0, 1, 2])
    #plt.yticks(indices, [0, 1, 2])
    label2name={label:name for name,label in name2label.items() }
    plt.xticks(indices, [label2name[i] for i in range(len(label2name))])
    plt.yticks(indices, [label2name[i] for i in range(len(label2name))])

    plt.colorbar()

    plt.xlabel('预测值')
    plt.ylabel('真实值')
    plt.title(model+'混淆矩阵')

   
    plt.rcParams['font.sans-serif']=['SimHei']
    plt.rcParams['axes.unicode_minus'] = False

    # 显示数据
    for first_index in range(len(confusion)):    #第几行
        for second_index in range(len(confusion[first_index])):    #第几列
            plt.text( second_index,first_index, confusion[first_index][second_index])
    if not os.path.exists("./image_confusion"):
        os.makedirs("./image_confusion")
    plt.savefig(f"./image_confusion/{model}_{lr}.png")
   
    plt.show()
