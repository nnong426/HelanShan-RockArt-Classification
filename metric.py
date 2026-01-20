import numpy as np
import torch
# -------------------- 指标 --------------------
def accuracy(output, target, topk=(1,),train=True):
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)
        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        if train:
            target=torch.argmax(target,dim=1)
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        result= [correct[:k].reshape(-1).float().sum(0) * 100. / batch_size for k in topk]
    return result


# 使用示例
if __name__ == '__main__':
    # 模拟数据：batch_size=4, num_classes=10
    output = torch.randn(4, 10)  # 模型输出
    target = torch.tensor([2, 5, 1, 8])  # 真实标签
    target_=torch.zeros_like(output)
    target_[0,2]=1
    target_[1,5]=1
    target_[2,1]=1
    target_[3,8]=1
    
    # 同时计算Top1和Top5准确率
    top1_acc, top5_acc = accuracy(output, target_, topk=(1, 5),train=True)
    print(f"Top1准确率: {top1_acc.item():.2f}%")
    print(f"Top5准确率: {top5_acc.item():.2f}%")
