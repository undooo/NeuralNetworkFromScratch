'''
Description: 
    1）创建dataloader
    2) 创建神经网络
    3) 训练模型
    4) 展示效果
Author: KuhnLiu
Date: 2024-07-11 10:04:42
LastEditTime: 2024-07-14 11:42:31
LastEditors: KuhnLiu
'''
import numpy as np
import pathlib
import matplotlib.pyplot as plt

class Dataloader():
    """
        数据读取器
    """  
    def get_data(self):
        with np.load(f"{pathlib.Path(__file__).parent.absolute()}/data/mnist.npz") as f:
            images, labels = f["x_train"], f["y_train"]
        images = images.astype("float32") / 255
        images = np.reshape(images, (images.shape[0], images.shape[1] * images.shape[2]))
        labels = np.eye(10)[labels]
        return images, labels
        

if __name__ == "__main__":
    # 通过dataloader读取数据
    dataloader = Dataloader()
    images, labels = dataloader.get_data()
    # 创建模型
    # 本代码重在快速实现神经网络，因此对模型不做封装
    w_i_h = np.random.uniform(-0.5, 0.5, (20, 784))
    w_h_o = np.random.uniform(-0.5, 0.5, (10, 20))
    b_i_h = np.zeros((20, 1))
    b_h_o = np.zeros((10, 1))
    # 训练模型
    # - 设置超参数
    learn_rate = 0.01
    nr_correct = 0
    epochs = 5    
    for epoch in range(epochs):
        for img, l in zip(images, labels):
            img.shape += (1,)
            l.shape += (1,)
            # 前向传播
            # - 输入层->隐藏层
            h_pre = b_i_h + w_i_h @ img
            h = 1 / (1 + np.exp(-h_pre))
            # - 隐藏层->输出层
            o_pre = b_h_o + w_h_o @ h
            o = 1 / (1 + np.exp(-o_pre))
            # - 损失函数计算
            e = 1 / len(o) * np.sum((o - l) ** 2, axis=0)
            nr_correct += int(np.argmax(o) == np.argmax(l))
            # 反向传播
            # - 输出层->隐藏层
            delta_o = 0.2* (o - l)
            delta_z = (o * (1 - o))
            delta_w_h = np.transpose(h)
            w_h_o += -learn_rate * delta_o @ delta_w_h * delta_z 
            b_h_o += -learn_rate * delta_o
            # - 隐藏层->输入层
            delta_h = np.transpose(w_h_o)
            delta_z_2 = (h * (1 - h))
            delta_w_i = np.transpose(img)
            w_i_h += -learn_rate * delta_h @ delta_o * delta_z_2 @ delta_w_i
            b_i_h += -learn_rate * delta_h @ delta_o * delta_z_2
        # 输出精准度
        print(f"Acc: {round((nr_correct / images.shape[0]) * 100, 2)}%")
        nr_correct = 0
    # 展示效果
    while True:
        index = int(input("输入编号进行预测 (0 - 59999): "))
        img = images[index]
        plt.imshow(img.reshape(28, 28), cmap="Greys")
        img.shape += (1,)
        # 前向传播
        h_pre = b_i_h + w_i_h @ img.reshape(784, 1)
        h = 1 / (1 + np.exp(-h_pre))
        o_pre = b_h_o + w_h_o @ h
        o = 1 / (1 + np.exp(-o_pre))
        plt.title(f"This figure is predicted to be: {o.argmax()} :")
        plt.show()
   



