import numpy as np
import cv2


# VOC2012 H W 3

def getdata(filename):
    return cv2.imread(filename, -1)


DataIndexPath = "VOCdevkit/VOC2012/ImageSets/Segmentation/"
DataPath = "VOCdevkit/VOC2012"


def getdatasets(type):
    datas = []
    labels = []
    f = open(DataIndexPath + type)
    alllines = f.readlines()
    for line in alllines:
        data = getdata(DataPath + "/JPEGImages/" + line.split("\n")[0] + ".jpg")
        label = getdata(DataPath + "/Label/" + line.split("\n")[0] + ".png")
        datas.append(data)
        labels.append(label)

    return datas, labels


class Unet_Data:
    def __init__(self, need_shuffle):
        self.ori_train_datas, self.ori_train_labels = getdatasets("train.txt")
        self.ori_trainval_datas, self.ori_trainval_labels = getdatasets("trainval.txt")
        self.batch_start = 0
        self.train_datas = []
        self.train_labels = []
        ISZ = 256
        total = len(self.ori_train_datas)
        for num in range(total):
            # 根据ISZ对图片裁剪为相同大小的Patch 以便于输入网络进行训练
            ISZ = int(1.0 * ISZ)
            a1 = self.ori_train_datas[num].shape[0] // ISZ
            b1 = self.ori_train_labels[num].shape[1] // ISZ
            if self.ori_train_datas[num].shape[0] < 256 or self.ori_train_datas[num].shape[1] < 256:
                continue
            # 可完整分割部分
            for i in range(a1):
                for j in range(b1):
                    data = self.ori_train_datas[num][i * ISZ:(i + 1) * ISZ, j * ISZ:(j + 1) * ISZ]
                    label = self.ori_train_labels[num][i * ISZ:(i + 1) * ISZ, j * ISZ:(j + 1) * ISZ]
                    self.train_datas.append(data)
                    cv2.imwrite(DataPath + "/split_pics/" + "{0}_{1}_{2}.jpg".format(num, i, j),
                                self.ori_train_datas[num][i * ISZ:(i + 1) * ISZ, j * ISZ:(j + 1) * ISZ])
                    self.train_labels.append(label)
            # 第一维不足未分割的内容
            for i in range(a1 - 1):
                data = self.ori_train_datas[num][i * ISZ:(i + 1) * ISZ,
                       self.ori_train_datas[num].shape[1] - ISZ:self.ori_train_datas[num].shape[1]]
                label = self.ori_train_labels[num][i * ISZ:(i + 1) * ISZ,
                        self.ori_train_labels[num].shape[1] - ISZ:self.ori_train_labels[num].shape[1]]
                self.train_datas.append(data)
                cv2.imwrite(
                    DataPath + "/split_pics/" + "{0}_{1}_{2}.jpg".format(num, i, self.ori_train_datas[num].shape[1]),
                    self.ori_train_datas[num][i * ISZ:(i + 1) * ISZ,
                    self.ori_train_datas[num].shape[1] - ISZ:self.ori_train_datas[num].shape[1]]
                )
                self.train_labels.append(label)
            # 第二维不足为分割内容
            for i in range(b1 - 1):
                data = self.ori_train_datas[num][
                       self.ori_train_datas[num].shape[0] - ISZ:self.ori_train_datas[num].shape[0],
                       i * ISZ:(i + 1) * ISZ]
                label = self.ori_train_labels[num][
                        self.ori_train_labels[num].shape[0] - ISZ:self.ori_train_labels[num].shape[0],
                        i * ISZ:(i + 1) * ISZ]
                self.train_datas.append(data)
                cv2.imwrite(
                    DataPath + "/split_pics/" + "{0}_{1}_{2}.jpg".format(num, self.ori_train_datas[num].shape[0], i),
                    self.ori_train_datas[num][
                    self.ori_train_datas[num].shape[0] - ISZ:self.ori_train_datas[num].shape[0],
                    i * ISZ:(i + 1) * ISZ]
                )
                self.train_labels.append(label)
            # 最后交界处内容
            data = self.ori_train_datas[num][
                   self.ori_train_datas[num].shape[0] - ISZ:self.ori_train_datas[num].shape[0],
                   self.ori_train_datas[num].shape[1] - ISZ:self.ori_train_datas[num].shape[1]]
            label = self.ori_train_labels[num][
                    self.ori_train_labels[num].shape[0] - ISZ:self.ori_train_labels[num].shape[0],
                    self.ori_train_labels[num].shape[1] - ISZ:self.ori_train_labels[num].shape[1]]
            self.train_datas.append(data)
            cv2.imwrite(
                DataPath + "/split_pics/" + "{0}_{1}_{2}.jpg".format(num, self.ori_train_datas[num].shape[0],
                                                                     self.ori_train_datas[num].shape[1]),
                self.ori_train_datas[num][self.ori_train_datas[num].shape[0] - ISZ:self.ori_train_datas[num].shape[0],
                self.ori_train_datas[num].shape[1] - ISZ:self.ori_train_datas[num].shape[1]]
            )
            self.train_labels.append(label)
        self.num_examples = len(self.train_datas)
        self.need_shuffle = need_shuffle
        if need_shuffle:
            self.shuffledata()

    def shuffledata(self):
        num = np.random.permutation(self.num_examples)
        self.train_datas = np.array(self.train_datas)[num]
        self.train_labels = np.array(self.train_labels)[num]

    def NextBatch(self, batch_size):
        batch_end = self.batch_start + batch_size
        if batch_end > self.num_examples:
            if self.need_shuffle:
                self.shuffledata()
                self.batch_start = 0
                batch_end = batch_size
            else:
                raise Exception("have no more datas")
        if batch_end > self.num_examples:
            raise Exception("batch size is lager than datas")

        batch_data = self.train_datas[self.batch_start: batch_end]
        batch_label = self.train_labels[self.batch_start: batch_end]
        self.batch_start = batch_end

        return batch_data, batch_label

# Train_Data = Unet_Data(False)
#
# Batch_Data, Batch_Label = Train_Data.NextBatch(2)
# print(Batch_Data.shape)
# print(Batch_Label.shape)
