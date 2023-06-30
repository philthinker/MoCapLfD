import torch
import torch.nn as nn  # nn是神经网络的缩写
from thop import profile

from MultiFus_RGB_Motion2.mypath import Path


class C3D_MoCap(nn.Module):
    """
    The changed C3D network with Mocap DIMENSEN by HYH on 2022-3-3.
    """

    def __init__(self, num_classes, pretrained=False):
        super(C3D_MoCap, self).__init__()

        #  torch.nn.Conv3D(in_channels, out_channels，kernel_size, stride, padding, ...)，
        #  --N: batch_size, 以此训练的样本数
        #  --Cin: 通道数，对于一般的RGB图像就是3
        #  ----kernel_size: 卷积核大小
        #  ----stride/padding: 步长/补位

        # 如果我们有一个n×n 的图像，用f×f的过滤器做卷积，那么输出的维度就是(n−f+1)×(n−f+1)

        self.conv1 = nn.Conv3d(3, 64, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool1 = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))
        # (1, 2, 2)对应（n,h,w）,第一次n设为1表示第一次不压缩16帧这个时间特征，h、w都变为原先的1/2;

        self.conv2 = nn.Conv3d(64, 128, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool2 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))

        self.conv3a = nn.Conv3d(128, 256, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.conv3b = nn.Conv3d(256, 256, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool3 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))

        self.conv4a = nn.Conv3d(256, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.conv4b = nn.Conv3d(512, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool4 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))

        self.conv5a = nn.Conv3d(512, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.conv5b = nn.Conv3d(512, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool5 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2), padding=(0, 1, 1))

        self.fc6 = nn.Linear(8192, 2048)  # 全连接层:8192就是h*w*c，表示特征数，将其变为4096个
        self.fc7 = nn.Linear(2048, 256)

        self.fc8 = nn.Linear(256 + 19*16, 64)
        self.fc9 = nn.Linear(64, num_classes)

        self.dropout = nn.Dropout(p=0.5)

        # Rectified Linear Unit(ReLU)激活函数，也叫线性整流函数f(x)=max(0,x)
        # 比Sigmoid函数收敛速度更快。输出以0为中心。
        self.relu = nn.ReLU()

        self.__init_weight()

        if pretrained:
            self.__load_C3Dpretrained_weights()

    def forward(self, x, input_mocap_tensor):

        # print ('1:', x.size())
        # 1: torch.Size([1, 3, 16, 112, 112])依次为batchsize=1，rgb3个通道，时间序列上为16帧，裁剪图大小为112*112；
        # 即batchsize，channel通道数，一次的序列长度（时间维度），crop的长宽h、w）
        x = self.relu(self.conv1(x))
        # print ('2:',x.size())  # 2: torch.Size([1, 64, 16, 112, 112])
        x = self.pool1(x)
        # print ('3:',x.size())

        x = self.relu(self.conv2(x))
        # print ('4:',x.size())
        x = self.pool2(x)
        # print ('5:',x.size())

        x = self.relu(self.conv3a(x))
        # print ('6:',x.size())
        x = self.relu(self.conv3b(x))
        # print ('7:',x.size())
        x = self.pool3(x)
        # print ('8:',x.size())

        x = self.relu(self.conv4a(x))
        # print ('9:',x.size())
        x = self.relu(self.conv4b(x))
        # print ('10:',x.size())
        x = self.pool4(x)
        # print ('11:',x.size())

        x = self.relu(self.conv5a(x))
        # print ('12:',x.size())
        x = self.relu(self.conv5b(x))
        # print ('13:',x.size())
        x = self.pool5(x)
        # print ('14:',x.size())
        # 14: torch.Size([1, 512, 1, 4, 4])
        # 512表示特征图的个数（从rgb3个变为了512个），512*1(t)*4(h)*4(w)=8192

        x = x.view(-1, 8192)  # view参数中的-1就代表这个位置由其他位置的数字来推断
        # print ('15:',x.size())
        x = self.relu(self.fc6(x))
        # print ('16:',x.size())
        x = self.dropout(x)
        x = self.relu(self.fc7(x))
        x = self.dropout(x)

        # 直接拼接concatenate:   x = torch.cat((z_img,z_dep,ft_tensor,tac_tensor), dim=1).to(device)
        # input_mocap_tensor = input_mocap_tensor.to(0)
        input_mocap_tensor = input_mocap_tensor.view(-1, 19*16)
        x = torch.cat((x, input_mocap_tensor), dim=1)
        x = self.relu(self.fc8(x))
        # print('17:', x.size())

        logits = self.fc9(x)
        # print ('18:',logits.size())
        return logits

    def __load_C3Dpretrained_weights(self):
        """Initialiaze network."""
        print("load_pretrained_weights")
        corresp_name = {
            # Conv1
            "features.0.weight": "conv1.weight",
            "features.0.bias": "conv1.bias",
            # Conv2
            "features.3.weight": "conv2.weight",
            "features.3.bias": "conv2.bias",
            # Conv3a
            "features.6.weight": "conv3a.weight",
            "features.6.bias": "conv3a.bias",
            # Conv3b
            "features.8.weight": "conv3b.weight",
            "features.8.bias": "conv3b.bias",
            # Conv4a
            "features.11.weight": "conv4a.weight",
            "features.11.bias": "conv4a.bias",
            # Conv4b
            "features.13.weight": "conv4b.weight",
            "features.13.bias": "conv4b.bias",
            # Conv5a
            "features.16.weight": "conv5a.weight",
            "features.16.bias": "conv5a.bias",
            # Conv5b
            "features.18.weight": "conv5b.weight",
            "features.18.bias": "conv5b.bias",
            # fc6
            "classifier.0.weight": "fc6.weight",
            "classifier.0.bias": "fc6.bias",
            # fc7
            "classifier.3.weight": "fc7.weight",
            "classifier.3.bias": "fc7.bias",
            # fc8
            "classifier.6.weight": "fc8.weight",
            "classifier.6.bias": "fc8.bias",
            # fc9
            "classifier.8.weight": "fc9.weight",
            "classifier.8.bias": "fc9.bias",
        }

        p_dict = torch.load(Path.C3D_model_dir())
        s_dict = self.state_dict()
        for name in p_dict:
            if name not in corresp_name:
                continue
            if s_dict[corresp_name[name]].shape != p_dict[name].shape:
                continue
            s_dict[corresp_name[name]] = p_dict[name]
        self.load_state_dict(s_dict)

    def __init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                # n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                # m.weight.data.normal_(0, math.sqrt(2. / n))
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


def get_1x_lr_params(model):
    """
    This generator returns all the parameters for conv and 3 fc layers of the net.
    """
    b = [model.conv1, model.conv2, model.conv3a, model.conv3b, model.conv4a, model.conv4b,
         model.conv5a, model.conv5b, model.fc6, model.fc7, model.fc8]
    for i in range(len(b)):
        for k in b[i].parameters():
            if k.requires_grad:
                yield k


def get_10x_lr_params(model):
    """
    This generator returns all the parameters for the last fc layer of the net.
    """
    b = [model.fc9]
    for j in range(len(b)):
        for k in b[j].parameters():
            if k.requires_grad:
                yield k


if __name__ == "__main__":
    inputs1 = torch.rand(1, 3, 16, 112, 112)
    # inputs2 = torch.rand(1, 16, 19)
    net = C3D_MoCap(num_classes=6, pretrained=False)
    input_mocap_tensor = torch.rand(16, 19)
    outputs = net.forward(inputs1, input_mocap_tensor)

    FLOPs,params = profile(net,(inputs1,input_mocap_tensor,))
    print(FLOPs)  # 38.53735 GFLOPs  (G:10E9)
    print(params)  # 44.996 MParams   (M:10E6)

    print(outputs.size())
    print(outputs)