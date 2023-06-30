import torch
import torch.nn as nn  # nn是神经网络的缩写
from thop import profile

from mypath import Path


class C3D_MoCap(nn.Module):
    """
    The changed C3D network with Mocap DIMENSEN by HYH on 2022-3-3.
    """

    def __init__(self, num_classes, pretrained=False, LSTM_pretrained=False):
        super(C3D_MoCap, self).__init__()

        #  torch.nn.Conv3D(in_channels, out_channels，kernel_size, stride, padding, ...)，
        #  --N: batch_size, 以此训练的样本数
        #  --Cin: 通道数，对于一般的RGB图像就是3
        #  ----kernel_size: 卷积核大小
        #  ----stride/padding: 步长/补位

        # 如果我们有一个n×n 的图像，用f×f的过滤器做卷积，那么输出的维度就是(n−f+1)×(n−f+1)

        self.conv1 = nn.Conv3d(3, 64, kernel_size=(3, 3, 3), padding=(0, 1, 1))
        self.pool1 = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))
        # (1, 2, 2)对应（n,h,w）,第一次n设为1表示第一次不压缩16帧这个时间特征，h、w都变为原先的1/2;

        self.conv2 = nn.Conv3d(64, 128, kernel_size=(3, 3, 3), padding=(0, 1, 1))
        self.pool2 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))

        self.conv3a = nn.Conv3d(128, 256, kernel_size=(3, 3, 3), padding=(0, 1, 1))
        self.conv3b = nn.Conv3d(256, 256, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool3 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))

        self.conv4a = nn.Conv3d(256, 512, kernel_size=(3, 3, 3), padding=(0, 1, 1))
        self.conv4b = nn.Conv3d(512, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool4 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))

        self.conv5a = nn.Conv3d(512, 512, kernel_size=(3, 3, 3), padding=(0, 1, 1))
        self.conv5b = nn.Conv3d(512, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool5 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2), padding=(0, 1, 1))

        self.fc6 = nn.Linear(8192, 4096)  # 全连接层:8192就是h*w*c，表示特征数，将其变为4096个
        self.fc7 = nn.Linear(4096, 4096)  # 特征层融合
        self.fc8 = nn.Linear(4096, num_classes)

        self.input_size = 19 * 16
        self.hidden_size = 30
        self.num_layers = 2
        self.lstm = nn.LSTM(self.input_size, self.hidden_size, self.num_layers,
                            batch_first=True)  # input_size,hidden_size ,num_layers

        self.fc_lstm = nn.Linear(self.hidden_size, num_classes)

        self.fc9 = nn.Linear(num_classes*2, num_classes)
        self.dropout = nn.Dropout(p=0.5)
        # Rectified Linear Unit(ReLU)激活函数，也叫线性整流函数f(x)=max(0,x)
        # 比Sigmoid函数收敛速度更快。输出以0为中心。
        self.relu = nn.ReLU()
        self.__init_weight()

        if pretrained:
            self.__load_C3Dpretrained_weights()
        if LSTM_pretrained:
            self.__load_LSTMpretrained_weights()

    def forward(self, x, input_mocap_tensor, x_pre0,x_pre1,x_pre2,x_pre3,x_pre4, h,c):
        x = torch.cat((x_pre0, x), dim=2)
        # loc = torch.tensor([x.shape[2] - 2, x.shape[2] - 1]).to(x.device) # loc会随x的变化而变化，后续无需撒互信---训练时会报错？
        x_pre0 = torch.index_select(x, 2, torch.tensor([x.shape[2] - 2, x.shape[2] - 1]).to(x.device))  # 这种写法返回的张量不与原始张量共享内存空间--采取。其中的loc必须为tensor，且训练时需要保持与x在一个device上
        x = self.relu(self.conv1(x))
        x = self.pool1(x)
        # tmp.append(x.index_select(2, torch.tensor([x.shape[2] - 2, x.shape[2] - 1])))

        x = torch.cat((x_pre1, x), dim=2)
        x_pre1 = torch.index_select(x, 2, torch.tensor([x.shape[2] - 2, x.shape[2] - 1]).to(x.device))
        x = self.relu(self.conv2(x))
        x = self.pool2(x)
        # tmp.append(x.index_select(2, torch.tensor([x.shape[2] - 2, x.shape[2] - 1])))

        x = torch.cat((x_pre2, x), dim=2)
        x_pre2 = torch.index_select(x, 2, torch.tensor([x.shape[2] - 2, x.shape[2] - 1]).to(x.device))
        x = self.relu(self.conv3a(x))
        x = self.relu(self.conv3b(x))
        x = self.pool3(x)
        # tmp.append(x.index_select(2, torch.tensor([x.shape[2] - 2, x.shape[2] - 1])))

        x = torch.cat((x_pre3, x), dim=2)
        x_pre3 = torch.index_select(x, 2, torch.tensor([x.shape[2] - 2, x.shape[2] - 1]).to(x.device))
        x = self.relu(self.conv4a(x))
        x = self.relu(self.conv4b(x))
        x = self.pool4(x)
        # tmp.append(x.index_select(2, torch.tensor([x.shape[2] - 2, x.shape[2] - 1])))


        x = torch.cat((x_pre4, x), dim=2)
        x_pre4 = torch.index_select(x, 2, torch.tensor([x.shape[2] - 2, x.shape[2] - 1]).to(x.device))
        x = self.relu(self.conv5a(x))
        x = self.relu(self.conv5b(x))
        x = self.pool5(x)
        # 14: torch.Size([1, 512, 1, 4, 4])
        # 512表示特征图的个数（从rgb3个变为了512个），512*1(t)*4(h)*4(w)=8192

        x = x.view(-1, 8192)  # view参数中的-1就代表这个位置由其他位置的数字来推断
        # print ('15:',x.size())
        x = self.relu(self.fc6(x))
        # print ('16:',x.size())
        x = self.dropout(x)

        # # 直接拼接concatenate:
        # input_mocap_tensor = input_mocap_tensor.view(-1, 19*16)
        # x = torch.cat((x, input_mocap_tensor), dim=1)
        x = self.relu(self.fc7(x))
        x = self.dropout(x)
        x = self.fc8(x)

        input_mocap_tensor = input_mocap_tensor.view(-1, 1, 16 * 19)# 注意lstm的输入我在init中定义了是batch first，
        input_mocap_tensor, (h, c) = self.lstm(input_mocap_tensor, (h, c))
        input_mocap_tensor = input_mocap_tensor.view(-1, 30)
        input_mocap_tensor = self.fc_lstm(input_mocap_tensor)

        # 决策层拼接
        input_mocap_tensor = input_mocap_tensor.view(-1, x.shape[-1])
        x = torch.cat((x, input_mocap_tensor), dim=1)
        logits = self.fc9(x)

        return logits, x_pre0,x_pre1,x_pre2,x_pre3,x_pre4, h,c

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

    def __load_LSTMpretrained_weights(self):
        print("load_LSTM_pretrained_weights")
        corresp_name = {
            "lstm.weight_ih_l0": "lstm.weight_ih_l0",
            "lstm.weight_hh_l0": "lstm.weight_hh_l0",
            "lstm.bias_ih_l0": "lstm.bias_ih_l0",
            "lstm.bias_hh_l0": "lstm.bias_hh_l0",
            "lstm.weight_ih_l1": "lstm.weight_ih_l1",
            "lstm.weight_hh_l1": "lstm.weight_hh_l1",
            "lstm.bias_ih_l1": "lstm.bias_ih_l1",
            "lstm.bias_hh_l1": "lstm.bias_hh_l1",
            # fc:
            "fc_lstm.weight": "fc_lstm.weight",
            "fc_lstm.bias": "fc_lstm.bias",
        }
        p_dict = torch.load(Path.LSTM_model_dir())
        s_dict = self.state_dict()
        for name in p_dict['state_dict']:
            if name not in corresp_name:
                continue
            if s_dict[corresp_name[name]].shape != p_dict['state_dict'][name].shape:
                continue
            s_dict[corresp_name[name]] = p_dict['state_dict'][name]
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
         model.conv5a, model.conv5b, model.fc6, model.fc7, model.fc8, model.lstm, model.fc_lstm, model.fc9]
    for i in range(len(b)):
        for k in b[i].parameters():
            if k.requires_grad:
                yield k


def get_10x_lr_params(model):
    """
    This generator returns all the parameters for the last fc layer of the net.
    """
    b = []
    for j in range(len(b)):
        for k in b[j].parameters():
            if k.requires_grad:
                yield k


if __name__ == "__main__":
    inputs1 = torch.rand(12, 3, 16, 112, 112)
    # x_pre = x_o.index_select(2, torch.tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]))
    # x_cur = x_o.index_select(2, torch.tensor([2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17]))
    # tmp = []
    # tmp.append(torch.rand(1, 3, 2, 112, 112))
    # tmp.append(torch.rand(1, 64, 2, 56, 56))
    # tmp.append(torch.rand(1, 128, 2, 28, 28))
    # tmp.append(torch.rand(1, 256, 2, 14, 14))
    # tmp.append(torch.rand(1, 512, 2, 7, 7))
    x_pre0 = torch.zeros(12, 3, 2, 112, 112).clone().detach()
    x_pre1 = torch.zeros(12, 64, 2, 56, 56).clone().detach()
    x_pre2 = torch.zeros(12, 128, 2, 28, 28).clone().detach()
    x_pre3 = torch.zeros(12, 256, 2, 14, 14).clone().detach()
    x_pre4 = torch.zeros(12, 512, 2, 7, 7).clone().detach()



    # inputs2 = torch.rand(1, 16, 19)
    net = C3D_MoCap(num_classes=6, pretrained=False,LSTM_pretrained=True)
    input_mocap_tensor = torch.rand(12, 16, 19)
    h0 = torch.zeros(2, input_mocap_tensor.size(0), 30).clone().detach()
    c0 = torch.zeros(2, input_mocap_tensor.size(0), 30).clone().detach()

    outputs, x_pre0,x_pre1,x_pre2,x_pre3,x_pre4, h,c = net.forward(inputs1, input_mocap_tensor, x_pre0,x_pre1,x_pre2,x_pre3,x_pre4, h0, c0)

    FLOPs, params = profile(net, (inputs1, input_mocap_tensor,x_pre0,x_pre1,x_pre2,x_pre3,x_pre4, h0, c0))
    print(FLOPs)  # 462.8448 GFLOPs  (G:10E9)
    print(params)  # 78.048296MParams  (M:10E6)
    print(outputs.size())
    print(outputs)