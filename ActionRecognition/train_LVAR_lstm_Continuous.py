#dataset:https://www.crcv.ucf.edu/data/UCF101.php

import glob
# from datetime import datetime
# import socket
import os
import timeit

import torch
from tensorboardX import SummaryWriter
from torch import nn, optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset_Continuous import VideoDataset
# todo 注意修改此处：C3D_MoCap3_LVAR_lstm是决策层融合，C3D_MoCap3_LVAR_lstm2是特征层融合
from network import C3D_MoCap3_LVAR_lstm2 as C3D_MoCap

# Use GPU if available else revert to CPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Device being used:", device)

nEpochs = 300  # Number of epochs for training
resume_epoch = 132  # Default is 0, change if want to resume 接着原来第resume_epoch次继续训练，0就是从头训练
useTest = True  # See evolution of the test set when training
nTestInterval = 10  # Run on test set every nTestInterval epochs
snapshot = 12  # Store a model every snapshot epochs
lr = 1e-3  # Learning rate #lstm中lr不能为0.1，使得直接h变成了nan
# lr = 5*1e-5  # Learning rate--run12_0:5*1e-5; run12_1:1e-4; run12_3:1e-3

dataset = 'ContiACT'
# preTraindataset = '0504_VideoandMocapData'
preTraindataset = 'ContiACT'

cliplen = 16
batch_clip = 80
batch_size = 10  # 每个视频抽出clip_len+batch_clip帧，一次网络吞进去clip_len，一个视频进行了batch_clip次训练

if dataset == 'all_withgloves_VideoandMocap':
    num_classes = 3
elif dataset == '0330_VideoandMocapData':
    num_classes = 4
elif dataset == '0504_VideoandMocapData' or dataset == 'ContiACT':
    num_classes = 6
else:
    print('Wrong--dataset')
    raise NotImplementedError

# hidden_size = 19*3
# hidden_size = 19*16
hidden_size = 30
lstm_layers = 2
save_dir_root = os.path.join(os.path.dirname(os.path.abspath(__file__)))
# os.path.abspath()取绝对路径（如果是一个绝对路径，就返回，如果不是绝对路径，根据编码执行getcwd/getcwdu.然后把path和当前工作路径连接起来）
exp_name = os.path.dirname(os.path.abspath(__file__)).split('/')[-1]  # os.path.abspath(__file__)取当前文件的绝对路径

if resume_epoch != 0:
    # sorted() 函数对所有可迭代的对象进行排序操作。
    # glob.glob()匹配所有的符合条件的文件，并将其以list的形式返回
    runs = sorted(glob.glob(os.path.join(save_dir_root, 'run', 'run14_*')))
    run_id = int(runs[-1].split('_')[-1]) if runs else 0
else:
    runs = sorted(glob.glob(os.path.join(save_dir_root, 'run', 'run14_*')))
    run_id = int(runs[-1].split('_')[-1]) + 1 if runs else 0

save_dir = os.path.join(save_dir_root, 'run', 'run14_' + str(run_id))
# modelName = 'C3D_MoCap19f'
modelName = 'C3D_MoCapLvarLstm2'
saveName = modelName + '-' + dataset
preTrainName = modelName + '-' + preTraindataset
print(save_dir)

def train_model(dataset=dataset, save_dir=save_dir, num_classes=num_classes, lr=lr,
                num_epochs=nEpochs, save_epoch=snapshot, useTest=useTest, test_interval=nTestInterval):
    """
        Args:
            num_classes (int): Number of classes in the data
            num_epochs (int, optional): Number of epochs to train for.
    """

    model = C3D_MoCap.C3D_MoCap(num_classes=num_classes, pretrained=True, LSTM_pretrained=True)
    train_params = [{'params': C3D_MoCap.get_1x_lr_params(model), 'lr': lr},
                    {'params': C3D_MoCap.get_10x_lr_params(model), 'lr': lr * 10}]

    criterion = nn.CrossEntropyLoss()  # standard crossentropy loss for classification
    optimizer = optim.SGD(train_params, lr=lr, momentum=0.9, weight_decay=5e-4)

    # the scheduler divides the lr by 10 every 'step_size' epochs  -->表示scheduler.step()每调用step_size=10次，对应的学习率就会按照策略gamma=0.1调整一次
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)

    model.to(device)
    # 将从中断的训练中继续训练需要重新加载检查点，一些优化器（比如adam）的一些变量也需要被保存到检查点中，而在使用load_state_dict()还原时，有可能会将这些变量还原到CPU上。
    # 为了解决上述问题，在还原检查点之前，将模型转到GPU，即将model.to(device)放到checkpoint加载之前。
    if resume_epoch == 0:
        print("Training {} from scratch...".format(modelName))
    else:
        checkpoint = torch.load(os.path.join(save_dir, 'models', preTrainName + '_epoch-' + str(resume_epoch - 1) + '.pth.tar'),
                       map_location=lambda storage, loc: storage)   # Load all tensors onto the CPU
        print("Initializing weights from: {}...".format(
            os.path.join(save_dir, 'models', preTrainName + '_epoch-' + str(resume_epoch - 1) + '.pth.tar')))
        model.load_state_dict(checkpoint['state_dict'])
        # optimizer.load_state_dict(checkpoint['opt_dict'])

    print('Total params: %.2fM' % (sum(p.numel() for p in model.parameters()) / 1000000.0))
    criterion.to(device)
    # log_dir = os.path.join(save_dir, 'models', datetime.now().strftime('%b%d_%H-%M-%S') + '_' + socket.gethostname())
    log_dir = os.path.join(save_dir, 'models',  'TrainRecord')
    writer = SummaryWriter(log_dir=log_dir)

    print('Training model on {} dataset...'.format(dataset))
    # torch.utils.data.DataLoader()函数中：   shuffle为true表示打乱顺序，不按照字典序读取所有“视频”；batch_size = 12表示一次输入12组视频，求loss，进行梯度下降。
    #  默认值为DataLoader(VideoDataset(dataset=dataset, split='train', clip_len=16), batch_size=12, shuffle=True, num_workers=0)
    train_dataloader = DataLoader(VideoDataset(dataset=dataset, split='train', clip_len=cliplen, batch_clip=batch_clip), batch_size=batch_size, shuffle=True, num_workers=0)
    val_dataloader   = DataLoader(VideoDataset(dataset=dataset, split='val',  clip_len=cliplen, batch_clip=batch_clip), batch_size=batch_size, num_workers=0)
    test_dataloader  = DataLoader(VideoDataset(dataset=dataset, split='test', clip_len=cliplen, batch_clip=batch_clip), batch_size=batch_size, num_workers=0)

    trainval_loaders = {'train': train_dataloader, 'val': val_dataloader}
    trainval_sizes = {x: len(trainval_loaders[x].dataset) for x in ['train', 'val']}
    test_size = len(test_dataloader.dataset)

    for epoch in range(resume_epoch, num_epochs):
        print("第%d个epoch的学习率：%f" % (epoch, optimizer.param_groups[0]['lr']))
        # each epoch has a training and validation step
        for phase in ['train', 'val']:
            start_time = timeit.default_timer()

            # reset the running loss and corrects
            running_loss = 0.0
            running_corrects = 0.0

            # set model to train() or eval() mode depending on whether it is trained
            # or being validated. Primarily affects layers such as BatchNorm or Dropout.
            if phase == 'train':
                # scheduler.step()  #  is to be called once every epoch during training，新版本要求放在optimizer后
                model.train()
            else:
                model.eval()  # 评估模式,而非训练模式.在对模型进行评估时，应该配合使用with torch.no_grad() 与 model.eval()

            for inputs_twi, inputs_buffer_csv_twi, labels_twi in tqdm(trainval_loaders[phase]):
                x_pre0 = torch.zeros(inputs_twi.shape[0], 3, 2, 112, 112).clone().detach()
                x_pre1 = torch.zeros(inputs_twi.shape[0], 64, 2, 56, 56).clone().detach()
                x_pre2 = torch.zeros(inputs_twi.shape[0], 128, 2, 28, 28).clone().detach()
                x_pre3 = torch.zeros(inputs_twi.shape[0], 256, 2, 14, 14).clone().detach()
                x_pre4 = torch.zeros(inputs_twi.shape[0], 512, 2, 7, 7).clone().detach()
                h = torch.zeros(lstm_layers, inputs_buffer_csv_twi.shape[0], hidden_size).clone().detach()
                c = torch.zeros(lstm_layers, inputs_buffer_csv_twi.shape[0], hidden_size).clone().detach()
                # todo: 注意此处进行了修改以对一个视频连续训练, 也添加了lstm的h和c
                for i in range(batch_clip):
                    inputs = torch.index_select(inputs_twi, 2, torch.tensor(range(i, i + cliplen)))
                    # TODO: 考虑修改了range(i, i + cliplen, 2)表示选取0，2，4，..
                    inputs_buffer_csv = torch.index_select(inputs_buffer_csv_twi, 1, torch.tensor(range(i, i + cliplen)))
                    labels = (torch.index_select(labels_twi, 1, torch.tensor(range(i, i + 1)))).reshape(inputs_buffer_csv.shape[0])
                    # move inputs and labels to the device the training is taking place on
                    inputs = Variable(inputs, requires_grad=True).to(device)
                    labels = Variable(labels).to(device)
                    inputs_buffer_csv = Variable(inputs_buffer_csv, requires_grad=True).to(device)
                    x_pre0 = Variable(x_pre0).to(device)
                    x_pre1 = Variable(x_pre1).to(device)
                    x_pre2 = Variable(x_pre2).to(device)
                    x_pre3 = Variable(x_pre3).to(device)
                    x_pre4 = Variable(x_pre4).to(device)
                    h = Variable(h, requires_grad=True).to(device)
                    c = Variable(c, requires_grad=True).to(device)
                    optimizer.zero_grad()  # 将梯度归零

                    if phase == 'train':
                        outputs, x_pre0,x_pre1,x_pre2,x_pre3,x_pre4, h,c = model(inputs, inputs_buffer_csv, x_pre0,x_pre1,x_pre2,x_pre3,x_pre4, h,c)
                    else:
                        with torch.no_grad():  # 表示不计算梯度
                            outputs, x_pre0,x_pre1,x_pre2,x_pre3,x_pre4, h,c = model(inputs, inputs_buffer_csv, x_pre0,x_pre1,x_pre2,x_pre3,x_pre4, h,c)

                    probs = nn.Softmax(dim=1)(outputs)
                    preds = torch.max(probs, 1)[1]
                    loss = criterion(outputs, labels.long())
                    # if i == batch_clip-1:
                    #     print("loss:", loss)

                    if phase == 'train':
                        loss.backward()  # 反向传播计算得到每个参数的梯度值
                        optimizer.step()  # 通过梯度下降执行一步参数更新

                    #todo 需要更新关于running_loss和running_corrects的处理方式---是否需要除以cliplen？
                    running_loss += loss.item() * inputs_twi.size(0)
                    running_corrects += torch.sum(preds == labels.data)
            scheduler.step()
            epoch_loss = running_loss / (trainval_sizes[phase]*batch_clip)
            epoch_acc = running_corrects.double() / (trainval_sizes[phase]*batch_clip)

            if phase == 'train':
                writer.add_scalar('data/train_loss_epoch', epoch_loss, epoch)
                writer.add_scalar('data/train_acc_epoch', epoch_acc, epoch)
            else:
                writer.add_scalar('data/val_loss_epoch', epoch_loss, epoch)
                writer.add_scalar('data/val_acc_epoch', epoch_acc, epoch)

            stop_time = timeit.default_timer()
            print("[{}] Epoch: {}/{} Loss: {} Acc: {}".format(phase, epoch + 1, nEpochs, epoch_loss, epoch_acc))
            print("Execution time: " + str(stop_time - start_time) + "\n")

        if epoch % save_epoch == (save_epoch - 1):
            torch.save({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'opt_dict': optimizer.state_dict(),
            }, os.path.join(save_dir, 'models', saveName + '_epoch-' + str(epoch) + '.pth.tar'))
            print("Save model at {}\n".format(os.path.join(save_dir, 'models', saveName + '_epoch-' + str(epoch) + '.pth.tar')))

        if useTest and epoch % test_interval == (test_interval - 1):
            model.eval()
            start_time = timeit.default_timer()
            running_loss = 0.0
            running_corrects = 0.0
            for inputs_twi, inputs_buffer_csv_twi, labels_twi in tqdm(test_dataloader):
                x_pre0 = torch.zeros(inputs_twi.shape[0], 3, 2, 112, 112).clone().detach()
                x_pre1 = torch.zeros(inputs_twi.shape[0], 64, 2, 56, 56).clone().detach()
                x_pre2 = torch.zeros(inputs_twi.shape[0], 128, 2, 28, 28).clone().detach()
                x_pre3 = torch.zeros(inputs_twi.shape[0], 256, 2, 14, 14).clone().detach()
                x_pre4 = torch.zeros(inputs_twi.shape[0], 512, 2, 7, 7).clone().detach()
                h = torch.zeros(lstm_layers, inputs_buffer_csv_twi.shape[0], hidden_size).clone().detach()
                c = torch.zeros(lstm_layers, inputs_buffer_csv_twi.shape[0], hidden_size).clone().detach()
                for i in range(batch_clip):
                    inputs = torch.index_select(inputs_twi, 2, torch.tensor(range(i, i + cliplen)))
                    inputs_buffer_csv = torch.index_select(inputs_buffer_csv_twi, 1, torch.tensor(range(i, i + cliplen)))
                    labels = (torch.index_select(labels_twi, 1, torch.tensor(range(i, i + 1)))).reshape(inputs_buffer_csv.shape[0])
                    inputs = inputs.to(device)
                    labels = labels.to(device)
                    inputs_buffer_csv = inputs_buffer_csv.to(device)
                    x_pre0 = x_pre0.to(device)
                    x_pre1 = x_pre1.to(device)
                    x_pre2 = x_pre2.to(device)
                    x_pre3 = x_pre3.to(device)
                    x_pre4 = x_pre4.to(device)
                    h = Variable(h).to(device)
                    c = Variable(c).to(device)
                    with torch.no_grad():
                        outputs, x_pre0,x_pre1,x_pre2,x_pre3,x_pre4, h,c = model(inputs,inputs_buffer_csv,x_pre0,x_pre1,x_pre2,x_pre3,x_pre4,h,c)
                    probs = nn.Softmax(dim=1)(outputs)
                    preds = torch.max(probs, 1)[1]
                    loss = criterion(outputs, labels.long())

                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / (test_size*batch_clip)
            epoch_acc = running_corrects.double() / (test_size*batch_clip)

            writer.add_scalar('data/test_loss_epoch', epoch_loss, epoch)
            writer.add_scalar('data/test_acc_epoch', epoch_acc, epoch)

            stop_time = timeit.default_timer()
            print("[test] Epoch: {}/{} Loss: {} Acc: {}".format(epoch+1, nEpochs, epoch_loss, epoch_acc))
            print("Execution time: " + str(stop_time - start_time) + "\n")

    writer.close()


if __name__ == "__main__":
    train_model()
