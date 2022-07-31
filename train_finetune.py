# encoding:utf-8
import torch
import torchvision
import torch.optim as optim
import torchvision.transforms as transforms
import torch.nn as nn
import hbp_model
import os
import torch.backends.cudnn as cudnn


def train(epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(trainloader):
        data, target = data.cuda(), target.cuda()
        model.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 100 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tLR: {}'.format(
                epoch, batch_idx * len(data), len(trainloader.dataset),
                       100. * batch_idx / len(trainloader), loss.data.item(),
                optimizer.param_groups[0]['lr']))


def test():
    model.eval()
    test_loss = 0
    correct = 0
    for data, target in testloader:
        data, target = data.cuda(), target.cuda()
        output = model(data)
        test_loss += criterion(output, target).data.item()
        pred = output.data.max(1, keepdim=True)[1]
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()
    test_loss /= len(testloader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        test_loss * 8., correct, len(testloader.dataset),
        100.0 * float(correct) / len(testloader.dataset)))


def adjust_learning_rate(optimizer, epoch):
    if epoch % 40 == 0:
        for param_group in optimizer.param_groups:
            param_group['lr'] = param_group['lr'] * 0.1


if __name__ == '__main__':
    print('GPU是否可用：', torch.cuda.is_available())

    os.environ["CUDA_VISIBLE_DEVICES"] = "1"

    trainset = torchvision.datasets.ImageFolder(
        'data/NEU-DET/training_set',
        transform=transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        ]))

    valset = torchvision.datasets.ImageFolder(
        'data/NEU-DET/val_set',
        transform=transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        ])
    )
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=8,
                                              shuffle=True, num_workers=0)

    testloader = torch.utils.data.DataLoader(valset, batch_size=8,
                                             shuffle=False, num_workers=0)
    cudnn.benchmark = True

    model = hbp_model.Net()
    model.cuda()

    pretrained = True
    if pretrained:
        pre_dic = torch.load('firststep.pth')
        model.load_state_dict(pre_dic)

    criterion = nn.NLLLoss()
    lr = 1e-2

    optimizer = optim.SGD([
        {'params': model.features.parameters(), 'lr': lr},
        {'params': model.proj0.parameters(), 'lr': lr},
        {'params': model.proj1.parameters(), 'lr': lr},
        {'params': model.proj2.parameters(), 'lr': lr},

        {'params': model.fc_concat.parameters(), 'lr': lr},
    ], lr=0.01, momentum=0.9, weight_decay=1e-5)

    for epoch in range(1, 101):
        train(epoch)
        test()
        adjust_learning_rate(optimizer, epoch)

    torch.save(model.state_dict(), 'finalstep.pth')