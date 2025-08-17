import torch 
import torch.nn as nn 
from torch.nn import init
from torchvision import models
from torch.autograd import Variable

# defining weight initialization for the linear layer using kaiming-he 
def weights_init_kaiming(m):
    classname = m.__class__.__name__
    # print(classname)
    if classname.find('Conv') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in') # For old pytorch, you may use kaiming_normal.
    elif classname.find('Linear') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_out')
    elif classname.find('BatchNorm1d') != -1:
        init.normal_(m.weight.data, 1.0, 0.02)
    if hasattr(m, 'bias') and m.bias is not None:
        init.constant_(m.bias.data, 0.0)

# defining weight initialization for the classification layer using normal 
def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        init.normal_(m.weight.data, std=0.001)
        init.constant_(m.bias.data, 0.0)


def activate_drop(m):
    classname = m.__class__.__name__
    if classname.find('Drop') != -1:
        m.p = 0.1
        m.inplace = True

# defining the custom classifier block 
# |--Linear--|--bn--|--relu--|--Linear--|
class ClassBlock(nn.Module):
    def __init__(self, input_dim, class_num, droprate, relu=False, bnorm=True, linear=512, return_f = False):
        super(ClassBlock, self).__init__()
        self.return_f = return_f
        add_block = []
        if linear>0:
            add_block += [nn.Linear(input_dim, linear)]
        else:
            linear = input_dim
        if bnorm:
            add_block += [nn.BatchNorm1d(linear)]
        if relu:
            add_block += [nn.LeakyReLU(0.1)]
        if droprate>0:
            add_block += [nn.Dropout(p=droprate)]
        add_block = nn.Sequential(*add_block)
        add_block.apply(weights_init_kaiming)

        classifier = []
        classifier += [nn.Linear(linear, class_num)]
        classifier = nn.Sequential(*classifier)
        classifier.apply(weights_init_classifier)

        self.linear_num = linear
        self.add_block = add_block
        self.classifier = classifier
    def forward(self, x):
        x = self.add_block(x)
        if self.return_f:
            f = x
            x = self.classifier(x)
            return [x,f]
        else:
            x = self.classifier(x)
            return x

# defining feature extractor class to extract meaningful features from the object of interests 
class feature_net(nn.Module):
    def __init__(self, num_of_class=751, dropout_rate=0.5, circle=False, linear_num=512, return_f=False):
        super(feature_net, self).__init__()
        # defining resnet-50
        ft_extractor = models.resnet50(pretrained=True)
        # resnet_50 = models.resnet50(weights="IMAGENET1K_V1")
        # resnet_50 = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)

        # change Avg pooling to Global pooling layer
        ft_extractor.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.model = ft_extractor
        self.circle = circle
        self.return_f = return_f
        self.classifier = ClassBlock(2048, num_of_class, dropout_rate, linear=linear_num, return_f= return_f)

    def forward(self, x):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)
        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)
        x = self.model.avgpool(x)
        # x = torch.squeeze(x)
        x = x.view(x.size(0), x.size(1))
        x = self.classifier(x)
        return x
    
'''
# debug model structure
# Run this code with:
python model.py
'''
if __name__ == '__main__':
# Here I left a simple forward function.
# Test the model, before you train it. 
    net = feature_net(751)
    print(net)
    input = Variable(torch.FloatTensor(8, 3, 224, 224))
    output = net(input)
    print(f"Output shape: {output.shape}")
    print(f"Embedding size: {net.classifier.linear_num}")