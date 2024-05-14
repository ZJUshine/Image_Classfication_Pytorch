from torchvision import models
import torch.nn as nn

def get_model(model_name ,pretrain, class_num):
    if model_name == "vgg":
        if pretrain == True:
            # 加载预训练的vgg16模型
            transfer_model = models.vgg16(pretrained=True)
            # 冻结模型的参数
            for param in transfer_model.parameters():
                param.requires_grad = False
        else:
            # 不加载预训练的vgg16模型
            transfer_model = models.vgg16(pretrained=False)

        # 修改最后一层维数，即把原来的全连接层替换成输出维数为class_num的全连接层
        transfer_model.classifier[-1] = nn.Linear(transfer_model.classifier[-1].in_features, class_num)
    
    elif model_name == "resnet50":
        if pretrain == True:
            # 加载预训练的resnet50模型
            transfer_model = models.resnet50(pretrained=True)
            # 冻结模型的参数
            for param in transfer_model.parameters():
                param.requires_grad = False
        else:
            # 不加载预训练的resnet50模型
            transfer_model = models.resnet50(pretrained=False)
        
        # 修改最后一层维数，即把原来的全连接层替换成输出维数为class_num的全连接层
        transfer_model.fc = nn.Linear(transfer_model.fc.in_features, class_num)

    elif model_name == "densenet":
        if pretrain == True:
            # 加载预训练的densenet169模型
            transfer_model = models.densenet169(pretrained=True)
            # 冻结模型的参数
            for param in transfer_model.parameters():
                param.requires_grad = False
        else:
            # 不加载预训练的densenet169模型
            transfer_model = models.densenet169(pretrained=False)

        # 修改最后一层维数，即把原来的全连接层替换成输出维数为class_num的全连接层
        transfer_model.classifier = nn.Linear(transfer_model.classifier.in_features, class_num)

    elif model_name == "efficientnet":
        if pretrain == True:
            # 加载预训练的efficientnet_b3模型
            transfer_model = models.efficientnet_b3(pretrained=True)
            # 冻结模型的参数
            for param in transfer_model.parameters():
                param.requires_grad = False
        else:
            # 不加载预训练的efficientnet_b3模型
            transfer_model = models.efficientnet_b3(pretrained=False)

        # 修改最后一层维数，即把原来的全连接层替换成输出维数为class_num的全连接层
        transfer_model.classifier[-1] = nn.Linear(transfer_model.classifier[-1].in_features, class_num)

    elif model_name == "mobilenet":
        if pretrain == True:
            # 加载预训练的mobilenet_v2模型
            transfer_model = models.mobilenet_v2(pretrained=True)
            # 冻结模型的参数
            for param in transfer_model.parameters():
                param.requires_grad = False
        else:
            # 不加载预训练的mobilenet_v2模型
            transfer_model = models.mobilenet_v2(pretrained=False)

        # 修改最后一层维数，即把原来的全连接层替换成输出维数为class_num的全连接层
        transfer_model.classifier[-1] = nn.Linear(transfer_model.classifier[-1].in_features, class_num)

    elif model_name == "vit":
        if pretrain == True:
            # 加载预训练的vit模型
            transfer_model = models.vit_b_16(pretrained=True)
            # 冻结模型的参数
            for param in transfer_model.parameters():
                param.requires_grad = False
        else:
            # 不加载预训练的vit模型
            transfer_model = models.vit_b_16(pretrained=False)

        # 修改最后一层维数，即把原来的全连接层替换成输出维数为class_num的全连接层
        transfer_model.heads.head = nn.Linear(transfer_model.heads.head.in_features, class_num)

    elif model_name == "convnext":
        if pretrain == True:
            # 加载预训练的convnext_base模型
            transfer_model = models.convnext_base(pretrained=True)
            # 冻结模型的参数
            for param in transfer_model.parameters():
                param.requires_grad = False
        else:
            # 不加载预训练的convnext_base模型
            transfer_model = models.convnext_base(pretrained=False)

        # 修改最后一层维数，即把原来的全连接层替换成输出维数为class_num的全连接层
        transfer_model.classifier[-1] = nn.Linear(transfer_model.classifier[-1].in_features, class_num)