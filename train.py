# import time
# import torch
# import torch.nn as nn 
# from data_loader import process_dataset
# from model import feature_net
# from pytorch_metric_learning import losses, miners
# from tqdm import tqdm
# import collections
# from utils import save_network, draw_curve

# data_dir = "C:/Users/AI/Desktop/Re-ID/Market/pytorch"
# image_datasets, dataloaders = process_dataset(data_dir, train_all=True, batch_size=32, debug=False)

# # getting insights of the dataset 
# dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
# # class_names = image_datasets['train'].classes
# # print(f"Dataset sizes:\n{dataset_sizes}")
# # print(f"Total train images: {len(image_datasets['train'])}")
# # print(f"Total validation images: {len(image_datasets['val'])}")
# # print(f"Total classes in validation data: {len(image_datasets['train'].classes)}")
# # print(f"Total classes in validation data: {len(image_datasets['val'].classes)}")

# # getting gpu
# use_gpu = torch.cuda.is_available()
# if use_gpu:
#     print(f"gpu is available")
# else:
#     print(f"You are cpu bruh..!")

# # track the time 
# since = time.time()

# inputs, classes = next(iter(dataloaders['train']))
# # print(f"Input shape: {inputs.shape}")

# print(f"Time duration: {time.time()-since}")

# # defining loss variables to stiore loss values 
# y_loss = {}
# y_loss['train'] = []
# y_loss['val'] = []
# y_error = {}
# y_error['train'] = []
# y_error['val'] = []
# x_epoch = []

# # function to flip an image 
# def flip_lr(img):
#     '''flip horizontal'''
#     inv_idx = torch.arange(img.size(3)-1,-1,-1).long().cuda()  # N x C x H x W
#     img_flip = img.index_select(3,inv_idx)
#     return img_flip

# # some params can be utlized by argparser
# batch_size = 32
# warm_epochs = 5
# dropout_rate =0.5
# criterion = None #triplet

# # defining the model 
# model = feature_net(num_of_class=751, dropout_rate=0.5, circle=False, linear_num=512)

# # define the train function
# def train(model, criterion, optimizer, scheduler, num_epochs):
#     '''
#     Main train function 

#     Parameters
#     --------
#     model: cnn
#         feature extractor
#     criterion: loss funciton
#         triplet or others
#     optimizer: instance
#         SGD or Adam
#     scheduler: instance
#         learning rate scheduler
#     num_epochs: int 
#         num of iterations for training 
#         '''
    
#     since = time.time()
#     warm_up = 0.1 # We start from the 0.1*lrRate
#     warm_iteration = round(dataset_sizes['train']/batch_size)*warm_epochs # first 5 epoch

#     embedding_size = model.classifier.linear_num

#     if criterion == 'triplet':
#         miner = miners.MultiSimilarityMiner()
#         triplet_criterion = losses.TripletMarginLoss(margin=0.3)
#     for epoch in range(num_epochs):
#         print('Epoch {}/{}'.format(epoch, num_epochs - 1))
#         for phase in ['train', 'val']:
#             if phase == 'train':
#                 model.train(True)
#             else:
#                 model.train(False)
#             # Phases 'train' and 'val' are visualized in two separate progress bars
#             pbar = tqdm()
#             pbar.reset(total=len(dataloaders[phase].dataset))
#             ordered_dict = collections.OrderedDict(phase="", Loss="", Acc="")

#             running_loss = 0.0
#             running_corrects = 0.0
#             # Iterate over data 
#             for iter, data in enumerate(dataloaders[phase]):
#                 # get the inputs 
#                 inputs, labels = data
#                 now_batch_size, c, h, w = inputs.shape
#                 pbar.update(now_batch_size)   # update the pbar in the last batch
#                 # skip the last batch if it is not equals to defined batch size
#                 # it will ensure to avoid shape mismatch during the training 
#                 if now_batch_size < batch_size:
#                     continue
#                 if use_gpu:
#                     inputs = inputs.cuda()
#                     lables = labels.cuda()
#                 # zero the parameter gradients 
#                 optimizer.zero_grad()

#                 # forward pass 
#                 if phase == 'val':
#                     with torch.no_grad():
#                         outputs = model(inputs)
#                 else:
#                     outputs = model(inputs)
#                 # 
#                 _, preds = torch.max(outputs.data, 1)
#                 loss = criterion(preds, labels)
#                 if phase == 'train':
#                     loss.backward()
#                     optimizer.step()
#                 # adding loss
#                 running_loss += loss.data[0]
#                 ordered_dict["Loss"] = f"{loss.data[0]:.4f}"
#                 running_corrects += float(torch.sum(preds == labels.data))
#             # Refresh the progress bar in every batch
#             ordered_dict["phase"] = phase
#             ordered_dict[
#                 "Acc"
#             ] = f"{(float(torch.sum(preds == labels.data)) / now_batch_size):.4f}"
#             pbar.set_postfix(ordered_dict=ordered_dict)
        
#             epoch_loss = running_loss / dataset_sizes[phase]
#             epoch_acc = running_corrects / dataset_sizes[phase]
#             ordered_dict["phase"] = phase
#             ordered_dict["Loss"] = f"{epoch_loss:.4f}"
#             ordered_dict["Acc"] = f"{epoch_acc:.4f}"
#             pbar.set_postfix(ordered_dict=ordered_dict)
#             pbar.close()

#             y_loss[phase].append(epoch_loss)
#             y_error[phase].append(1.0-epoch_acc)   
            
#             if phase == 'val' and epoch % 10 == 9:
#                 last_model_wts = model.state_dict()
#                 save_network(model, dir_name, epoch + 1)
#             if phase == 'val':
#                 draw_curve(epoch, y_loss, y_error)
#             if phase == 'train':
#                 scheduler.step()
        
#         time_elapsed = time.time() - since
#         print('Training complete in {:.0f}m {:.0f}s'.format(
#             time_elapsed // 60, time_elapsed % 60))
#         print()
#     time_elapsed = time.time() - since
#     print('Training complete in {:.0f}m {:.0f}s'.format(
#         time_elapsed // 60, time_elapsed % 60))
    
#     save_network(model, dir_name, 'last') 
#     return model 

# exp_lr_scheduler = optim.lr_scheduler.StepLR(optimizer_ft, step_size=opt.total_epoch*2//3, gamma=0.1)
# if opt.cosine:
#     exp_lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer_ft, opt.total_epoch, eta_min=0.01*opt.lr) 

# optimizer_ft = optim_name([
#              {'params': base_params, 'lr': 0.1*opt.lr},
#              {'params': classifier_params, 'lr': opt.lr}
#          ], weight_decay=opt.weight_decay, momentum=0.9, nesterov=True)