import os
import torch
import time 
import math 
from model import feature_net
import torch.nn as nn 
from torchvision import datasets, models, transforms
from tqdm import tqdm
import scipy.io
import argparse

# defining device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# defining height and width of the test image 
h = 224
w = 224

# defining the test dirctory 
# test_dir = "C:/Users/AI/Desktop/Re-ID/Market/pytorch"


parser = argparse.ArgumentParser(description='Test')
parser.add_argument('--name', default='ft_ResNet50', type=str, help='save model path')
parser.add_argument('--test_dir',default='../Market/pytorch',type=str, help='./test_data')
parser.add_argument('--batch_size', default=32, type=int, help='batchsize')
parser.add_argument('--linear_num', default=512, type=int, help='feature dimension: 512 or default or 0 (linear=False)')
parser.add_argument('--which_epoch',default='last', type=str, help='0,1,2,3...or last')
parser.add_argument('--multi', action='store_true', help='use multiple query' )
parser.add_argument('--ms',default='1', type=str,help='multiple_scale: e.g. 1 1,1.1  1,1.1,1.2')

opt = parser.parse_args()

# getting the parameters
data_dir = opt.test_dir
multi = opt.multi
batch_size = opt.batch_size
linear_num = opt.linear_num
# load the model 
name = opt.name
which_epoch = opt.which_epoch

# defining multi-scale test
ms = opt.ms
print('We use the scale: %s'%ms)
str_ms = ms.split(',')
ms = []
for s in str_ms:
    s_f = float(s)
    ms.append(math.sqrt(s_f))

# defining the data transformation for the test data 
data_transforms = transforms.Compose([
        transforms.Resize((h, w), interpolation=3),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

# Data loader
if multi:
    image_datasets = {x: datasets.ImageFolder( os.path.join(data_dir,x) ,data_transforms) for x in ['gallery','query','multi-query']}
    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size,
                                             shuffle=False) for x in ['gallery','query','multi-query']}
else:
    image_datasets = {x: datasets.ImageFolder( os.path.join(data_dir,x) ,data_transforms) for x in ['gallery','query']}
    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size,
                                             shuffle=False) for x in ['gallery','query']}

query_class_names = image_datasets['query'].classes
# print(f"Class names of query dataset: {query_class_names}")

gallery_class_names = image_datasets['query'].classes
# print(f"Class names of query dataset: {gallery_class_names}")


# function to load the model 
def load_network(network, name=name, which_epoch=which_epoch, device=device):
    save_path = os.path.join('./model', name, f'net_0{which_epoch}.pth')
    # Load the model weights to the specified device (CPU or GPU)
    # network.load_state_dict(torch.load(save_path, map_location=torch.device(device)))
    network.load_state_dict(torch.load(save_path, map_location=torch.device(device), weights_only=True))

    # Optional: Compile the model for optimization (only if using PyTorch 2.0+ and a compatible GPU)
    # if torch.cuda.get_device_capability()[0] >= 7:  # Ampere or newer (RTX 30 series+)
    #     print("Compiling model for performance optimization...")
    #     torch.set_float32_matmul_precision('high')  # Improves matrix multiplications
    #     network = torch.compile(network, mode="default", dynamic=True)  # Enable PyTorch 2.0 compilation
    
    return network

# function to flip 
def fliplr(img):
    '''flip horizontal'''
    inv_idx = torch.arange(img.size(3)-1,-1,-1).long()  # N x C x H x W
    img_flip = img.index_select(3,inv_idx)
    return img_flip

# function to extract features 
def extract_features(model, dataloader):
    pbar = tqdm()
    for iter, data in enumerate(dataloader):
        img, label = data
        n, c, h, w = img.size()
        pbar.update(n)

        ff = torch.FloatTensor(n, linear_num).zero_().to(device)
        for i in range(2):
            if (i==1):
                img = fliplr(img)
            input_img = img.to(device)
            for scale in ms:
                if scale != 1:
                    # bicubic is only  available in pytorch>= 1.1
                    input_img = nn.functional.interpolate(input_img, scale_factor=scale, mode='bicubic', align_corners=False)
                    input_img = input_img.to(device)
                logits, embeddings = model(input_img) 
                ff += embeddings
        fnorm = torch.norm(ff, p=2, dim=1, keepdim=True)
        ff = ff.div(fnorm.expand_as(ff))
        
        if iter == 0:
            features = torch.FloatTensor(len(dataloader.dataset), ff.shape[1])
        #features = torch.cat((features,ff.data.cpu()), 0)
        start = iter* batch_size
        end = min( (iter+1)* batch_size, len(dataloader.dataset))
        features[ start:end, :] = ff
    
    pbar.close()
    return features

# function to get id 
def get_id(img_path):
    camera_id = []
    labels = []
    for path, v in img_path:
        #filename = path.split('/')[-1]
        filename = os.path.basename(path)
        label = filename[0:4]
        camera = filename.split('c')[1]
        if label[0:2]=='-1':
            labels.append(-1)
        else:
            labels.append(int(label))
        camera_id.append(int(camera[0]))
    return camera_id, labels

gallery_path = image_datasets['gallery'].imgs
query_path = image_datasets['query'].imgs

gallery_cam,gallery_label = get_id(gallery_path)
query_cam,query_label = get_id(query_path)

if multi:
    mquery_path = image_datasets['multi-query'].imgs
    mquery_cam,mquery_label = get_id(mquery_path)

# load the trained model 
model_structure = feature_net(num_of_class=751, return_f=True)
model = load_network(model_structure)

# Change to test mode
model = model.to(device)
model = model.eval()

# Extract feature
since = time.time()
with torch.no_grad():
    gallery_feature = extract_features(model, dataloaders['gallery'])
    query_feature = extract_features(model, dataloaders['query'])
    if multi:
        mquery_feature = extract_features(model,dataloaders['multi-query'])

time_elapsed = time.time() - since
print('Training complete in {:.0f}m {:.2f}s'.format(
            time_elapsed // 60, time_elapsed % 60))

# Save to Matlab for check
result = {'gallery_f':gallery_feature.numpy(),'gallery_label':gallery_label,'gallery_cam':gallery_cam,'query_f':query_feature.numpy(),'query_label':query_label,'query_cam':query_cam}
scipy.io.savemat('pytorch_result.mat',result)

print(name)
result = './model/%s/result.txt'%name
# os.system('python evaluate_gpu.py | tee -a %s'%result)   # it is linux command 
os.system(f'python evaluate_gpu.py >> {result} 2>&1')


if multi:
    result = {'mquery_f':mquery_feature.numpy(),'mquery_label':mquery_label,'mquery_cam':mquery_cam}
    scipy.io.savemat('multi_query.mat',result)