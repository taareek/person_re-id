import os
import torch
from torchvision import datasets, transforms


# function to count total objects and their corresponding samples 
def count_samples(path):
    if os.path.isdir(path):
        print(f"It is a directory")
        # list all items within the directory
        items = os.listdir(path)

        # counting 
        num_dirs = 0
        num_files = 0

        for item in items:
            item_path = os.path.join(path, item)
            if os.path.isdir(item_path):
                num_dirs += 1
                
                print(f"Checking directory: {item_path}")
                # Now count files inside each subdirectory
                sub_items = os.listdir(item_path)
                num_sub_files = 0  # Counter for files in the current subdirectory
                
                # Loop through each subitem in the subdirectory
                for sub_item in sub_items:
                    sub_item_path = os.path.join(item_path, sub_item)
                    if os.path.isfile(sub_item_path):  # Check if it's a file
                        num_sub_files += 1             
                # Print the number of files in the current subdirectory
                print(f"Total files in {item}: {num_sub_files}")
                num_files += num_sub_files  # Add the number of files in this subdirectory to the total
                
            elif os.path.isfile(item_path):
                num_files += 1

        print(f"Total directory found: {num_dirs}")
        print(f"Total files found: {num_files}")

    else:
        print(f"It is not a directory")

def process_dataset(data_dir, train_all=True, batch_size=32, debug=False):
    # defining height and weight of the image size
    h = 224
    w = 224

    # defining train transformation for training and test data 
    # train transform
    train_transform_list = [
        transforms.Resize((h, w), interpolation=3),
        transforms.Pad(10),
        transforms.RandomCrop((h, w)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]

    # validation transform
    valid_transform_list = [
        transforms.Resize(size=(h, w),interpolation=3), #Image.BICUBIC
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]

    # print(train_transform_list)
    data_transforms = {
        'train': transforms.Compose(train_transform_list),
        'val': transforms.Compose(valid_transform_list),
    }
    # print(f"Data Transforms: {data_transforms}")
    train_all = ''
    if train_all:
        train_all = '_all'
    
    image_datasets = {}
    image_datasets['train'] = datasets.ImageFolder(os.path.join(data_dir, 'train' + train_all),
                                          data_transforms['train'])
    image_datasets['val'] = datasets.ImageFolder(os.path.join(data_dir, 'val'),
                                          data_transforms['val'])
    
    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size,
                                             shuffle=True)
              for x in ['train', 'val']}

    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
    class_names = image_datasets['train'].classes

    # print out dataset specific information
    if debug:
        print(f"Dataset sizes: {dataset_sizes}")
        print(f"Total classes in train data: {len(class_names)}")
        print(f"Total classes in validation data: {len(image_datasets['val'].classes)}")
        print(f"Data loaders: {dataloaders['train']}")
        # count samples 
        # path = os.path.join(data_dir, 'train'+train_all)
        # path = os.path.join(data_dir, 'val')
        # count_samples(path)
    return image_datasets, dataloaders

if __name__ == __name__:
    data_dir = "C:/Users/AI/Desktop/Re-ID/Market/pytorch"
    process_dataset(data_dir, debug=False)