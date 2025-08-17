import time
import torch
import argparse
from data_loader import process_dataset
from model import feature_net
from pytorch_metric_learning import losses, miners
from tqdm import tqdm
import collections
from utils import save_network, draw_curve
from optimizer import get_optimizer

#  option arguments 
parser = argparse.ArgumentParser(description='Training')
# data
parser.add_argument('--name',default='ft_ResNet50', type=str, help='output model name')
parser.add_argument('--data_dir',default='../Market/pytorch',type=str, help='training dir path')
parser.add_argument('--train_all', action='store_true', help='use all training data' )
parser.add_argument('--batch_size', default=32, type=int, help='batch size')
# optimizer
parser.add_argument('--optimizer_name', default='sgd', type=str, help="Name of the optimizer (sgd/adam)")
parser.add_argument('--lr', default=0.05, type=float, help='learning rate')
parser.add_argument('--weight_decay', default=5e-4, type=float, help='Weight decay. More Regularization Smaller Weight.')
parser.add_argument('--total_epoch', default=60, type=int, help='total training epoch')
# backbone 
parser.add_argument('--linear_num', default=512, type=int, help='feature dimension: 512 or default or 0 (linear=False)')
parser.add_argument('--dropout_rate', default=0.5, type=float, help='drop out rate')
# loss
parser.add_argument('--warm_epoch', default=0, type=int, help='the first K epoch that needs warm up')
parser.add_argument('--loss_fn', default='cross_entropy', type= str, help='use triplet loss' )

opt = parser.parse_args()

# initialize the arguments and necessary parameters 
data_dir = opt.data_dir
name = opt.name
train_all = ''
if opt.train_all:
     train_all = '_all'
loss_fn = opt.loss_fn

image_datasets, dataloaders = process_dataset(data_dir, train_all=True, batch_size=32, debug=False)
dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
inputs, classes = next(iter(dataloaders['train']))

y_loss = {}
y_loss['train'] = []
y_loss['val'] = []
y_error = {}
y_error['train'] = []
y_error['val'] = []
x_epoch = []

def train(model, loss_fn, optimizer, scheduler, num_epochs):
    """
    Train function for image-based person re-identification.

    Parameters:
    - model: torch.nn.Module, feature extractor.
    - criterion: loss function (triplet, cross-entropy, etc.).
    - optimizer: optimizer instance (SGD, Adam).
    - scheduler: learning rate scheduler.
    - num_epochs: int, total number of training epochs.
    - name: str, experiment name (used for saving model checkpoints).
    """
    
    since = time.time()
    best_model_wts = model.state_dict()
    best_acc = 0.0

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    warm_up = 0.1
    warm_iteration = round(dataset_sizes['train'] / opt.batch_size) * opt.warm_epoch

    # Define loss functions if using triplet loss
    if loss_fn == 'triplet':
        miner = miners.MultiSimilarityMiner()
        miner = miner.to(device)
        triplet_criterion = losses.TripletMarginLoss(margin=0.3).to(device)
    else:
        criterion = torch.nn.CrossEntropyLoss()

    for epoch in range(num_epochs):
        print(f"Epoch {epoch}/{num_epochs - 1}")

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  
            else:
                model.eval()   

            pbar = tqdm(total=len(dataloaders[phase].dataset), desc=f"Epoch {epoch} - {phase}")
            ordered_dict = collections.OrderedDict(Phase=phase, Loss="0", Acc="0")

            running_loss = 0.0
            running_corrects = 0.0

            for iter, data in enumerate(dataloaders[phase]):
                inputs, labels = data
                now_batch_size, _, _, _ = inputs.shape
                pbar.update(now_batch_size)  

                if now_batch_size < opt.batch_size:
                    continue  

                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                # Forward
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)

                    if loss_fn == 'triplet':
                        # here we need to take the embedding features as well 
                        embeddings = outputs
                        hard_pairs = miner(embeddings, labels)
                        loss = triplet_criterion(embeddings, labels, hard_pairs)
                    else:
                        loss = criterion(outputs, labels)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * now_batch_size
                running_corrects += torch.sum(preds == labels.data).item()

            pbar.close()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects / dataset_sizes[phase]

            print(f"{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")

            y_loss[phase].append(epoch_loss)
            y_error[phase].append(1.0 - epoch_acc)

            # Save best model weights
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = model.state_dict()

            # Save model every 10 epochs
            if phase == 'val' and epoch % 10 == 9:
                save_network(model, name, epoch + 1)

            # Draw loss curves
            if phase == 'val':
                draw_curve(x_epoch, epoch, y_loss, y_error, name)

        # Step scheduler only at the end of the training phase
        if phase == 'train':
            scheduler.step()

        time_elapsed = time.time() - since
        print(f"Training time: {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s\n")

    # Load best model weights before saving
    model.load_state_dict(best_model_wts)
    save_network(model, name, 'last')  
    return model

# parameters for the optimizer 
class Opt:
    def __init__(self):
        self.lr = 0.001
        self.weight_decay = 0.0005
        self.optimizer = opt.optimizer_name  # Change to "sgd" if needed
        self.momentum = 0.9
        self.nesterov = True
        self.total_epoch = 100
        self.cosine = False  # Use CosineAnnealingLR if True

# Initialize arguments
optimizer_params = Opt()

# defining the model 
model = feature_net(num_of_class=751, dropout_rate=opt.dropout_rate, circle=False, linear_num=opt.linear_num)

optimizer, scheduler = get_optimizer(model, optimizer_params)

# calling the train funciton 
trained_model = train(model, loss_fn=opt.loss_fn, optimizer=optimizer, scheduler=scheduler, num_epochs=opt.total_epoch)