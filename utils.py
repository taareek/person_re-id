import os
import torch
import matplotlib.pyplot as plt 

def save_network(network, dirname, epoch_label):
    """Save the model's state dictionary to a specified directory."""
    
    if isinstance(epoch_label, int):
        save_filename = 'net_%03d.pth' % epoch_label
    else:
        save_filename = 'net_%s.pth' % epoch_label
    
    save_path = os.path.join('./model', dirname, save_filename)
    
    # Ensure directory exists
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # Save the model's state dictionary
    torch.save(network.state_dict(), save_path)

    # Move the network back to GPU (if available)
    if torch.cuda.is_available():
        network.cuda()

    print(f"Model saved at: {save_path}")

# x_epoch = []
fig = plt.figure()
ax0 = fig.add_subplot(121, title="loss")
ax1 = fig.add_subplot(122, title="top1err")
def draw_curve(x_epoch, current_epoch, y_loss, y_err, name="resnet-50_exp"):
    x_epoch.append(current_epoch)
    ax0.plot(x_epoch, y_loss['train'], 'bo-', label='train')
    ax0.plot(x_epoch, y_loss['val'], 'ro-', label='val')
    ax1.plot(x_epoch, y_err['train'], 'bo-', label='train')
    ax1.plot(x_epoch, y_err['val'], 'ro-', label='val')
    if current_epoch == 0:
        ax0.legend()
        ax1.legend()
    save_path = os.path.join('./model', name)  # Construct the directory path
    os.makedirs(save_path, exist_ok=True)  # Create directory if it doesn't exis
    fig.savefig( os.path.join('./model',name,'train.jpg'))