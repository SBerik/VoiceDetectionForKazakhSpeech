import torch
from collections import defaultdict

def configure_optimizer(cfg, model):
    assert cfg['training']["optim"] in ['Adam', 'SGD'], "Invalid optimizer type"
    return (torch.optim.Adam if cfg['training']["optim"] == 'Adam' else torch.optim.SGD) (model.parameters(), 
                 lr=cfg['training']["lr"], weight_decay=cfg['training']["weight_decay"])


def torch_logger (writer, epoch, phase, epoch_loss, epoch_metrics, metrics):
    writer.add_scalar(f'{phase}/Loss', epoch_loss, epoch)
    [writer.add_scalar(f'{phase}/{m}', epoch_metrics[m], epoch) for m in metrics.keys()]


def p_output_log(num_epochs, epoch, phase, epoch_loss, epoch_metrics, metrics):
    if phase == 'train':
        print(f'Epoch {epoch+1}/{num_epochs}')
    print(f"{phase.upper()}, Loss: {epoch_loss:.4f}, ", end="")
    for m in metrics.keys():
        print(f"{m}: {epoch_metrics[m]:.4f} ", end="")
    print() 
    if phase == 'valid':
        print('-' * 108, '\n')


def getNumParams (model):
    num_params = sum(p.numel() for p in model.parameters())
    return num_params


def metadata_info (model, dtype = 'float32') -> None:
    num_params = getNumParams(model)
    if dtype == "float32":
        model_size = (num_params/1024**2) * 4
    elif dtype == "float16" or dtype == "bfloat16":
        model_size = (num_params/1024**2) * 2
    elif dtype == "int8":
        model_size = (num_params/1024**2) * 1
    else:
        raise ValueError(f"Unsupported dtype '{dtype}'. Supported dtypes are 'float32', 'float16', 'bfloat16', and 'int8'.")
    
    print(f"Trainable parametrs: {num_params}")
    print("Size of model: {:.2f} MB, in {}".format(model_size, dtype), '\n')


class EpochState:
    '''
    Class: contains for current epoc losses and metrics 
    '''
    def __init__ (self, metrics: dict):
        phases = ['train', 'val']
        self.state = {phase: {'loss': float('inf')} for phase in phases}
        for m in metrics:
            for phase in phases:
                self.state[phase][m] = 0.0
                
    def update (self, loss, phase: str, metrics_val:dict):
        self.state[phase]['loss'] = loss
        self.state[phase] = metrics_val

# def torch_logger(writer, epoch, train_loss, val_loss, train_accuracy, val_accuracy):
#     # Объединяем train и val для Loss
#     writer.add_scalars('Loss', {'Train': train_loss, 'Validation': val_loss}, epoch)
    
#     # Объединяем train и val для Accuracy
#     writer.add_scalars('Accuracy', {'Train': train_accuracy, 'Validation': val_accuracy}, epoch)
