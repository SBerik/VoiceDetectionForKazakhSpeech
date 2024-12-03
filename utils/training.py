import torch


def torch_logger (writer, phase, epoch, epoch_loss, epoch_metrics, metrics):
    writer.add_scalar(f'{phase}/Loss', epoch_loss, epoch)
    [writer.add_scalar(f'{phase}/{m}', epoch_metrics[m], epoch) for m in metrics.keys()]


def p_output_log(epoch, num_epochs, phase, epoch_loss, epoch_metrics, metrics):
    if phase == 'train':
        print(f'Epoch {epoch+1}/{num_epochs}')
    print(f"{phase.upper()}, Loss: {epoch_loss:.4f}, ", end="")
    for m in metrics.keys():
        print(f"{m}: {epoch_metrics[m]:.4f} ", end="")
    print() 
    if phase == 'valid':
        print('-' * 108, '\n')


def save_best_weight(model, optimizer, epoch, epoch_loss, epoch_metrics, path_to_weights, model_name):
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': epoch_loss,
        'epoch acc': epoch_metrics['Accuracy'],
        'epoch_metrics': epoch_metrics
    }, '{}/{}_{}_{:.4f}_{:.4f}.pt'.format(path_to_weights, model_name, epoch, epoch_loss, epoch_metrics['Accuracy']))

def save_checkpoint(model, optimizer, epoch, epoch_loss, epoch_metrics, checkpoint_path, model_name):
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': epoch_loss,
        'epoch acc': epoch_metrics['Accuracy'],
        'epoch_metrics': epoch_metrics
    }, '{}/checkpoint_{}_epoch_{}.pt'.format(checkpoint_path, model_name, epoch))