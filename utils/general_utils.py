from pathlib import Path
import torch.optim as optim
import shutil

optimizers = ['SGD', 'Adam']
models = ['Base', 'AlexNet']


def create_checkpoint_name(prefix,
                           model_type,
                           optimizer_type,
                           batch_size,
                           lr,
                           dropout, ):
    pattern = "{}_model-{}_optim-{}_batch-{}_lr-{}_dropout-{:.2f}"
    return pattern.format(prefix, model_type, optimizer_type, batch_size, lr, dropout)


def check_checkpoint_exist(checkpoint_path: str):
    path = Path(checkpoint_path)
    return path.exists()


def check_optimizer_type(optimizer_type):
    return optimizer_type in optimizers


def check_model_type(model_type):
    return model_type in models


def ask_for_delete(path: str):
    """
    Function for asking user to delete desired file. If answer is not in ['y', 'Y', 'YES', 'yes', 'yep'],
    when KeyboardInterrupt will be raised.

    Parameters:
    :param path: (str or PosixPath) path to object/directory for check if exists
    """
    path_item = Path(path)
    if path_item.exists():
        print("\nDIRECTORY/FILE ALREADY EXISTS : {}".format(path))
        answer = input("Do you want to continue and delete this FILE/DIRECTORY? Answer [y / n]: ")

        if answer.lower() in ['y', 'yes', 'yep']:
            if path_item.is_dir():
                shutil.rmtree(path_item)
            elif path_item.is_file():
                path_item.unlink(missing_ok=True)
            else:
                raise Exception("Path: '{}' is not a directory or a file".format(path))
            print('SUCCESSFULLY deleted object by path: {}'.format(path))
        else:
            raise KeyboardInterrupt('Decline to delete path. Program stopped')


def create_optimizer(optimizer_type,
                     model, lr, momentum=None):
    # TODO set schedule for lr decreasing
    optimizer = None
    if optimizer_type == "SGD":
        optimizer = optim.SGD(model.parameters(), lr=lr)
    elif optimizer_type == "Adam":
        optimizer = optim.Adam(model.parameters(), lr=lr)
    return optimizer
