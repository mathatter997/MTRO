import torch


def get_device():
    if torch.cuda.is_available():
        device = "cuda:{}".format(torch.cuda.current_device())
    else:
        device = "cpu"
    print("Use device: ", device)
    return device  # return "cpu"
