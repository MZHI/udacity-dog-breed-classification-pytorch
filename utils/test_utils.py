# -*- coding utf-8 -*-

from tqdm import tqdm
import numpy as np


def evaluate(loader, model, criterion, use_cuda):
    # monitor test loss and accuracy
    test_loss = 0.0
    correct = 0.0
    total = 0.0

    if use_cuda:
        model.cuda()

    model.eval()

    with tqdm(loader, unit="batch") as ttepoch:
        for data, target in ttepoch:
            # move to GPU
            if use_cuda:
                data, target = data.cuda(), target.cuda()
            # forward pass: compute predicted outputs by passing inputs to the model
            output = model(data)
            # calculate the loss
            loss = criterion(output, target)
            # update average test loss
            test_loss += loss.item() * data.size(0)
            #         test_loss = test_loss + ((1 / (batch_idx + 1)) * (loss.data - test_loss))
            # convert output probabilities to predicted class
            pred = output.data.max(1, keepdim=True)[1]
            # compare predictions to true label
            correct += np.sum(np.squeeze(pred.eq(target.data.view_as(pred)).cpu().numpy()))
            total += data.size(0)
            ttepoch.set_postfix(loss=test_loss, accuracy=f"{correct * 100.0 / total:.2f}%")

    test_loss /= len(loader.sampler)
    print('Test Loss: {:.6f}\n'.format(test_loss))

    accuracy = 100.0 * correct / total
    print('\nTest Accuracy: %2d%% (%2d/%2d)' % (accuracy, correct, total))

    return accuracy, test_loss
