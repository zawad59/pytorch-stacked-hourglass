import argparse

import torch
import torch.backends.cudnn
from torch.utils.data import DataLoader
from tqdm import tqdm

from stacked_hourglass import hg1, hg2, hg8
from stacked_hourglass.datasets.mpii import mpii, print_mpii_validation_accuracy
from stacked_hourglass.losses import joints_mse_loss
from stacked_hourglass.utils.evaluation import accuracy, AverageMeter, final_preds
from stacked_hourglass.utils.transforms import fliplr, flip_back


def main(args):
    # Select the hardware device to use for inference.
    if torch.cuda.is_available():
        device = torch.device('cuda', torch.cuda.current_device())
        torch.backends.cudnn.benchmark = True
    else:
        device = torch.device('cpu')

    # Disable gradient calculations.
    torch.set_grad_enabled(False)

    # Create the model, downloading pretrained weights if necessary.
    if args.model == 'hg1':
        model = hg1(pretrained=True)
    elif args.model == 'hg2':
        model = hg2(pretrained=True)
    elif args.model == 'hg8':
        model = hg8(pretrained=True)
    else:
        raise Exception('unrecognised model name: ' + args.model)
    model = model.to(device)

    # Initialise the MPII validation set dataloader.
    val_dataset = mpii(args.image_path, is_train=False)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.workers, pin_memory=True)

    # Generate predictions for the validation set.
    predictions = validate(val_loader, model, device, args.flip)

    # Report PCKh for the predictions.
    print('\nFinal validation PCKh scores:')
    print_mpii_validation_accuracy(predictions)


def validate(val_loader, model, device, flip=True):
    losses = AverageMeter()
    accuracies = AverageMeter()
    predictions = torch.empty(len(val_loader.dataset), 16, 2)

    # A list of joints to include in the accuracy reported as part of the progress bar.
    idx = [1, 2, 3, 4, 5, 6, 11, 12, 15, 16]

    # Put the model in evaluation mode.
    model.eval()

    progress = tqdm(enumerate(val_loader), total=len(val_loader), ascii=True, leave=True)
    for i, (input, target, meta) in progress:
        # Copy data to the training device (eg GPU).
        input = input.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        target_weight = meta['target_weight'].to(device, non_blocking=True)

        # Forward pass with the model.
        output = model(input)

        # Calculate the loss (summed across all stages).
        loss = sum(joints_mse_loss(o, target, target_weight) for o in output)

        # Get the heatmaps.
        if flip:
            # If `flip` is true, perform horizontally flipped inference as well. This should
            # result in more robust predictions at the expense of additional compute.
            flip_input = fliplr(torch.tensor(input, device='cpu').numpy())
            flip_input = torch.as_tensor(flip_input, dtype=torch.float32, device=device)
            flip_output = model(flip_input)
            flip_output = flip_output[-1].cpu()
            flip_output = flip_back(flip_output)
            heatmaps = (output[-1].cpu() + flip_output) / 2
        else:
            heatmaps = output[-1].cpu()

        # Calculate PCKh from the predicted heatmaps.
        acc = accuracy(heatmaps, target.cpu(), idx)

        # Calculate locations in original image space from the predicted heatmaps.
        preds = final_preds(heatmaps, meta['center'], meta['scale'], [64, 64])
        for example_index, pose in zip(meta['index'], preds):
            predictions[example_index] = pose

        # Record accuracy and loss for this batch.
        losses.update(loss.item(), input.size(0))
        accuracies.update(acc[0].item(), input.size(0))

        # Show accuracy and loss as part of the progress bar.
        progress.set_postfix(loss=f'{losses.avg:0.6f}', acc=f'{accuracies.avg:0.3f}')

    return predictions


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch Stacked Hourglass training')
    parser.add_argument('--image-path', required=True, type=str,
                        help='path to MPII Human Pose images')
    parser.add_argument('--model', metavar='MODEL', default='hg1',
                        choices=['hg1', 'hg2', 'hg8'],
                        help='model architecture')
    parser.add_argument('--workers', default=4, type=int, metavar='N',
                        help='number of data loading workers')
    parser.add_argument('--batch-size', default=6, type=int, metavar='N',
                        help='batch size')
    parser.add_argument('--flip', dest='flip', action='store_true',
                        help='flip the input during validation')

    main(parser.parse_args())
