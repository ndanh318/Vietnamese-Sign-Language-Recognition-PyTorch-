import os
import torch
import argparse
import torch.nn as nn
import torch.optim as optim
from tqdm.autonotebook import tqdm
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from torchvision.models import resnet50, ResNet50_Weights
from torchvision import transforms
from sklearn.metrics import accuracy_score

import numpy as np
from src.model import MyCNNs
from src.dataset import VSLR_Dataset


def get_args():
    parser = argparse.ArgumentParser("Training Vietnamese Sign Language model")
    parser.add_argument("--data_path", "-d", type=str, default="dataset/alphabet")
    parser.add_argument("--image_size", "-i", type=int, default=224)
    parser.add_argument("--num_epochs", "-e", type=int, default=100)
    parser.add_argument("--batch_size", "-b", type=int, default=32)
    parser.add_argument("--ratio", "-r", type=float, default=0.8)
    parser.add_argument("--learning_rate", "-lr", type=float, default=0.001)
    parser.add_argument("--total_image_per_class", "-t", type=int, default=500)
    parser.add_argument("--tensorboard_dir", "-tb", type=str, default="tensorboard")
    parser.add_argument("--checkpoint_dir", "-c", type=str, default="trained_models")
    parser.add_argument("--checkpoint", "-p", type=str, default=None)

    args = parser.parse_args()
    return args


def train(args):
    output_file = open(args.checkpoint_dir + os.sep + "logs.txt", "w")
    output_file.write("Model's parameters: {}".format(vars(args)))

    # setting tensorboard
    if not os.path.isdir(args.tensorboard_dir):
        os.makedirs(args.tensorboard_dir)
    writer = SummaryWriter(args.tensorboard_dir)

    # setting device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((args.image_size, args.image_size), antialias=True)
    ])

    training_set = VSLR_Dataset(args.data_path,
                                transform=transform,
                                total_image_per_class=args.total_image_per_class,
                                ratio=args.ratio,
                                mode="train")
    print("There are {} images in training set".format(len(training_set)))
    train_dataloader = DataLoader(
        dataset=training_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0
    )
    test_set = VSLR_Dataset(args.data_path,
                            transform=transform,
                            total_image_per_class=args.total_image_per_class,
                            ratio=args.ratio,
                            mode="test")
    print("There are {} images in test set".format(len(test_set)))
    test_dataloader = DataLoader(
        dataset=test_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0
    )

    # model
    num_class = test_set.num_class
    # model = MyCNNs(num_classes=num_classes).to(device)
    model = resnet50(weights=ResNet50_Weights.DEFAULT)
    model.fc = nn.Linear(model.fc.in_features, num_class)
    model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    if args.checkpoint is not None and os.path.isfile(args.checkpoint):
        checkpoint = torch.load(args.checkpoint)
        start_epoch = checkpoint["epoch"]
        best_acc = checkpoint["best_acc"]
        model.load_state_dict(checkpoint["model_params"])
        optimizer.load_state_dict(checkpoint["optimizer"])
    else:
        best_acc = -1
        start_epoch = 0

    criterion = nn.CrossEntropyLoss()
    num_iters = len(train_dataloader)

    for epoch in range(start_epoch, args.num_epochs):
        model.train()
        all_losses = []
        process_bar = tqdm(train_dataloader, colour="cyan")
        for iter, (images, labels) in enumerate(process_bar):
            images = images.to(device)
            labels = labels.to(device)

            # optimizer reset
            optimizer.zero_grad()

            # forward pass
            output = model(images)

            # calculate loss
            loss = criterion(output, labels)
            all_losses.append(loss.item())
            loss_value = np.mean(all_losses)
            process_bar.set_description("Epoch: {}/{}. Loss: {:0.4f}".format(epoch + 1, args.num_epochs, loss_value))
            writer.add_scalar("Train/Loss", loss_value, epoch * num_iters + iter)

            # backpropagation
            loss.backward()
            optimizer.step()

        model.eval()
        all_labels = []
        all_predictions = []
        all_losses = []
        with torch.no_grad():
            for iter, (images, labels) in enumerate(test_dataloader):
                images = images.to(device)
                labels = labels.to(device)

                # forward pass
                output = model(images)

                # calculate loss
                loss = criterion(output, labels)
                all_losses.append(loss.item())
                predictions = torch.argmax(output, dim=1)
                all_labels.extend(labels.tolist())
                all_predictions.extend(predictions.tolist())

        acc_score = accuracy_score(all_labels, all_predictions)
        loss_value = np.mean(all_losses)
        output_file.write("Epoch: {}/{} \nTest loss: {} \nTest accuracy: {}".format(
            epoch + 1,
            args.num_epochs,
            loss_value,
            acc_score))
        print("Epoch: {}/{}. Accuracy: {}. Loss: {}".format(
            epoch + 1,
            args.num_epochs,
            acc_score,
            loss_value))
        writer.add_scalar("Val/Loss", loss_value, epoch)
        writer.add_scalar("Val/Accuracy", acc_score, epoch)
        checkpoint = {
            "epoch": epoch + 1,
            "best_acc": best_acc,
            "model_params": model.state_dict(),
            "optimizer": optimizer.state_dict(),
        }
        torch.save(checkpoint, os.path.join(args.checkpoint_dir, "last.pt"))
        if best_acc < acc_score:
            torch.save(checkpoint, os.path.join(args.checkpoint_dir, "last.pt"))
            best_acc = acc_score
    writer.close()
    output_file.close()


if __name__ == '__main__':
    args = get_args()
    train(args)