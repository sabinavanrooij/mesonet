import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import pickle
from meso import MesoNet
from mix_datasets import get_mixed_batches
import numpy as np
from random import shuffle
from PIL import ImageFile, Image
from itertools import islice
import argparse

def save_model(model, optim, epoch, folder, i):
    """
    Saves the model so that we can continue training.
    """

    filename = folder + "/model_" + str(epoch) + "_" + str(i) + ".pt"

    checkpoint = {
        "epoch": epoch,
        "state_dict": model.state_dict(),
        "optimizer": optim.state_dict()
    }
    torch.save(checkpoint, filename)

def accuracy_bce(predictions, targets):

    pred_args = predictions.round()
    # print(pred_args)
    # print(targets)

    correct = pred_args == targets
    correct = correct.sum()
    accuracy = correct.item() / len(targets)

    return accuracy

def main(args):
    net = MesoNet(args.batch_size)
    optimizer = optim.Adam(net.parameters(), lr=0.001)

    if not args.mix:
        if args.test:
            if args.data_type == 'celeba':
                real_imgs = torchvision.datasets.ImageFolder(root=args.folder,
                transform=torchvision.transforms.Compose([torchvision.transforms.CenterCrop(178), torchvision.transforms.Resize(args.real_resize), torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize((0.5, 0.5, 0.5),(0.5, 0.5, 0.5))]))

            elif args.data_type == 'deepfakes':
                real_imgs = torchvision.datasets.ImageFolder(root=args.folder,
                transform=torchvision.transforms.Compose([torchvision.transforms.Resize(args.real_resize), torchvision.transforms.CenterCrop(args.real_resize),
                torchvision.transforms.ToTensor(), torchvision.transforms.Normalize((0.5, 0.5, 0.5),(0.5, 0.5, 0.5))]))

            else:
                real_imgs = torchvision.datasets.ImageFolder(root=args.folder,
                transform=torchvision.transforms.Compose([torchvision.transforms.Resize(args.real_resize), torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize((0.5, 0.5, 0.5),(0.5, 0.5, 0.5))]))


    criterion_bce = nn.BCELoss(size_average=True)
    
    if args.test:
        net.load_state_dict(torch.load(args.trained_model)['state_dict'])
        net = net.eval()

    if args.resume:
        net.load_state_dict(torch.load(args.trained_model)['state_dict'])
        optimizer.load_state_dict(torch.load(args.trained_model)['optimizer'])
        for state in optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.cuda()


    if torch.cuda.is_available():
        net = net.cuda()

    losses = []
    losses_per_epoch = []
    accuracy_list = []
    avg_losses_per_epoch = []


    for epoch in range(args.max_epochs):

        if args.resume:
            print('Epoch: ', epoch + 10)
        else:
            print('Epoch: ', epoch)

        if args.mix:
            if args.test:
                batches_list = get_mixed_batches(args.batch_size, False)
            else:
                batches_list = get_mixed_batches(args.batch_size)
        
        elif args.test:
            real_dataloader = torch.utils.data.DataLoader(real_imgs,
                batch_size=int(args.batch_size), shuffle=True, drop_last=True)

            batches_list = zip(real_dataloader)
        

        epoch_acc = []
        epoch_loss = []

        for i, current_batch in enumerate(batches_list):
            if args.mix:
                batch = torch.cat((current_batch[0][0],current_batch[1][0],current_batch[2][0],current_batch[3][0],current_batch[4][0], current_batch[5][0], current_batch[6][0], current_batch[7][0],current_batch[8][0],current_batch[9][0]),dim=0)
            elif args.test:
                batch = current_batch[0][0]
            else:

                # print('i: ', i )
                real_batch = current_batch[0][0]
                fake_batch = current_batch[1][0]

                batch = torch.cat((real_batch,fake_batch),dim=0)


            if args.test:
                if args.mix:
                    real_labels = torch.zeros(int(args.batch_size/2))
                    fake_labels = torch.ones(int(args.batch_size/2))
                    labels = torch.cat((real_labels,fake_labels),dim=0)
                elif args.test_type == "fake":
                    labels = torch.ones(int(args.batch_size))
                else:
                    labels = torch.zeros(int(args.batch_size))
            else:
                real_labels = torch.zeros(int(args.batch_size/2))
                fake_labels = torch.ones(int(args.batch_size/2))
                labels = torch.cat((real_labels,fake_labels),dim=0)

            # shuffle batch
            indices = list(range(args.batch_size))
            shuffle(indices)
            batch = batch[indices]
            labels = labels[indices]

            if torch.cuda.is_available():
                batch = batch.cuda()
                labels = labels.cuda()

            optimizer.zero_grad()

            if args.test:
                with torch.no_grad():

                    # run model
                    pred = net(batch)

                    acc = accuracy_bce(pred,labels)
                    print(acc)
                    epoch_acc.append(acc)


            else:

                pred= net(batch)

                # compute CE Loss
                pred = pred.squeeze()

                loss_bce = criterion_bce(pred,labels)

                total_loss = loss_bce

                print('loss: ', total_loss.item())
                losses.append(total_loss.item())
                epoch_loss.append(total_loss.item())

                total_loss.backward()
                optimizer.step()

                acc = accuracy_bce(pred,labels)
                print('accuracy: ',acc)

                epoch_acc.append(acc)

            if not args.test:
                if i % 100 == 99:
                    print(i)
                    print('loss:',total_loss.item())
                    print('acc:', acc)

                if i % args.eval_freq == args.eval_freq-1:
                    print(i)
                    print('loss:',total_loss.item())
                    print('acc:', acc)

                    # if args.resume:
                    #     save_model(net, optimizer, epoch + 10, args.models_folder, i)
                    #     with open('losses_epoch_{}_{}.pkl'.format(str(epoch + 10), str(i)), 'wb') as f:
                    #         pickle.dump(losses, f)
                    # else:
                    #     save_model(net, optimizer, epoch, args.models_folder, i)
                    #     with open('losses_epoch_{}_{}.pkl'.format(str(epoch), str(i)), 'wb') as f:
                    #         pickle.dump(losses, f)


        accuracy_list.append(np.mean(epoch_acc))
        print('average acc:',np.mean(epoch_acc))


        if args.test:
            break

        if not args.test:

            avg_losses_per_epoch.append(np.mean(epoch_loss))
            print('average loss per epoch:',np.mean(epoch_loss))



    # if not args.test:
    #     save_model(net, optimizer, epoch, args.models_folder, i)

    #     with open('losses_final.pkl', 'wb') as f:
    #         pickle.dump(losses, f)

    #     with open('avg_losses_per_epoch.pkl','wb') as f:
    #         pickle.dump(avg_losses_per_epoch,f)

    #     with open('accs_per_epoch.pkl','wb') as f:
    #         pickle.dump(accuracy_list,f)

    # else:
    #     with open('test_accs_per_epoch.pkl','wb') as f:
    #         pickle.dump(accuracy_list,f)




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder", default='pggan_fake/test')
    parser.add_argument("--real_folder", default="celebA")
    parser.add_argument("--fake_folder", default="stargan/train")
    parser.add_argument("--models_folder", default="models")
    parser.add_argument("--real_resize", type=int, default=256)
    parser.add_argument("--fake_resize", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--max_epochs", type=int, default=10)
    parser.add_argument("--eval_freq", type=int, default=100)
    parser.add_argument("--test", type=bool, default=False)
    parser.add_argument("--trained_model", default="")
    parser.add_argument("--data_type", default="starGAN")
    parser.add_argument("--test_type", default="fake")
    parser.add_argument("--noresidual", type=bool, default=False)
    parser.add_argument("--num_images",type=int,default=1)
    parser.add_argument("--model_1024", type=bool,default=False)
    parser.add_argument("--mix", type=bool, default=False)
    parser.add_argument("--resume", type=bool, default=False)
    parser.add_argument("--few_shot", type=bool, default=False)
    args = parser.parse_args()
    main(args)

