import argparse
import os
from os.path import join
import pandas as pd
import matplotlib.pyplot as plt
import configparser
import torch
from torchvision.utils import save_image, make_grid
import numpy as np

def create_dependencies(model_path):

    os.makedirs(os.path.join(model_path, "autoencoding/exported_data"), exist_ok=True)
    os.makedirs(os.path.join(model_path, "autoencoding/logs"), exist_ok=True)



def parse_arguments():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('-m', '--model_path', type=str,
                        help='model_path')
    args = parser.parse_args()

    config_file = join(args.model_path, "config/ae.cfg")

    conf_parser = configparser.ConfigParser()
    conf_parser.read(config_file)
    
    config = dict(conf_parser["config"])
    model = dict(conf_parser["model"])
    data = dict(conf_parser["data"])

    model["model_path"] = args.model_path
    model['num_blocks']  = eval(model.get('num_blocks'))
    model['features']  = eval(model.get('features'))
    model['latent_dim']  = eval(model.get('latent_dim'))
    config['num_epochs'] = eval(config.get('num_epochs'))
    config['learning_rate']  = eval(config.get('learning_rate'))
    config['num_workers']  = eval(config.get('num_workers'))
    config['quick']  = eval(config.get('quick'))
    data['batch_size']  = eval(data.get('batch_size'))
    data['modalities']  = eval(data.get('modalities'))
    data['min_dims']  = eval(data.get('min_dims'))
    data['whole_brain']  = eval(data.get('whole_brain'))


    return {**model, **data, **config}


def update_curves(report,  output_dir):

    print("Updating metric curves...")
    metrics = ["loss", "ssim"]

    fig, ax = plt.subplots(nrows=len(metrics), sharex=True, figsize=(6, 8))
    for i, metric in enumerate(metrics):
        ax[i].plot(report.index, report["train_" + metric], label='Train')
        ax[i].plot(report.index, report["test_" + metric], label='Test')


        ax[i].set_title(metric.upper())
        ax[i].set_xlabel('Epochs')
        ax[i].set_ylabel(metric)
        ax[0].legend()
    fig.savefig(f"{output_dir}/autoencoding/logs/curves.png")
    plt.close(fig)



def update_report(
    config, train_cases, val_cases, optimizer,
    epoch, train_epoch_metrics, test_epoch_metrics, beta
):

    output_dir = config['model_path'] 

    report_path = f"{output_dir}/autoencoding/logs/report.csv"
    report = pd.read_csv(report_path) if os.path.exists(report_path) else pd.DataFrame()    
    
    epoch_report = {}
    epoch_report['model'] = [config['model_name']]
    epoch_report['train cases'] = int(train_cases)
    epoch_report['val cases'] = int(val_cases)
    epoch_report['num_mod'] = len(config['modalities'])
    epoch_report['init_features'] = int(config['features'])
    epoch_report['batch_size'] = config['batch_size']
    epoch_report['learning rate'] = optimizer.param_groups[0]['lr']
    epoch_report['total_epoch'] = config["num_epochs"]
    epoch_report['epoch'] = int(epoch)+1
    epoch_report['train_loss'] = round(train_epoch_metrics['loss'], 6)
    epoch_report['train_ssim'] = round(train_epoch_metrics['ssim'], 6)
    epoch_report['test_loss'] = round(test_epoch_metrics['loss'], 6)
    epoch_report['test_ssim'] = round(test_epoch_metrics['ssim'], 6)
    epoch_report['beta'] = round(beta, 3)

    epoch_report = pd.DataFrame.from_dict(epoch_report)
    report = pd.concat([report, epoch_report], ignore_index=True)
    report.to_csv(report_path, index=False)

    return report



def reconstruct_image(net, device, output_dir, testloader, **kwargs):

    imgs, _ = next(iter(testloader))

    batch_size, nmod, depth, height, width = imgs.size()
    orig = torch.reshape(
        imgs[:, :, depth // 2, :, :],
        [batch_size * nmod, 1, height, width]
        )

    net.eval()
    imgs = imgs.to(device)
    reconstructed = net(imgs)[0]
    reconstructed = reconstructed.cpu().data

    reconstructed = torch.reshape(
        reconstructed[:, :, depth // 2, :, :],
        [batch_size*nmod, 1, height, width]
        )
    
    concat = torch.cat([orig, reconstructed])

    save_image(
        make_grid(concat, nrow=nmod*batch_size),
        f'{output_dir}/autoencoding/logs/reconstructions.png'
    )

    return


def frange_cycle_sigmoid(start, stop, n_epoch, n_cycle=4, ratio=0.5):
    L = np.ones(n_epoch)
    period = n_epoch/n_cycle
    step = (stop-start)/(period*ratio) # step is in [0,1]
    
    for c in range(n_cycle):

        v , i = start , 0
        while v <= stop:
            L[int(i+c*period)] = 1.0/(1.0+ np.exp(- (v*12.-6.)))
            v += step
            i += 1
    return L    
