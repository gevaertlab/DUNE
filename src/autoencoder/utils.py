import json
import argparse
import os
from os.path import join
import pandas as pd
import matplotlib.pyplot as plt
import configparser


def create_dependencies(output_dir, model_name):

    output_dir = os.path.join(output_dir, model_name)
    os.makedirs(os.path.join(output_dir, "autoencoding/exported_data"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "autoencoding/logs"), exist_ok=True)

    return output_dir


def parse_arguments():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--model_path', type=str,
                        help='model_path')
    args = parser.parse_args()

    config_file = join(args.model_path, "config/ae.cfg")

    config = configparser.ConfigParser()
    config.read(config_file)
    config = dict(config["config"])

    config['batch_size']  = eval(config['batch_size'])
    config['num_blocks']  = eval(config['num_blocks'])
    config['features']  = eval(config['features'])
    config['num_epochs'] = eval(config['num_epochs'])
    config['learning_rate']  = eval(config['learning_rate'])
    config['num_workers']  = eval(config['num_workers'])
    config['modalities']  = eval(config['modalities'])
    config['min_dims']  = eval(config['min_dims'])
    config['unet']  = eval(config['unet'])
    config['quick']  = eval(config['quick'])

    return config


def update_curves(report, criterion_name, output_dir):

    print("Updating metric curves...")
    metrics = ["loss", "ssim", "psnr", "mse"]

    fig, ax = plt.subplots(nrows=len(metrics), sharex=True, figsize=(6, 16))
    for i, metric in enumerate(metrics):
        ax[i].plot(report.index, report["train_" + metric], label='Train')
        ax[i].plot(report.index, report["test_" + metric], label='Test')

        if metric == "loss":
            title =  f"{metric.upper()} = {criterion_name}"
            col = "red"
        else:
            title = metric.upper()
            col = "black"

        ax[i].set_title(title, color=col)
        ax[i].set_xlabel('Epochs')
        ax[i].set_ylabel(metric)
        ax[0].legend()
    fig.savefig(f"{output_dir}/autoencoding/logs/curves.png")
    plt.close(fig)



def update_report(
    output_dir, model_name, quick, num_cases, modalities, features, batch_size, criterion_name, learning_rate,
    num_epochs, epoch, train_epoch_metrics, test_epoch_metrics
):

    report_path = f"{output_dir}/autoencoding/logs/report.csv"

    if os.path.exists(report_path):
        report = pd.read_csv(report_path)
    else:
        report = pd.DataFrame()    
    
    epoch_report = {}
    epoch_report['model'] = [model_name]
    epoch_report['quick_exec'] = quick
    epoch_report['num_cases'] = int(num_cases)
    epoch_report['num_mod'] = len(modalities)
    epoch_report['init_features'] = int(features)
    epoch_report['batch_size'] = batch_size
    epoch_report['criterion'] = criterion_name
    epoch_report['learning rate'] = learning_rate
    epoch_report['total_epoch'] = num_epochs
    epoch_report['epoch'] = int(epoch)+1
    epoch_report['train_loss'] = round(train_epoch_metrics['loss'], 6)
    epoch_report['train_ssim'] = round(train_epoch_metrics['ssim'], 6)
    epoch_report['train_psnr'] = round(train_epoch_metrics['psnr'], 6)
    epoch_report['train_mse'] = round(train_epoch_metrics['mse'], 6)
    epoch_report['test_loss'] = round(test_epoch_metrics['loss'], 6)
    epoch_report['test_ssim'] = round(test_epoch_metrics['ssim'], 6)
    epoch_report['test_psnr'] = round(test_epoch_metrics['psnr'], 6)
    epoch_report['test_mse'] = round(test_epoch_metrics['mse'], 6)

    epoch_report = pd.DataFrame.from_dict(epoch_report)
    report = pd.concat([report, epoch_report], ignore_index=True)
    report.to_csv(report_path, index=False)

    return report
