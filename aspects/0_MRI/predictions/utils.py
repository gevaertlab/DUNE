import json
import argparse
import os
import pandas as pd
import matplotlib.pyplot as plt

def create_dependencies(output_dir, variable):

    output_dir = os.path.join(output_dir, variable)
    os.makedirs(output_dir, exist_ok=True)

    return output_dir



def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str,
                    default='config.json', help='configuration json file')
    args = parser.parse_args()

    with open(args.config) as f:
        config = json.load(f)

    return config


def update_report(output_dir, config, epoch, train_epoch_metrics, val_epoch_metrics):

    print("Updating report...")
    report_path = f"{output_dir}/report.csv"

    if os.path.exists(report_path):
        report = pd.read_csv(report_path)
    else:
        report = pd.DataFrame()    
    
    epoch_report = {}
    epoch_report['variable'] = config["variable"]
    epoch_report['hidden_layer_size'] = int(config["hidden_layer_size"])
    epoch_report['batch_size'] = config["batch_size"]
    epoch_report['epoch'] = [int(epoch)+1]
    epoch_report['num_epochs'] = config["num_epochs"]

    for m in train_epoch_metrics.keys():
        epoch_report["train_"+ m] = round(train_epoch_metrics[m], 6)
        epoch_report["val_"+ m] = round(val_epoch_metrics[m], 6)

    epoch_report = pd.DataFrame.from_dict(epoch_report)
    report = pd.concat([report, epoch_report], ignore_index=True)
    report.to_csv(report_path, index=False)

    return report


def update_curves(report, metrics, output_dir):

    print("Updating metric curves...")
    
    fig, ax = plt.subplots(nrows=len(metrics), sharex=True, figsize=(7, 10))
    for i, metric in enumerate(metrics):
        ax[i].plot(report.index, report["train_" + metric], label='Train')
        ax[i].plot(report.index, report["val_" + metric], label='Val')
        ax[i].set_xlabel('Epochs')
        ax[i].set_ylabel(metric)
        ax[i].set_title(metric.upper())
        ax[0].legend()
    fig.savefig(f"{output_dir}/curves.png")
    plt.close(fig)
