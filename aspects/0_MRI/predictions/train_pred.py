
import logging
import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from torch.optim import Adam, lr_scheduler
from torch.utils.data import DataLoader, RandomSampler
from sle.models import MROnlyModel
from datasets import MRIs
from evaluation import train_loop
from utils import create_dependencies,parse_arguments, update_report, update_curves
plt.switch_backend('agg')


def main():

    # Initialisation
    config = parse_arguments()
    variable, task, batch_size = config["variable"], config['task'], config['batch_size']
    model_dir = os.path.join(config['output_dir'], config["model_name"])
    nw = config['num_workers']  
    output_dir = create_dependencies(model_dir, variable)
    torch.multiprocessing.set_sharing_strategy('file_system')
    # device = torch.device("cuda:3" if torch.cuda.is_available() else 'cpu')
    device = torch.device("cpu")
    best_val_loss = np.inf
    best_epoch = 0

    # Create training and validation datasets
    train_csv_path = os.path.join(model_dir, "autoencoding/features/concat_train.csv.gz")
    val_csv_path =  os.path.join(model_dir, "autoencoding/features/concat_val.csv.gz")
    test_csv_path =  os.path.join(model_dir, "autoencoding/features/concat_test.csv.gz")

    train_dataset = MRIs(train_csv_path, variable, task, "train")
    val_dataset = MRIs(val_csv_path,variable, task, "val")
    test_dataset = MRIs(test_csv_path, variable, task, "test")

    # train_sampler = RandomSampler(train_dataset)
    # val_sampler = RandomSampler(val_dataset)
    # test_sampler = RandomSampler(test_dataset)

    train_dataloader = DataLoader(train_dataset, batch_size, num_workers=nw)
    val_dataloader = DataLoader(val_dataset, batch_size, num_workers=nw)
    test_dataloader = DataLoader(test_dataset, batch_size, num_workers=nw)

    # Get features caracteristics
    num_features, num_classes = train_dataset._get_feat_characteristics()

    # logging.info LOG
    logging.basicConfig(filename=os.path.join(output_dir, f"predict_{variable}.log"),
                        filemode='w', format='%(message)s', level=logging.INFO, force=True)

    logging.info(f"Dataset = {config['dataset']}")
    logging.info(f"Batches = {config['batch_size']}")
    logging.info(f"Task = {task}")
    logging.info(f"Variable = {config['variable']}")
    logging.info(f"Number of features = {num_features}")
    logging.info(f"Hidden Layer Size = {config['hidden_layer_size']}")
    for x in [train_dataset, val_dataset, test_dataset]:
        logging.info(f"Number of cases in {x.name} dataset = {x.num_cases}")
    logging.info(f"Number of classes = {num_classes}")
    logging.info(f"\nLabels counts in train dataset, \n{train_dataset.label_counts}")
    logging.info(f"\nLabels counts in val dataset, \n{val_dataset.label_counts}")
    logging.info(f"\nLabels counts in test dataset, \n{test_dataset.label_counts}")

    # Send the model to GPU
    model = MROnlyModel(task, num_classes, num_features, config['hidden_layer_size'])
    model = model.to(device)

    if os.path.exists(output_dir + "/model.pt"):
        logging.info("\nLoading the last version of the model...")
        model.load_state_dict(torch.load(output_dir + "/model.pt"))

    # Define optimizer
    optimizer = Adam(model.parameters(), lr=config['lr'])
    # scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, 'min', verbose=True)

    # Train and evaluate
    num_epochs = config["num_epochs"]

    for epoch in range(num_epochs):

        print(f'Training model {config["model_name"]}')
        print(f'Parameter : {variable} ({task} task)')
        print(f'Epoch {epoch+1}/{num_epochs}')

        train_epoch_metrics = train_loop(
            model, train_dataloader, task, variable, num_classes, optimizer, device, train=True)
        val_epoch_metrics = train_loop(
            model, val_dataloader, task, variable, num_classes, optimizer, device, train=False)
        
        # scheduler.step(val_epoch_metrics['loss'])
        optimizer.step()

        metrics = [m for m in train_epoch_metrics.keys()]
        report = update_report(
            output_dir, num_features, num_classes, config, epoch, train_epoch_metrics, val_epoch_metrics)
        update_curves(report, metrics, output_dir)

        logging.info(f'\nEpoch {epoch+1}/{num_epochs}')
        logging.info(f"Model train loss = {train_epoch_metrics['loss']:6f}")
        logging.info(f"Model val loss = {val_epoch_metrics['loss']:6f}")
        torch.save(model.state_dict(), output_dir+"/model.pt")

        if val_epoch_metrics['loss'] < best_val_loss:
            best_epoch = epoch
            torch.save(model.state_dict(), output_dir+f"/best_model.pt")
            best_val_loss = val_epoch_metrics['loss']


        os.system("clear")

    # Fin de la boucle d'entrainement
    logging.info(f"\nEVALUATION ON TEST SET (best epoch = {best_epoch}")
    model.load_state_dict(torch.load(output_dir+f"/best_model.pt"))
    test_metrics = train_loop(
        model, test_dataloader, task, variable, num_classes, optimizer, device, train=False)

    for k in test_metrics.keys():
        logging.info(f"Final model test {k} = {test_metrics[k]:6f}")



# MAIN
##########

if __name__ == '__main__':
    main()

