
import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from torch.nn import CrossEntropyLoss, BCELoss, MSELoss
from torch.optim import Adam
from torch.utils.data import DataLoader, RandomSampler
from models import MROnlyModel
from datasets import MRIs
from evaluation import evaluate, cox_loss
from utils import create_dependencies,parse_arguments, update_report, update_curves
plt.switch_backend('agg')



def train_model(model, dataloaders, optimizer, device, config, num_classes, num_features, output_dir):

    num_epochs = config["num_epochs"]
    task = config["task"]
    train_metrics, val_metrics = {}, {}

    for epoch in range(num_epochs):
        model.train()

        print(f'\nEpoch {epoch+1}/{num_epochs}')
        for _, (features, labels, vital_status) in enumerate(dataloaders['train']):
            inputs = features.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)

            if task == "survival":
                loss_func = cox_loss
                vital_status = vital_status.to(device)
                labels = (labels, vital_status)
                outputs = outputs.view(-1)

            elif task == "regression" or num_classes == 2 :
                labels = labels.float()                
                outputs = outputs.view(-1)
                loss_func = MSELoss() if task == "regression" else BCELoss()

            else: # task == "classification" and num_classes > 2
                loss_func = CrossEntropyLoss()
                    
            loss = loss_func(outputs, labels)
            loss.backward()
            optimizer.step()                

        train_epoch_metrics = evaluate(
            model, dataloaders['train'], task, num_classes, device)
        val_epoch_metrics = evaluate(
            model, dataloaders['val'], task, num_classes, device)

        metrics = [m for m in train_epoch_metrics.keys()]
        for m in metrics:
            if epoch == 0:
                train_metrics[m] = [train_epoch_metrics[m]]
                val_metrics[m] = [val_epoch_metrics[m]]
            else:
                train_metrics[m].append(train_epoch_metrics[m])
                val_metrics[m].append(val_epoch_metrics[m])

        report = update_report(
            output_dir, num_features, num_classes, config, epoch, train_epoch_metrics, val_epoch_metrics)
        update_curves(report, metrics, output_dir)

        print(f"Model train loss = {train_epoch_metrics['loss']:6f}")
        print(f"Model val loss = {val_epoch_metrics['loss']:6f}")
        torch.save(model.state_dict(), output_dir+"/model.pt")


    # Fin de la boucle d'entrainement
    print("\nEVALUATION ON TEST SET")
    test_metrics = evaluate(
        model, dataloaders['test'], task, num_classes, device)
    
    for k in test_metrics.keys():
        print(f"Final model test {k} = {test_metrics[k]:6f}")
    

def main():
    np.random.seed(333)
    torch.random.manual_seed(333)
    config = parse_arguments()
    variable = config["variable"]
    output_dir = create_dependencies(config["output_path"], variable)

    torch.multiprocessing.set_sharing_strategy('file_system')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    # Create training and validation datasets
    train_csv_path = config["csv_paths"]+"/concat_train.csv"
    val_csv_path = config["csv_paths"]+"/concat_val.csv"
    test_csv_path = config["csv_paths"]+"/concat_test.csv"

    image_datasets = {}
    image_datasets['train'] = MRIs(train_csv_path, variable, config["task"])
    image_datasets['val'] = MRIs(val_csv_path,variable, config["task"])
    image_datasets['test'] = MRIs(test_csv_path, variable, config["task"])

    image_samplers = {}
    image_samplers['train'] = RandomSampler(image_datasets['train'])
    image_samplers['val'] = RandomSampler(image_datasets['val'])
    image_samplers['test'] = RandomSampler(image_datasets['test'])

    # Create training and validation dataloaders
    dataloaders_dict = {
        x: DataLoader(image_datasets[x], batch_size=config['batch_size'], sampler=image_samplers[x], num_workers=config["num_workers"]) for x in ['train', 'val', 'test']}


    # Get features caracteristics
    num_features, num_classes = image_datasets['train']._get_feat_characteristics()

    # Send the model to GPU
    model = MROnlyModel(config["task"], num_classes, num_features, config['hidden_layer_size'])
    model = model.to(device)

    if os.path.exists(output_dir + "/model.pt"):
        print("Loading backup version of the model...")
        model.load_state_dict(torch.load(output_dir + "/model.pt"))

    # Define optimizer
    optimizer_ft = Adam(model.parameters(), lr=config['lr'])

    # PRINT LOG
    print(f"Dataset = {config['dataset']}")
    print(f"Batches = {config['batch_size']}")
    print(f"Task = {config['task']}")
    print(f"Variable = {config['variable']}")
    print(f"Number of classes = {num_classes}")
    print(f"Number of features = {num_features}")
    print(f"Hidden Layer Size = {config['hidden_layer_size']}")
    for x in ['train', 'val', 'test']:
        print(f"Number of cases in {x} dataset = {image_datasets[x].num_cases}")


    # Train and evaluate

    train_model(
        model, dataloaders_dict, optimizer_ft, device, config, num_classes, num_features, output_dir)


# MAIN
##########

if __name__ == '__main__':
    main()

