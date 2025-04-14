import torch
import lightning.pytorch as pl
import timm
import torchvision
import argparse


def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument("model", )

def main(args):
    print("Shark")
    model = timm.create_model()
    
    # Define dataset
    dataset = torchvision.datasets.CIFAR10(download=True)

if __name__ == "__main__":
    args = parse_arguments()
    main(args)