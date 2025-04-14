import timm
import torchvision
import argparse
from torch.utils.data import DataLoader
from misc import collate_fn, Augment_v2
from lightning.pytorch import Trainer

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("model_name", default="vit_base_patch16_224")


def main(args):
    # Define dataset
    dataset = torchvision.datasets.CIFAR10(root="../../data", download=False)
    img_size = dataset[0][0].size[0]
    
    augment = Augment_v2(image_size=img_size)
    
    train_loader = DataLoader(dataset=dataset,
                    batch_size=32,
                    shuffle=True,
                    collate_fn=lambda batch: collate_fn(batch, augment),
                    pin_memory=True)

    x_q, x_k = next(iter(train_loader))
    print(x_q.shape, x_k.shape)
    
    trainer = Trainer(
        max_epochs=args.epochs,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=args.gpu_id,
        strategy="ddp" if args.distributed else "auto",
        accumulate_grad_batches=args.accum_grad_steps,
        log_every_n_steps=1,
        logger=logger,
        callbacks=[checkpoint_callback, early_stop_callback]
    )

    

if __name__ == "__main__":
    args = parse_arguments()
    main(args)