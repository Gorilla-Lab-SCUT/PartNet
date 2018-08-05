from data.cub200 import cub200
from data.stanford_dogs import stanford_dogs
from data.flowers import flowers
from data.cars import cars
from data.aircrafts import aircrafts

def generate_dataloader(args, process_name, part_index=-1):
    print('the required dataset is', args.dataset)
    if args.dataset == 'cub200':
        train_loader, val_loader = cub200(args, process_name, part_index)
    elif args.dataset == 'stanford_dogs':
        raise ValueError('the required dataset is not prepared')
        # train_loader, val_loader = stanford_dogs(args, process_name, part_index)
    elif args.dataset == 'flowers':
        train_loader, val_loader = flowers(args, process_name, part_index)
    elif args.dataset == 'cars':
        raise ValueError('the required dataset is not prepared')
        # train_loader, val_loader = cars(args, process_name, part_index)
    elif args.dataset == 'aircrafts':
        raise ValueError('the required dataset is not prepared')
        # train_loader, val_loader = aircrafts(args, process_name, part_index)
    else:
        raise ValueError('the required dataset is not prepared')

    return train_loader, val_loader

