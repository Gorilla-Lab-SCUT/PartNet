from models.vgg import vgg
from models.partnet_vgg import partnet_vgg
def Model_Construct(args, process_name):
    if args.arch.find('vgg') == 0:  ## the required model is vgg structure
        if process_name == 'image_classifier' or process_name == 'part_classifiers':
            model = vgg(args)
            return model
        elif process_name == 'partnet' or process_name == 'download_proposals':
            model = partnet_vgg(args)
            return model
        else:
            raise ValueError('the required process name is not exist')
    elif args.arch.find('resnet') == 0:
        raise ValueError('the resnet structure is not well finished')
    else:
        raise ValueError('the request model is not exist')
