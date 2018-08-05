##############################################################################
# Pytorch PartNet
# Licensed under The MIT License [see LICENSE for details]
# Written by Yabin Zhang
##############################################################################

from opts import opts  # The options for the project
from process import Process1_Image_Classifier
from process import Process2_PartNet
from process import Process3_Download_Proposals
from process import Process4_Part_Classifiers
from process import Process5_Final_Result
import os
def main():
    global args
    args = opts()
    if args.test_only:
        os.environ['CUDA_VISIBLE_DEVICES'] = "0"
        print('download the detected part images in the disk')
        Process3_Download_Proposals(args) ### We use hook in this process, so only one gpu should be used.
        os.environ['CUDA_VISIBLE_DEVICES'] = "0,1,2,3,4,5,6,7"
        print('only test the final results, please make sure all the needed files is prepared.')
        Process5_Final_Result(args)
    else:
        # fine-tune on the Fine-Grained dataset an ImageNet pre-trained model
        os.environ['CUDA_VISIBLE_DEVICES'] = "0,1,2,3,4,5,6,7"
        Process1_Image_Classifier(args)
        # Train the PartNet based on the above fine-tuned model on the Fine-Grained dataset
        Process2_PartNet(args)
        # Download the proposals detected by the PartNet
        os.environ['CUDA_VISIBLE_DEVICES'] = "0"
        Process3_Download_Proposals(args) ### We use hook in this process, so only one gpu should be used.
        os.environ['CUDA_VISIBLE_DEVICES'] = "0,1,2,3,4,5,6,7"
        # Train individual Classifier for each part
        Process4_Part_Classifiers(args)
        # Averaging the probabilities of the above models (image level + part 1-2-3 + ParNet) for final prediction
        Process5_Final_Result(args)

if __name__ == '__main__':
    main()





