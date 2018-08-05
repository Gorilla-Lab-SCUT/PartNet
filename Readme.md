# Part-Aware Fine-grained Object Categorization using Weakly Supervised Part Detection Network

The **pytorch** implementation of the paper: Part-Aware Fine-grained Object Categorization using Weakly Supervised Part Detection Network.

The paper is available at: https://arxiv.org/abs/1806.06198

A **torch** version of the PartNet (Only the PartNet module) is going to be available.

To train the PartNet based on the ImageNet pre-trained model (e.g., VGGNet), you need to download the ImageNet pre-trained model firstly and place it in the "vgg19-bn-modified" folder.
We provide a ImageNet pre-trained model, which obtains 74.262% top1 acc (almost the same as 74.266% provided by: https://github.com/Cadene/pretrained-models.pytorch#torchvision).

The commands to train the model from scratch and to verify the final results can be found in the **run.sh**.

We provide all the intermediate models for fast implementation and verification, which can be downloaded from:
* [Baidu Cloud](https://pan.baidu.com/s/1h5oTI4POrSWBo_XEDkFZnw)
* [Google Cloud](https://drive.google.com/drive/folders/1HNXGE2fI5BHSHCROw8aXyKc4R2aZJzDx?usp=sharing)

Note that the 'DPP' and 'ROI-Pooling' modules are need be compiled. Details can be find the Readme.md in each folder.

If you have any question about the paper and the code, feel free to sent email to me: zhang.yabin@mail.scut.edu.cn

