# Attention Guided CAM: Visual Explanations of Vision Transformer Guided by Self-Attention

This repo contains code for a mechanistic study of how ViTs localize and represent visual evidence. The project implements Attention-Guided CAM (AGCAM) and compares it against other explanation methods for ViTs, including Attention Rollout and LRP-based approaches. Rather than treating visual explanations as purely post-hoc artifacts, this work examines how self-attention structure interacts with gradient-based signals to produce localized evidence maps, and how these mechanisms behave across different inputs.

This code was originally developed as part of a research study and is presented here as a reproducible framework for probing internal representations of ViTs.

## Research Motivation

ViTs often achieve strong aggregate performance while relying on distributed, non-local representations that are difficult to interpret. In weak-signal or weakly supervised regimes, this can lead to explanations that appear confident but are poorly grounded in relevant visual evidence.

This project investigates:
* how self-attention influences gradient-based explanations,
* whether attention-guided methods improve localization consistency,
* and where explanation methods break down despite strong classification accuracy.

## Method Overview
AGCAM performs a gradient-based analysis of ViT models that is explicitly guided by the model’s self-attention structure. By incorporating attention information directly into the explanation pipeline, the method aims to align saliency maps with the internal decision pathways of the model rather than treating attention and gradients independently.

We evaluate AGCAM alongside:
* Attention Rollout (Abnar & Zuidema, 2020), which aggregates attention weights across layers, and
* LRP-based ViT explanations (Chefer et al., 2021), which propagate relevance scores through the transformer architecture.

## Experimental Setup
Experiments are conducted on the ImageNet ILSVRC 2012 validation set using pretrained Vision Transformer models released via the timm library, which can be downloaded from https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_base_p16_224-80ecf9dd.pth. While the models themselves are not trained as part of this work, the focus is on internal behavior and explanation quality, not task optimization. For the quantitative evalution (ABPC and localiztion performance test), you need the validation set of ImageNet ILSVRC 2012 and should provide the root folder of the dataset as an argument.

We evaluate explanation methods using:
* pixel accuracy, IoU, Dice coefficient, precision, and recall (for localization)
*  Area Between Perturbation Curves (ABPC) to measure explanation faithfulness under systematic input perturbation

Across methods, the evaluation framework surfaces several recurring failure patterns:
* explanations that collapse to high-frequency textures,
* attention maps that diffuse across irrelevant regions,
* and saliency maps that remain stable under perturbation despite changes in model confidence.

## Example Code Configurations
Due to the restriction on the size of the submission file, a small subset of the dataset is selected as the sample images for the visualization example. 

This submitted zip file contains the following contents:
* The implementation code of our method.
* The ViT model implementation provided by Timm library adapted to our method. 
* The implementation code of Attention Rollout whose method was introduced in https://arxiv.org/abs/2005.00928, provided by https://github.com/jacobgil/vit-explain
* The implementation code of LRP-based method devised for Vision Transformer whose method was introduced in https://arxiv.org/abs/2012.09838, provided by https://github.com/hila-chefer/Transformer-Explainability.
* The sample code for calculating the localization performance with the metrics shown in our paper (pixel accuracy, IoU, dice coefficient, recall and precision scores).
* The sample code for calculating the ABPC (Area Between Perturbation Curves) score. 
* The sample notebook for generating visual explanation and the class-specific visual explanation.
* The sample images.
* A requirements.txt file

# To set the environment,
```
conda create -n env_name python=3.9
conda activate env_name
cd ./AGCAM
pip install -r requirements.txt
```
Type the commands as written above to set the environment to execute the implementation codes.


### To measuer the localizaiton performance, 
```
python localization.py --method=agcam --data_root=./ILSVRC --threshold=0.5
```

This will print the average of the localization performance of all images calculated from the validation set of ImageNet ILSVRC 2012.
Note that you need to provide the root of the dataset in `--data_root`. 

You can choose the method to evaluate by using the `--method` argument. 
Type as follows to evaluate each method:
- 'agcam': AGCAM
- 'rollout': Attention Rollout, proposed in https://arxiv.org/abs/2005.00928
- 'lrp': The LRP-based method devised for ViT, proposed in https://arxiv.org/abs/2012.09838


All the metrics used in our paper will be printed, including
* pixel accuracy
* iou score
* dice coefficient
* precision score
* recall score

The threshold for creating bounding boxes is set to 0.5 by default, but you can choose any number form 0.0 to 1.0 as threshold by using `--threshold` argument.

### To measure the ABPC score,
```
python save_h5.py --method=agcam  --save_root=.\saveroot --data_root=.\ILSVRC
python ABPC.py --method=agcam --h5_root=.\saveroot --csv=True --file=True
```

<save_h5.py>

This code saves the [heatmap, image, class label] set in a hdf5 file for the heatmaps generated by the selected method.
Note that you need to provide the path to save the file in `--save_root` and the root of the dataset in `--data_root`.

**Make sure that the save path exist.** The code does not create a new folder.
Note that the size of the resulting hdf5 file is very big.

You can choose the method to evaluation by using the `--method` argument. 
Type as follows to evaluate each method:
- 'agcam': AGCAM
- 'rollout': Attention Rollout, proposed in https://arxiv.org/abs/2005.00928
- 'lrp': The LRP-based method devised for ViT, proposed in https://arxiv.org/abs/2012.09838


<ABPC.py>
The ABPC.py will read the saved hdf5 file and calculate the ABPC score.
You can choose the method to evaluate same as above.

Note that you need to provide the path where the hdf5 file is located. You can provide the path that you have provided in `--save_root` when running save_h5.py file.
It can produce a csv file that shows the ABPC score of all images by `--csv` and save the average result in a txt file by `--file` argument.
These two result files will be generated in the folder where your hdf5 file was located.

### A notebook of the visual explanation and the class-specific visual explanation provided by our method

The Jypyter notbook of file 'visualization.ipynb' contains the visualization of the sample images.

