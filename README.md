# Leveraging Synthetic Data

In this project, I investigate the potential benefits of synthetic data for training deep learning models, seeking to overcome the hurdles of high costs and extensive time required in real data collection and labelling. The central question I aim to answer is the effectiveness of synthetic data in the training process of deep learning models, especially those deployed in computer vision object classification tasks.

For quantitative comparison, I trained two models: one using a synthetic dataset and another using a subset of the real ImageNet dataset. A highlight of this project is my development of an efficient synthetic data generation scheme, designed specifically for this study but versatile enough to cater to other computer vision tasks.

The outcomes reveal a noticeable improvement in model performance (an increase by 13.16% over the baseline) when trained on a mixed dataset, made up of ImageNet and the newly generated synthetic data. In addition, I demonstrate that replacing a large chunk of synthetic data with real data can still achieve performance levels nearly equivalent to the baseline.

Conducted as the final project for my MSc in Machine Learning, these findings underscore the value of synthetic data as a viable complement to real data in certain scenarios.

**Instructions**

In order to run inference use the pytorch/infer_synth.py file. 
The code should be executed against ImageNet test set – attached in imagenet_test directory
The code should load pre trained model – 3 pre trained models are attached in saved_models directory

Execution parameters:
To test baseline_100real.pth.tar
--incl-background
0
-m
resnet50
--print-freq
10
--restore-checkpoint
/saved_models/baseline_100real.pth.tar
--data
/imagenet_test/

To test 100real_100synth.pth.tar
--incl-background
1
-m
resnet50
--print-freq
10
--restore-checkpoint
/saved_models/100real_100synth.pth.tar
--data
/imagenet_test/

To test 33real_100synth.pth.tar
--incl-background 
1 
-m resnet50 
--print-freq 
10 
--restore-checkpoint 
/saved_models/33real_100synth.pth.tar 
--data 
/imagenet_test/
