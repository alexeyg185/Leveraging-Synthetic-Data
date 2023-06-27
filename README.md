# Leveraging Synthetic Data

In this project, I investigate the potential benefits of synthetic data for training deep learning models, seeking to overcome the hurdles of high costs and extensive time required in real data collection and labelling. The central question I aim to answer is the effectiveness of synthetic data in the training process of deep learning models, especially those deployed in computer vision object classification tasks.

For quantitative comparison, I trained two models: one using a synthetic dataset and another using a subset of the real ImageNet dataset. A highlight of this project is my development of an efficient synthetic data generation scheme, designed specifically for this study but versatile enough to cater to other computer vision tasks.

The outcomes reveal a noticeable improvement in model performance (an increase by 13.16% over the baseline) when trained on a mixed dataset, made up of ImageNet and the newly generated synthetic data. In addition, I demonstrate that replacing a large chunk of synthetic data with real data can still achieve performance levels nearly equivalent to the baseline.

Conducted as the final project for my MSc in Machine Learning, these findings underscore the value of synthetic data as a viable complement to real data in certain scenarios.
