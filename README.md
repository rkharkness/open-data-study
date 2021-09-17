# The pitfalls of using open data to develop deep learning solutions for COVID-19 detection in chest X-rays

**This repository contains the source code requierd to reproduce the main results presented in our paper: 

Harkness, R., Ravikumar, N., Zucker, K. [The pitfalls of using open data to develop deep learning solutions for COVID-19 detection](https://arxiv.org/ftp/arxiv/papers/2109/2109.08020.pdf)**

Since the emergence of COVID-19, deep learning models have been developed to identify COVID-19 from chest X-rays. With little to no direct access to hospital data, the AI community relies heavily on public data comprising numerous data sources. Model performance results have been exceptional when training and testing on open-source data, surpassing the reported capabilities of AI in pneumonia-detection prior to the COVID-19 outbreak. In this study impactful models are trained on a widely used open-source data and tested on an external test set and a hospital dataset, for the task of classifying chest X-rays into one of three classes: COVID-19, non-COVID pneumonia and no-pneumonia. Data analysis and model evalutions show that the popular open-source dataset COVIDx is not representative of the real clinical problem andthat results from testing on this are inflated. Dependence on open-source data can leave models vulnerable to bias and confounding variables, requiring careful analysis to developclinically useful/viable AI tools for COVID-19 detection in chest X-rays.

## COVIDx data analysis
The proportions of misaligned labels in the pneumonia class of COVIDx were identified. We found that only 6.13% of RSNA pneumonia were accurately labelled within the COVIDx data. The figure below shows the frequency of the eight most commonly observed alternative pathologies included in the COVIDx pneumonia data, with all of these pathologies occurring more frequently than pneumonia. Many of these exist as comorbidities in a single chest X-ray, further obscuring the true pneumonia class.
<p align="center">
  <img src="https://github.com/rkharkness/open-data-study/blob/master/assets/covidx-pathology-freq.png"/>
</p>

## COVIDx-trained model generalisability
In this component of our research we aim to assess bias within models trained on flawed COVIDX data. In assessing the generalisability of existing AI approaches for the detection of COVID-19 from chest X-rays study,  we selected three highly cited models: COVIDNet [1], DarkCovidNet [2] and CoroNet [3]. We then trained these models on a balanced version of COVIDX and evaluated on two external test datasets, hospital data (LTHT) and external test data.

ROC curves (below) reflect exceptional performance across all chosen models when tested on the COVIDx test data. 

<p align="center">
  <img src="https://github.com/rkharkness/open-data-study/blob/master/assets/model-generalisability-roc.png">
  <i>
    ROC curves of (A) DarkCovidNet (B) CoroNet and (C) COVIDNet when evaluated on COVIDX test data.
  </i>
</p>

However, testing on non-COVIDx data shows a steep drop in all model performances, this is highlighted by the confusion matrices.

<p align="center">
  <img src="https://github.com/rkharkness/open-data-study/blob/master/assets/model-generalisability-cm.png">
  <i>
    Confusion matrices of prediction of external data for (A) DarkCovidNet, (C) CoroNet and (E) COVIDNet. Confusion matrices of prediction of LTHT data (B) DarkCovidNet, (D)         CoroNet and (F) COVIDNet.
  </i>
</p>

## Frankenstein data study

The term *Frankenstein* datasets refers to datasets made up of other datasets, this appears to be a common theme in data in the problem domain. To evaluate the impact of using *Frankenstein* training data we train and test a deep CNN of our own design to separate COVIDX images according to data source. Our deep CNN model performs incredibly well, separating image by source with high accuracy.

<p align="center">
  <img src="https://github.com/rkharkness/open-data-study/blob/master/assets/frankenstein-dcnn-results.png">
  <i>
    Frankenstein Deep CNN prediction performance presented in (A) Confusion matrix and (B) ROC curve.
  </i>
</p>

We also use t-SNE to present the 2D projection of the features learned by our deep CNN model. The distinct clustering of the source-specific features demonstrates the easy separation of COVID-19 negative repositories from COVID-19 positive.

<p align="center">
  <img src="https://github.com/rkharkness/open-data-study/blob/master/assets/frankenstein-tsne-plot.png">
  <i>
    2D t-SNE projection of hidden features extracted from the trained Deep CNN ’Frankenstein’ classifier during inference.
  </i>
</p>

## References

[1] Wang, L., Lin, Z.Q. & Wong, A. COVID-Net: a tailored deep convolutional neural network design for detection of COVID-19 cases from chest X-ray images. Sci Rep 10, 19549 (2020). https://doi.org/10.1038/s41598-020-76550-z

[2] Ozturk, T., Talo, M., Yildirim, E. A., Baloglu, U. B., Yildirim, O., & Rajendra Acharya, U. (2020). Automated detection of COVID-19 cases using deep neural networks with X-ray images. Computers in biology and medicine, 121, 103792. https://doi.org/10.1016/j.compbiomed.2020.103792

[3] Khobahi, S., Agarwal, C., & Soltanalian, M. (2020). CoroNet: A Deep Network Architecture for Semi-Supervised Task-Based Identification of COVID-19 from Chest X-ray Images. medRxiv.
