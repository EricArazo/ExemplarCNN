# Official PyTorch implementation for the IMVIP 2019 paper: "Improving Unsupervised Learning with ExemplarCNNs"

### To train a model:
- Download the STL-10 datset into ./data.
- Use "create_surrogate_classes_002.ipynb" to randomly select 16000 images from the unlabeled set from STL10.
- Use "create_surrogate_data_STL.py" to randomly crop and augment patches for each of the previously selected images.
- Use "./surrogate_dataset/splitting_data_small_val.py" to generate a validation and train set.
- Use "./training_scripts/exp083.py" to train a model.

### To evaluate the model:
- Extract features from teh testing sed using "./evaluation/extract_32x32_largeNet.py"
- Train linear SVMs on each predefined-fold with "./evaluation/eval_long.ipynb"



### Accuracies on STL10

|Algorithm| Accuracy (%) |
|----|----|
|ExemplarCNN (Dosovitsky et al., 2015)|74.2 ± 0.4|
|ExemplarCNN (ours)|74.14 ± 0.39|
|Clustering (ours)|76.42 ± 0.35|

### Please consider citing the following paper if you find this work useful for your research.


```
 @inproceedings{IMVIP2019_ImproveExemplarCNN,
  title = {Improving Unsupervised Learning with ExemplarCNNs},
  authors = {Eric Arazo and Noel E O'Connor and Kevin McGuinness},
  booktitle = {Irish Machine Vision and Image Processing (ICML)},
  month = {August},
  year = {2019}
 }
```

<!---I still need to re-run experiments.
Careful: The scripts save models, images, and so on... Make sure everything is in the gitignore!-->

