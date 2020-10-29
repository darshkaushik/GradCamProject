# Grad-CAM for the skin-mnist dataset for skin lesion diagnosis

`This projects is currently under development and relevant scripts are yet to be added`  
This project uses a convnet model for detection of skin lesion detection and uses Grad-CAM for explaing the prediction of the model.

## Data
---

This Project uses a modified version of the [skin-mnist dataset](https://www.kaggle.com/kmader/skin-cancer-mnist-ham10000) Which only contains the classes `nv`, `bkl`, `mel` (around 550 images of each).

- `nv` - Melanocytic nevi are benign neoplasms of melanocytes and appear in a myriad of variants, which all are included in our series. The variants may differ significantly from a dermatoscopic point of view.

- `bkl` - "Benign keratosis" is a generic class that includes seborrheic ker- atoses ("senile wart"), solar lentigo - which can be regarded a flat variant of seborrheic keratosis - and lichen-planus like keratoses (LPLK), which corresponds to a seborrheic keratosis or a solar lentigo with inflammation and regression. The three subgroups may look different dermatoscop- ically, but we grouped them together because they are similar biologically and often reported under the same generic term histopathologically. From a dermatoscopic view, lichen planus-like keratoses are especially challeng- ing because they can show morphologic features mimicking melanoma and are often biopsied or excised for diagnostic reasons.

- `mel` - Melanoma is a malignant neoplasm derived from melanocytes that may appear in different variants. If excised in an early stage it can be cured by simple surgical excision. Melanomas can be invasive or non-invasive (in situ). We included all variants of melanoma including melanoma in situ, but did exclude non-pigmented, subungual, ocular or mucosal melanoma.

## Model
---
We used transfer learning on different models and selected `Final Model` according to the stats we obtained from the training.

## Grad-CAM
---

## Process
---

### Phase 1 - Data Hunt and Preprocessing
- [x] Search about suitable datasets
- [x] Preprocess the data so that it is properly encoded and clean
- [x] Write Custom Pytorch Datasets and DataLoaders

### Phase 2 - Experimentation
- [x] Writing a training loop with checkpoint system
- [x] Writing utility methods for managing checkpoints and saved files
- [x] Trying out different configurations of the models given above
- [x] Selecting the best model and configuration
- [x] Converting the .ipynb file into scripts to push into the repo

### Phase 3 - Grad-CAM
- [ ] Reading and understanding the details about Grad-CAM from the paper.
- [ ] Implementing inference method with built in Grad-CAM

### Phase 4 - Deployment
- [ ] Write a web app in flask for deploying the model to make it more accessible
- [ ] Deplow the web app along with the model files online 
