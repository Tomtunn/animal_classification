# Animal classification
This object of this project is to classify animals using the [Animal-10 dataset](https://www.kaggle.com/alessiocorrado99/animals10) from kaggle. There are over 28,000 photos in the dataset, divided into 10 classes of animal images. The model is trained using a pretrained model ([resnet18](https://pytorch.org/hub/pytorch_vision_resnet/)) and fine-tuned on the dataset.


## Installation
```sh
git clone https://github.com/Tomtunn/animal_classification.git
cd animal_classification
pip install -r requirements.txt
```