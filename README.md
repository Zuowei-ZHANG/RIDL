# The Representation of Imprecision in Deep Learning techniques(RIDL)
This is the avaiable code for the paper ["Representation of Imprecision in Deep Neural Networks for Image Classification: Active Learning via Belief Functions"].
## Requirements
PyTorch==1.7.1+cu110, python==3.7.9, and more in `requirements.txt`
Install python dependencies
`pip install -r requirements.txt`
Codes for  basic cnn models are in the file `cnn_models`
## To get the potential labels
The file `main_function.py` provides a demo about how to get the potential labels and their corresponding manually checked labels after training, which are files `y_dot_alpha-x.xx_DCNN-1.txt`, `y_dot_alpha-x.xx_DCNN-2.txt`.

## To train the DCNN model
The file `DCNN.py` obtained the ultimate results, which contains parameters for selecting the dataset and different benchmark models. 

## Evaluation code
Evaluation code is `evaluation.py`,which will be called in `DCNN.py`.

## Pre-trained models

Because manually checked also costs a lot of time, and to facilitate viewing the results and save time, we save corresponding 'y_dot' file and the pre-trained parameters, by modifying the value of the `switch`. These files are located in the `model_dataset` folder, We save the parameter files for different datasets on different benchmark models when alpha=0.01 and data errer rate=0.01, which will facilitate the replication of experimental data.

Model parameter files are saved as .pkl files. To use the pre-trained parameters simply make `switch=1` and run `DCNN.py`. 

The file `model_dataset` includes the parameters of some trained DCNN  that are used in the demo. These parameter files are saved in `model_saved`.
## Data set description
We randomly labeled 1% of the training images as any other incorrect category, It is reflected in the `sign_error_matrix_0.01.txt`, including the index and modified labels.

The imagewoof-5 and Flowers datasets used in the experiments are available at [imagewoof](https://github.com/fastai/imagenette) and [flowers-recognition](https://www.kaggle.com/alxmamaev/flowers-recognition), respectively.
In particular, we used the 5 subclasses ('n02086240','n02087394','n02096294','n02111889','n02115641') in  imagewoof datasets as the new dataset imagewoof-5.
We packaged the dataset into three .h5 files( `xxx_train.h5`, `xxx_test.h5`, `xxx_val.h5`) according to the division ratio instructions in the text, containing images and corresponding labels.

Due to the overwhelming amount of data, we chose to upload the potential labels, data and model parameters files to [Google Drive](https://drive.google.com/file/d/1nO3imk0qz8M_-fnL1P3j9rZ6qm-Joi5Q/view?usp=sharing), which you can download to replace the existing `model_dataset`. You can also download the [imagewoof](https://github.com/fastai/imagenette) and [flowers-recognition](https://www.kaggle.com/alxmamaev/flowers-recognition) dataset, but that means you need to to modify the code that loads the data accordingly and retrain the model.


