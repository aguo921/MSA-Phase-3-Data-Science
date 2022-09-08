# Required Software:
* [Python 3.9+](https://www.python.org/downloads/)
* Text Editor for Python ([Visual Studio Code](https://code.visualstudio.com/) with [Python extension](https://marketplace.visualstudio.com/items?itemName=ms-python.python) installed is recommended)

Recommended:
* [Anaconda](https://www.anaconda.com/)
* [Virtualenv](https://virtualenv.pypa.io/en/latest/installation.html)

# Setting Up:
Install the libraries listed in the `requirements.txt` via the command:
```bash
pip install -r requirements.txt
```

Recommended to use a virtual environment either from either [virtualenv](https://docs.python.org/3/library/venv.html) or [anaconda](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html). This setup helps isolate project dependencies from user dependencies as to avoid version conflicts.

Download the [CIFAR-10 dataset](https://www.cs.toronto.edu/~kriz/cifar.html) and place it into the project directory.

# Usage:

## Processing the data

The `data_processing.py` file contains functions to process the data from the dataset.

## Training the model

The `make-model.ipynb` notebook trains the model using the dataset.

Run
```
tensorboard --logdir output/logs/
```
to run [tensorboard](https://github.com/tensorflow/tensorboard/blob/master/README.md). The main points of interests are accuracy and loss graphs. The tensorboard application can then be opened via the site `http://localhost:6006/`.

## Validating the model

The `use-model.ipynb` notebook uses the trained model to make and validate predictions.

## Hyperparameter tuning

The `hyperparameter-tuning.ipynb` notebook performs hyperparameter tuning of the model.

Run
```
tensorboard --logdir logs/hparam_tuning
```
to run [tensorboard](https://github.com/tensorflow/tensorboard/blob/master/README.md), open the tensorboard application via the site `http://localhost:6006/`, and navigate to the HParams dashboard.

## Model report

The `report.ipynb` notebook contains the report on the modelling process and performance.

## Image classification application

The `app.py` file contains the image classification application.

Run
```
streamlit run app.py
```
to run [streamlit](https://streamlit.io/) and the app will automatically run in a new tab in your browser.

You can upload image files to the app and it will return the probability of the image being an airplane.