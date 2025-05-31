# Identifying Cells in Breast Cancer H&E Stains. DATA3888 Imaging Group 07.

## DELIVERABLES Documentation

The DELIVERABLES folder contains all the files required for the final submission. The following is the documentation for DELIVERABLES.

This folder contains the main deliverables of the project, including the Shiny application, Jupyter Notebooks for model training and evaluation, and other project data.

### Folder Structure

```
DELIVERABLES/
├── 4fold_strat_down_up_no_leakage.ipynb
├── app.py
├── models_code/
│   ├── SimpleCNN/
│   │   ├── CNN50_train_on_centered_test_on_uncentered.ipynb
│   │   └── CNN100_train_on_centered_test_on_uncentered.ipynb
│   ├── HOGRGB_KNN_RF/
│   │   └── hogrgb_knn_rf.ipynb
│   ├── InceptionV3/
│   │   ├── inceptionv3_train_4_models_on_centered_data.ipynb
│   │   └── inceptionv3_create_final_model.ipynb
│   ├── ResNet50/
│   │   └── resnet50.ipynb
│   ├── VGG16/
│   │   └── VGG16_train_4_models_on_centered_data.ipynb
│   └── VGG19/
│       ├── VGG19_train_4_models_on_centered_data.ipynb
│       └── VGG19_create_final_model.ipynb
│   └── evaluate_model_on_uncentered.ipynb
├── projectdata/
│   ├── custom_style.css
│   ├── images/
│   ├── metadata_code/
│   └── model_h5_files/
└── uncentred_4fold_strat.ipynb
```

### File Descriptions

#### 1. Shiny Application (`app.py`)

The `app.py` file is the Shiny application used to visualize the performance of the finally selected model and to allow users to upload their own images to predict tumor cells.

**How to run the Shiny application:**

1.  Ensure that the necessary Python packages are installed (e.g., `shiny`, `pandas`, `numpy`, `tensorflow`, `Pillow`, etc.).
2.  In the terminal, navigate to the `DELIVERABLES` folder.
3.  Run the following command:
    ```bash
    shiny run app.py
    ```
    This will start a local server, and you can access the application in a web browser.

**Important Prerequisites for Running the Shiny Application:**

To ensure `app.py` runs successfully, please confirm that the following files and directory structure are configured as required under the `DELIVERABLES/projectdata/` path:

- **Evaluation Image File (`EVALUATION_IMAGE_PATH`)**:
  - The application expects this file to be located at: `projectdata/metadata_code/GSM7780153_Post-Xenium_HE_Rep1.ome.tif`
  - **Note**: This `.ome.tif` file is large and **is not included in the repository; it needs to be added manually**.

- **Model Directory (`MODEL_DIR`)**:
  - The application expects model files to be located in: `projectdata/model_h5_files/` (specific subfolders like `train_on_centered/` may be specified in the code).
  - Please ensure this directory contains the pre-trained model files (usually in `.h5` format) required by the application.

- **Test Data Directory (`TEST_DATA_DIR`)**:
  - The application expects test images to be located in: `projectdata/images/uncentred_ternary_224_ALL/`

- **Model Files**:
  - Please reconfirm that `projectdata/model_h5_files/` contains the correct `.h5` model files that can be loaded by the application.

#### 2. Root Directory Jupyter Notebooks

- **`4fold_strat_down_up_no_leakage.ipynb`**:
  - **Purpose**: This notebook is used to generate a dataset with four partitions and no leakage for 4-fold stratified cross-validation. It includes down-sampling and up-sampling as needed and introduces a "blank" image category. Specific steps include:
    1.  Splitting the original large image into 4 quadrants (Q1-Q4).
    2.  Ignoring cells that might overlap between quadrants at the maximum image size (e.g., 224x224) to prevent data leakage.
    3.  Within each quadrant, performing stratified sampling of tumor and non-tumor cells for a binary classification task. Stratification is based on cell groups (e.g., immune cells, connective tissue cells, etc.) and cell types.
    4.  Performing down-sampling or up-sampling as needed to meet the required number of samples for each category (e.g., `TOTAL_SAMPLE_SIZE / 2` images for each of the tumor and non-tumor groups in each quadrant).
    5.  Outputting the final images to a new folder as training/testing data and adding a third "blank" image category for multi-class classification tasks.
  - **Usage**: Open in a Jupyter environment and execute the notebook cells sequentially. The corresponding original image data and metadata need to be prepared. Key parameters such as `TOTAL_SAMPLE_SIZE`, `MAX_IMAGE_SIZE`, `BIG_IMAGE_PATH`, `IMAGE_DIR` (original small image path), `OUTPUT_BASE` (processed image output path), and `EMPTY_CLASS_SIZE` can be configured in the notebook.

- **`uncentred_4fold_strat.ipynb`**:
  - **Purpose**: This notebook is similar to `4fold_strat_down_up_no_leakage.ipynb` but primarily handles **uncentered** image grids. It is designed to perform the following:
    1.  Splitting the entire slide image (`.tif` file) into a grid based on the desired image size (e.g., 224x224).
    2.  Generating 4 data folds based on the quadrants (Q1-Q4) of the large image.
    3.  Labeling the grids based on the number of tumor cells contained within each grid (and whether non-tumor cells are present or if it's completely empty). This process utilizes cell boundary information.
    4.  Within each quadrant, performing stratified sampling of grids of different categories (e.g., empty, non-tumor only, few tumor cells, many tumor cells) for multi-class classification tasks.
    5.  Performing down-sampling or up-sampling as needed to meet the required number of samples for each category.
    6.  Outputting the final image grids to a new folder as training/testing data.
    7.  In addition to generating sampled datasets, this notebook also generates a complete dataset containing all valid grids for evaluation purposes.
  - **Usage**: Open in a Jupyter environment and execute the notebook cells sequentially. The corresponding original image data (`BIG_IMAGE_PATH`), cell boundary data (`CBR_PATH`), and cell annotation data (`ANNO_PATH`) need to be prepared. Key parameters such as `DESIRED_IMAGE_SIZE`, `TOTAL_SAMPLE_SIZE`, `EMPTY_CLASS_SIZE`, output paths (`OUTPUT_BASE_TERNARY` and `OUTPUT_BASE_MULTI`), and whether to generate ternary or multi-class data (`GENERATE_TERNARY_DATA`, `GENERATE_MULTI_DATA`) can be configured in the constants section at the beginning of the notebook.

#### 3. Model-related Code (`models_code/`)

This folder contains Jupyter Notebooks for training and evaluating different image recognition models. Each subfolder corresponds to a specific model architecture.

- **`models_code/SimpleCNN/`**:
  - **`CNN50_train_on_centered_test_on_uncentered.ipynb`**:
    - **Purpose**: Trains four independent simple CNN models on 50x50 pixel centered data using a 4-fold cross-validation method. In each fold, data from one quadrant is used as the test set, and data from the remaining three quadrants are merged for training. Four different models are obtained. Finally, the performance of each model is evaluated on the uncentered dataset.
    - **Usage**: Run in a Jupyter environment. `IMAGE_DIR` (pointing to centered training data) and `UNCENTERED_DIR` (pointing to uncentered test data) need to be set; these are preset in the code but may need manual adjustment if file locations change or files are not found. Models will be trained and evaluated for each fold.
  - **`CNN100_train_on_centered_test_on_uncentered.ipynb`**:
    - **Purpose**: Similar to `CNN50_train_on_centered_test_on_uncentered.ipynb`, this notebook trains four independent simple CNN models on 100x100 pixel centered data using a 4-fold cross-validation method. In each fold, data from one quadrant is used as the test set, and data from the remaining three quadrants are merged for training. Four different models are obtained. Finally, the performance of each model is evaluated on the uncentered dataset.
    - **Usage**: Run in a Jupyter environment. `IMAGE_DIR` (pointing to centered training data) and `UNCENTERED_DIR` (pointing to uncentered test data) need to be set; these are preset in the code but may need manual adjustment if file locations change or files are not found. Models will be trained and evaluated for each fold.

- **`models_code/HOGRGB_KNN_RF/`**:
  - **`hogrgb_knn_rf.ipynb`**:
    - **Purpose**: Performs binary classification of tumor vs. non-tumor using HOG-RGB features. First, HOG features for each color channel are extracted from 100x100 pixel centered data. These features are then fed into K-Nearest Neighbors (KNN) and Random Forest (RF) classifiers. A 4-fold cross-validation strategy is employed, and performance is evaluated on the same data. For KNN, GridSearchCV is used to determine the optimal number of neighbors `k`.
    - **Usage**: Run in a Jupyter environment. `IMAGE_DIR` (pointing to centered training data) needs to be set. It is preset in the code but may require manual adjustment if the file location changes or the file is not found. The notebook will compute HOG-RGB features, then train and evaluate both classifiers.

- **`models_code/InceptionV3/`**:
  - **`inceptionv3_train_4_models_on_centered_data.ipynb`**:
    - **Purpose**: Trains four independent InceptionV3 models on 299x299 pixel centered data using a 4-fold cross-validation method. In each fold, data from one quadrant is used as the test set, and data from the remaining three quadrants are merged for training. Four different models are obtained.
    - **Usage**: Run in a Jupyter environment. `base_dir` needs to be set to point to the directory containing the 4-fold stratified data. It is preset in the code but may require manual adjustment if the file location changes or the file is not found. After running all code, .h5 model files for each fold will be generated.
  - **`inceptionv3_create_final_model.ipynb`**:
    - **Purpose**: Trains a final InceptionV3 model using 299x299 pixel centered data. First, 10% of the data from each quadrant is reserved as a final validation set. Then, the remaining 90% of data from all quadrants is merged to train this model.
    - **Usage**: Run in a Jupyter environment. `base_dir` needs to be set to point to the directory containing the 4-fold stratified data. It is preset in the code but may require manual adjustment if the file location changes or the file is not found. After running all code, the final .h5 model file will be generated.

- **`models_code/ResNet50/`**:
  - **`resnet50.ipynb`**:
    - **Purpose**: Trains four independent ResNet50 models on 224x224 pixel centered data using a 4-fold cross-validation method. In each fold, data from one quadrant is used as the test set, and data from the remaining three quadrants are merged for training. Four different models are obtained.
    - **Usage**: Open in a Jupyter environment and execute notebook cells sequentially. `TRAIN_DATA_DIR` (pointing to centered training data) and `TEST_DATA_DIR` (pointing to uncentered test data) need to be set. These are preset in the code but may require manual adjustment if file locations change or files are not found. After running all code, .h5 model files for each fold will be generated.

- **`models_code/VGG16/`**:
  - **`VGG16_train_4_models_on_centered_data.ipynb`**:
    - **Purpose**: Trains four independent VGG16 models on 224x224 pixel centered data using a 4-fold cross-validation method. In each fold, data from one quadrant is used as the test set, and data from the remaining three quadrants are merged for training. Four different models are obtained.
    - **Usage**: Run in a Jupyter environment. `base_dir` needs to be set to point to the directory containing the 4-fold stratified data. It is preset in the code but may require manual adjustment if the file location changes or the file is not found. After running all code, .h5 model files for each fold will be generated.

- **`models_code/VGG19/`**:
  - **`VGG19_train_4_models_on_centered_data.ipynb`**:
    - **Purpose**: Trains four independent VGG19 models on 224x224 pixel centered data using a 4-fold cross-validation method. In each fold, data from one quadrant is used as the test set, and data from the remaining three quadrants are merged for training. Four different models are generated.
    - **Usage**: Run in a Jupyter environment. `base_dir` needs to be set to point to the directory containing the 4-fold stratified data. It is preset in the code but may require manual adjustment if the file location changes or the file is not found. After running all code, .h5 model files for each fold will be generated.
  - **`VGG19_create_final_model.ipynb`**:
    - **Purpose**: Trains a final VGG19 model using 224x224 pixel centered data. Similar to the final model training for InceptionV3, this notebook may reserve a portion of data from each quadrant as a final validation set and then merge the remaining data to train this model.
    - **Usage**: Run in a Jupyter environment. `base_dir` needs to be set to point to the directory containing the 4-fold stratified data. It is preset in the code but may require manual adjustment if the file location changes or the file is not found. After running all code, the final .h5 model file will be generated.

- **`models_code/evaluate_model_on_uncentered.ipynb`**:
  - **Purpose**: This Jupyter Notebook is used to evaluate the performance of pre-trained models on uncentered image data. It can load specified model files and perform predictions and evaluations against the uncentered test dataset, thereby understanding the model's generalization ability on images closer to real-world scenarios (without pre-processing alignment).
  - **Usage**: Open and execute this notebook in a Jupyter environment. You need to ensure that the correct model file path (usually an `.h5` file) and the path to the uncentered test data are specified. The notebook will output relevant evaluation metrics such as accuracy, confusion matrix, etc.

    **Note**: Only the 4-fold cross-validation models for InceptionV3, ResNet50, VGG16, and VGG19 will be evaluated in this file. SimpleCNN50/100 and HOG_KNN_RF also complete their evaluations within their respective training model files.

**General Usage Instructions (for `.ipynb` files):**

1.  Ensure your Python environment has all necessary libraries installed (e.g., `tensorflow`, `keras`, `scikit-learn`, `matplotlib`, `numpy`, `pandas`, `torch`, `torchvision`, `skimage`, etc.).
2.  Start Jupyter Notebook or JupyterLab.
3.  Navigate to the appropriate directory (`DELIVERABLES/` or subfolders under `DELIVERABLES/models_code/`).
4.  Open the corresponding `.ipynb` file.
5.  Follow the instructions and execute the code cells in the notebook sequentially. Typically, you will need to load data, then perform pre-processing, feature extraction, model training, or evaluation.
6.  Ensure that dataset paths and relevant constants are correctly configured in the notebook.

#### 4. Project Data (`projectdata/`)

This folder contains other data and resources that may be needed by the Shiny application and the model training process.

- **`custom_style.css`**: Custom Cascading Style Sheet (CSS) file for the Shiny application, used to define its appearance.
- **`images/`**: Stores image resources, including original small images, processed datasets, etc.
- **`metadata_code/`**: Stores metadata for the original H&E stained tissue images, such as `GSM7780153_Post-Xenium_HE_Rep1.ome.tif` (note: this file is not in the repository and needs to be added manually) and cell coordinate files (`cbr.csv`, `41467_2023_43458_MOESM4_ESM.xlsx`).
- **`model_h5_files/`**: Used to store trained model weight files (usually in `.h5` or `.pth` format). These files will be loaded by `app.py` or evaluation notebooks.

### Precautions

- Before running any code, ensure that all necessary dependencies are installed. It is recommended to use a virtual environment to manage project dependencies.
- You may need to adjust file path configurations in the code according to your specific setup.
- Model training can be very time-consuming and may require significant computational resources (e.g., GPU).