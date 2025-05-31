# DELIVERABLES Documentation

This folder contains the main deliverables of the project, including the Shiny application, Jupyter Notebooks for model training and evaluation, and other project data.

## Folder Structure

```
DELIVERABLES/
├── 4fold_strat_down_up_no_leakage.ipynb
├── app.py
├── models_code/
│   ├── CNN/
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
├── projectdata/
│   ├── custom_style.css
│   ├── images/
│   ├── metadata_code/
│   └── model_h5_files/
└── uncentred_4fold_strat.ipynb
```

## File Descriptions

### 1. Shiny Application (`app.py`)

The `app.py` file is a Shiny application used to visualize the performance of the final selected model and to allow users to upload their own images to predict Tumour cells.

**How to Run the Shiny Application:**

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
  - **Note**: This `.ome.tif` file is large and **is not included in the repository. You need to obtain this file manually and place it in the specified path.**

- **Model Directory (`MODEL_DIR`)**:

  - The application expects model files to be located in: `projectdata/model_h5_files/` (specific subfolders like `train_on_centered/` may be specified in the code).
  - Please ensure this directory contains the pre-trained model files (usually in `.h5` format) required by the application.

- **Test Data Directory (`TEST_DATA_DIR`)**:

  - The application expects test images to be located in: `projectdata/images/uncentred_ternary_224_ALL/`

- **Model Files**:
  - Please double-check that `projectdata/model_h5_files/` contains the correct `.h5` model files that can be loaded by the application.

### 2. Root Directory Jupyter Notebooks

- **`4fold_strat_down_up_no_leakage.ipynb`**:

  - **Purpose**: This notebook is used to generate a dataset with four partitions and no leakage for 4-fold stratified cross-validation. It includes downsampling and upsampling as needed and introduces a "blank" image category. Specific steps include:
    1.  Splitting the original large image into 4 quadrants (Q1-Q4).
    2.  Ignoring cells that might overlap between quadrants at the maximum image size (e.g., 224x224) to prevent data leakage.
    3.  Within each quadrant, performing stratified sampling of tumor and non-tumor cells for a binary classification task. Stratification is based on cell groups (e.g., immune cells, connective tissue cells, etc.) and cell types.
    4.  Performing downsampling or upsampling as needed to meet the required number of samples for each category (e.g., `TOTAL_SAMPLE_SIZE / 2` images for tumor and non-tumor groups in each quadrant).
    5.  Outputting the final images to a new folder as training/test data and adding a third "blank" image category for multi-class tasks.
  - **Usage**: Open in a Jupyter environment and execute the notebook cells sequentially. Appropriate raw image data and metadata need to be prepared. Key parameters such as `TOTAL_SAMPLE_SIZE`, `MAX_IMAGE_SIZE`, `BIG_IMAGE_PATH`, `IMAGE_DIR` (path to original small images), `OUTPUT_BASE` (output path for processed images), and `EMPTY_CLASS_SIZE` can be configured in the notebook.

- **`uncentred_4fold_strat.ipynb`**:
  - **Purpose**: This notebook is similar to `4fold_strat_down_up_no_leakage.ipynb` but primarily deals with **uncentered** image grids. It is designed to perform the following:
    1.  Splitting the entire slide image (`.tif` file) into a grid based on the desired image size (e.g., 224x224).
    2.  Generating 4 data folds based on the quadrants (Q1-Q4) of the large image.
    3.  Labeling grids based on the number of tumor cells contained within each grid (and whether non-tumor cells are present or if it's completely empty). This process utilizes cell boundary information.
    4.  Within each quadrant, performing stratified sampling of different categories of grids (e.g., empty, non-tumor only, few tumor cells, many tumor cells) for a multi-class task.
    5.  Performing downsampling or upsampling as needed to meet the required number of samples for each category.
    6.  Outputting the final image grids to a new folder as training/test data.
    7.  In addition to generating sampled datasets, this notebook also generates a complete dataset containing all valid grids for evaluation purposes.
  - **Usage**: Open in a Jupyter environment and execute the notebook cells sequentially. Appropriate raw image data (`BIG_IMAGE_PATH`), cell boundary data (`CBR_PATH`), and cell annotation data (`ANNO_PATH`) need to be prepared. Key parameters such as `DESIRED_IMAGE_SIZE`, `TOTAL_SAMPLE_SIZE`, `EMPTY_CLASS_SIZE`, output paths (`OUTPUT_BASE_TERNARY` and `OUTPUT_BASE_MULTI`), and whether to generate ternary or multi-class data (`GENERATE_TERNARY_DATA`, `GENERATE_MULTI_DATA`) can be configured in the constants section at the beginning of the notebook.

### 3. Model-Related Code (`models_code/`)

This folder contains Jupyter Notebooks for training and evaluating different image recognition models. Each subfolder corresponds to a specific model architecture.

- **`models_code/CNN/`**:

  - **`CNN50_train_on_centered_test_on_uncentered.ipynb`**:
    - **Purpose**: Trains a simple Convolutional Neural Network (CNN) model using 50x50 pixel centered images and tests it using corresponding uncentered images. A 4-fold cross-validation strategy is employed, where data from 3 quadrants are used for training, and data from 1 quadrant is used for testing (using its uncentered version).
    - **Usage**: Run in a Jupyter environment. `IMAGE_DIR` (pointing to centered training data) and `UNCENTERED_DIR` (pointing to uncentered test data) need to be set. These are pre-set in the code, but manual adjustment is needed if file locations change or files are not found. The model will be trained and evaluated for each fold.
  - **`CNN100_train_on_centered_test_on_uncentered.ipynb`**:
    - **Purpose**: Similar to the `CNN50...` notebook, but uses 100x100 pixel images. Trains a simple CNN model and performs 4-fold cross-validation and testing on uncentered data.
    - **Usage**: Run in a Jupyter environment. `IMAGE_DIR` (pointing to centered training data) and `UNCENTERED_DIR` (pointing to uncentered test data) need to be set. These are pre-set in the code, but manual adjustment is needed if file locations change or files are not found. The model will be trained and evaluated for each fold.

- **`models_code/HOGRGB_KNN_RF/`**:

  - **`hogrgb_knn_rf.ipynb`**:
    - **Purpose**: Uses HOG-RGB features for binary classification of tumor vs. non-tumor. First, HOG features for each color channel are extracted from 100x100 pixel images. Then, these features are fed into K-Nearest Neighbors (KNN) and Random Forest (RF) classifiers. A 4-fold cross-validation strategy is employed. For KNN, GridSearchCV is used to determine the optimal number of neighbors `k`.
    - **Usage**: Run in a Jupyter environment. `IMAGE_DIR` (pointing to centered training data) needs to be set. This is pre-set in the code, but manual adjustment is needed if file locations change or files are not found. The notebook will compute HOG-RGB features, then train and evaluate both classifiers.

- **`models_code/InceptionV3/`**:

  - **`inceptionv3_train_4_models_on_centered_data.ipynb`**:
    - **Purpose**: Trains four independent InceptionV3 models on centered data using a 4-fold cross-validation method. In each fold, data from one quadrant serves as the test set, and data from the remaining three quadrants are merged for training. This results in 4 different models.
    - **Usage**: Run in a Jupyter environment. `base_dir` (pointing to the directory containing 4-fold stratified data) needs to be set. This is pre-set in the code, but manual adjustment is needed if file locations change or files are not found. Input image size is 299x299. After running all code, .h5 model files for each fold will be generated.
  - **`inceptionv3_create_final_model.ipynb`**:
    - **Purpose**: Trains a final InceptionV3 model. First, 10% of the data from each quadrant is reserved as a final validation set. Then, the remaining 90% of data from all quadrants is merged to train the InceptionV3 model.
    - **Usage**: Run in a Jupyter environment. `base_dir` (pointing to the directory containing 4-fold stratified data) needs to be set. This is pre-set in the code, but manual adjustment is needed if file locations change or files are not found. Input image size is 299x299. After running all code, the final .h5 model file will be generated.

- **`models_code/ResNet50/`**:

  - **`resnet50.ipynb`**:
    - **Purpose**: Trains four independent ResNet50 models on centered data using a 4-fold cross-validation method. In each fold, data from one quadrant serves as the test set, and data from the remaining three quadrants are merged for training. This results in 4 different models. Input image size is typically 224x224.
    - **Usage**: Open in a Jupyter environment and execute notebook cells sequentially. `TRAIN_DATA_DIR` (pointing to centered training data) and `TEST_DATA_DIR` (pointing to uncentered test data) need to be set. These are pre-set in the code, but manual adjustment is needed if file locations change or files are not found. After running all code, .h5 model files for each fold will be generated.

- **`models_code/VGG16/`**:

  - **`VGG16_train_4_models_on_centered_data.ipynb`**:
    - **Purpose**: Trains four independent VGG16 models on centered data using a 4-fold cross-validation method. In each fold, data from one quadrant serves as the test set, and data from the remaining three quadrants are merged for training. This results in 4 different models. Input image size is typically 224x224.
    - **Usage**: Run in a Jupyter environment. `base_dir` (pointing to the directory containing 4-fold stratified data) needs to be set. This is pre-set in the code, but manual adjustment is needed if file locations change or files are not found. After running all code, .h5 model files for each fold will be generated.

- **`models_code/VGG19/`**:
  - **`VGG19_train_4_models_on_centered_data.ipynb`**:
    - **Purpose**: Trains four independent VGG19 models on centered data using a 4-fold cross-validation method. In each fold, data from one quadrant serves as the test set, and data from the remaining three quadrants are merged for training. This generates 4 different models.
    - **Usage**: Run in a Jupyter environment. `base_dir` (pointing to the directory containing 4-fold stratified data) needs to be set. This is pre-set in the code, but manual adjustment is needed if file locations change or files are not found. Input image size is typically 224x224. After running all code, .h5 model files for each fold will be generated.
  - **`VGG19_create_final_model.ipynb`**:
    - **Purpose**: Trains a final VGG19 model. Similar to the final InceptionV3 model training, this notebook may reserve a portion of data from each quadrant as a final validation set, then merge the remaining data to train the VGG19 model.
    - **Usage**: Run in a Jupyter environment. `base_dir` (pointing to the directory containing 4-fold stratified data) needs to be set. This is pre-set in the code, but manual adjustment is needed if file locations change or files are not found. Input image size is typically 224x224. After running all code, the final .h5 model file will be generated.

**General Usage Instructions (for `.ipynb` files):**

1.  Ensure that your Python environment has all necessary libraries installed (e.g., `tensorflow`, `keras`, `scikit-learn`, `matplotlib`, `numpy`, `pandas`, `torch`, `torchvision`, `skimage`, etc.).
2.  Start Jupyter Notebook or JupyterLab.
3.  Navigate to the appropriate directory (`DELIVERABLES/` or subfolders under `DELIVERABLES/models_code/`).
4.  Open the respective `.ipynb` file.
5.  Follow the instructions and execute the code cells in the notebook sequentially. Typically, you will need to load data, then perform preprocessing, feature extraction, model training, or evaluation.
6.  Ensure that dataset paths and relevant constants are correctly configured in the notebook.

### 4. Project Data (`projectdata/`)

This folder contains additional data and resources that may be needed by the Shiny application and model training processes.

- **`custom_style.css`**: Custom Cascading Style Sheets (CSS) file for the Shiny application, used to define its appearance.
- **`images/`**: Stores various image resources required for the project. Based on their generation method and characteristics, these mainly include the following types of datasets:
  - **`100_stratified4fold_1000per_seed3888`**:
    - **Source**: Generated by `4fold_strat_down_up_no_leakage.ipynb`.
    - **Description**: This dataset contains 100x100 pixel cell-centered images, sampled from the original image which was divided into 4 quadrants. Each quadrant aims to provide approximately 500 NonTumour cell samples, 500 Tumour cell samples, and 100 empty cell samples through selection, downsampling, or upsampling (image augmentation transformations) of the original cell images.
  - **`uncentred_ternary_224_stratified4fold_1000per_seed3888`**:
    - **Source**: Generated by `uncentred_4fold_strat.ipynb`.
    - **Description**: This dataset contains 224x224 pixel images that are not cell-centered. Similar to `100_stratified4fold_1000per_seed3888`, these are also selected from the 4 quadrants of the original image, aiming to obtain approximately 500 uncentered NonTumour cell samples, 500 uncentered Tumour cell samples, and 100 uncentered empty cell samples.
  - **`uncentred_ternary_224_ALL`**:
    - **Source**: Generated by `uncentred_4fold_strat.ipynb`.
    - **Description**: This dataset contains all valid 224x224 pixel tiles extracted from the annotated regions of the original data. These tiles are also divided according to the 4 quadrants of the original image. However, unlike the sampled versions, this dataset includes all qualifying tiles without selection, downsampling, or upsampling for specific sample counts, primarily intended for comprehensive model evaluation.
- **`metadata_code/`**: Stores metadata for the original H&E stained tissue images, such as `GSM7780153_Post-Xenium_HE_Rep1.ome.tif` (note: this large file is not included in the repository and needs to be added manually) and cell coordinate files (`cbr.csv`, `41467_2023_43458_MOESM4_ESM.xlsx`).
- **`model_h5_files/`**: Used to store trained model weight files (usually in `.h5` or `.pth` format). These files will be loaded by `app.py` or evaluation notebooks.

## Notes

- Before running any code, ensure that all necessary dependencies are installed. It is recommended to use a virtual environment to manage project dependencies.
- You may need to adjust file path configurations in the code according to your specific setup.
- Model training can be very time-consuming and may require significant computational resources (e.g., GPU).
