# Identifying Cells in Breast Cancer H&E Stains. DATA3888 Imaging Group 07.

## DELIVERABLES Documentation

The `DELIVERABLES` folder contains all the files required for the final project submission. This document provides an overview and usage instructions for these deliverables.

This folder houses the core components of the project, including the Shiny application for interactive model evaluation and prediction, Jupyter Notebooks detailing model training and evaluation processes, and other essential project data.

## File Descriptions

### 1. Shiny Application (`app.py`)

The `app.py` file is a Shiny application designed to visualize the performance of the final selected model. It also allows users to upload their own images for Tumour cell prediction.

**How to Run the Shiny Application:**

1.  Ensure all necessary Python packages are installed (e.g., `shiny`, `pandas`, `numpy`, `tensorflow`, `Pillow`, etc.).
2.  In your terminal, navigate to the `DELIVERABLES` folder.
3.  Execute the following command:
    ```bash
    shiny run app.py
    ```
    This will launch a local server, and the application can be accessed via a web browser.

**Important Prerequisites for Running the Shiny Application:**

To ensure `app.py` runs successfully, please verify that the following files and directory structure are correctly configured under the `DELIVERABLES/projectdata/` path:

- **Evaluation Image File (`EVALUATION_IMAGE_PATH`)**:

  - The application expects this file at: `projectdata/metadata_code/GSM7780153_Post-Xenium_HE_Rep1.ome.tif`
  - **Note**: This `.ome.tif` file is large and **is not included in the repository. You must obtain this file manually and place it in the specified path.**

- **Model Directory (`MODEL_DIR`)**:

  - The application expects model files to be located in: `projectdata/model_h5_files/` (specific subfolders like `train_on_centered/` may be specified within the code).
  - Please ensure this directory contains the pre-trained model files (typically in `.h5` format) required by the application.

- **Test Data Directory (`TEST_DATA_DIR`)**:

  - The application expects test images to be located in: `projectdata/images/uncentred_ternary_224_ALL/`

- **Model Files**:
  - Please double-check that `projectdata/model_h5_files/` contains the correct `.h5` model files that can be loaded by the application.

### 2. Root Directory Jupyter Notebooks

- **`4fold_strat_down_up_no_leakage.ipynb`**:

  - **Purpose**: This notebook generates a dataset with four partitions, ensuring no data leakage for 4-fold stratified cross-validation. It incorporates downsampling and upsampling as required and introduces a "blank" image category. Key steps include:
    1.  Splitting the original large image into 4 quadrants (Q1-Q4).
    2.  Ignoring cells that might overlap between quadrants at the maximum image size (e.g., 224x224) to prevent data leakage.
    3.  Performing stratified sampling of tumor and non-tumor cells within each quadrant for a binary classification task. Stratification is based on cell groups (e.g., immune cells, connective tissue cells) and cell types.
    4.  Applying downsampling or upsampling to achieve the target number of samples for each category (e.g., `TOTAL_SAMPLE_SIZE / 2` images for tumor and non-tumor groups in each quadrant).
    5.  Outputting the final images to a new folder for training/testing and adding a third "blank" image category for multi-class tasks.
  - **Usage**: Open in a Jupyter environment and execute the notebook cells sequentially. Ensure that the appropriate raw image data and metadata are prepared. Key parameters such as `TOTAL_SAMPLE_SIZE`, `MAX_IMAGE_SIZE`, `BIG_IMAGE_PATH`, `IMAGE_DIR` (path to original small images), `OUTPUT_BASE` (output path for processed images), and `EMPTY_CLASS_SIZE` can be configured within the notebook.

- **`uncentred_4fold_strat.ipynb`**:
  - **Purpose**: Similar to `4fold_strat_down_up_no_leakage.ipynb`, this notebook primarily handles **uncentered** image grids. It is designed to:
    1.  Split the entire slide image (`.tif` file) into a grid based on the desired image size (e.g., 224x224).
    2.  Generate 4 data folds based on the quadrants (Q1-Q4) of the large image.
    3.  Label grids based on the number of tumor cells they contain (and whether non-tumor cells are present or if it's completely empty), utilizing cell boundary information.
    4.  Perform stratified sampling of different grid categories (e.g., empty, non-tumor only, few tumor cells, many tumor cells) within each quadrant for a multi-class task.
    5.  Apply downsampling or upsampling as needed to meet the required sample count for each category.
    6.  Output the final image grids to a new folder for training/testing.
    7.  In addition to generating sampled datasets, this notebook also produces a complete dataset containing all valid grids for evaluation purposes.
  - **Usage**: Open in a Jupyter environment and execute the notebook cells sequentially. Ensure that the appropriate raw image data (`BIG_IMAGE_PATH`), cell boundary data (`CBR_PATH`), and cell annotation data (`ANNO_PATH`) are prepared. Key parameters such as `DESIRED_IMAGE_SIZE`, `TOTAL_SAMPLE_SIZE`, `EMPTY_CLASS_SIZE`, output paths (`OUTPUT_BASE_TERNARY` and `OUTPUT_BASE_MULTI`), and flags for generating ternary or multi-class data (`GENERATE_TERNARY_DATA`, `GENERATE_MULTI_DATA`) can be configured in the constants section at the beginning of the notebook.

### 3. Model-Related Code (`models_code/`)

This folder contains Jupyter Notebooks for training and evaluating various image recognition models. Each subfolder corresponds to a specific model architecture and **includes an `outputs/` directory where the performance metrics for each cross-validation fold are stored.**

- **`models_code/SimpleCNN/`**:

  - **Purpose**: This directory contains notebooks for training and evaluating simple Convolutional Neural Network (CNN) models. The naming convention of the `.ipynb` files indicates the specifics of the training and testing data. For instance, "50" in the filename refers to the use of 50x50 datasets, while "100" refers to 100x100 datasets. These notebooks typically employ a 4-fold cross-validation strategy.
  - **Usage**: Run in a Jupyter environment. Ensure that the paths to the corresponding training and testing datasets are correctly set within each notebook. The models are trained and evaluated for each fold.

- **`models_code/HOGRGB_KNN_RF/`**:

  - **`hogrgb_knn_rf.ipynb`**:
    - **Purpose**: Utilizes HOG-RGB features for binary classification (tumor vs. non-tumor). HOG features are extracted from each color channel of 100x100 pixel images. These features are then used to train K-Nearest Neighbors (KNN) and Random Forest (RF) classifiers with a 4-fold cross-validation strategy. GridSearchCV is used for KNN to find the optimal `k`.
    - **Usage**: Run in a Jupyter environment. `IMAGE_DIR` (centered training data) path needs to be set. The notebook computes features, then trains and evaluates both classifiers.

- **`models_code/InceptionV3/`**:

  - **`inceptionv3_train_4_models_on_centered_data.ipynb`**:
    - **Purpose**: Trains four independent InceptionV3 models on centered data using 4-fold cross-validation. In each fold, one quadrant's data is the test set, and the remaining three are merged for training.
    - **Usage**: Run in a Jupyter environment. `base_dir` (pointing to 4-fold stratified data) needs to be set. Input image size is 299x299. Generates `.h5` model files for each fold.
  - **`inceptionv3_create_final_model.ipynb`**:
    - **Purpose**: Trains a final InceptionV3 model. 10% of data from each quadrant is reserved for final validation. The remaining 90% from all quadrants is merged for training.
    - **Usage**: Run in a Jupyter environment. `base_dir` needs to be set. Input image size is 299x299. Generates the final `.h5` model file.

- **`models_code/ResNet50/`**:

  - **`resnet50.ipynb`**:
    - **Purpose**: Trains four independent ResNet50 models on centered data using 4-fold cross-validation (train on 3 quadrants, test on 1). Input image size is typically 224x224.
    - **Usage**: Open in a Jupyter environment and execute cells. `TRAIN_DATA_DIR` (centered training data) and `TEST_DATA_DIR` (uncentered test data) paths need to be set. Generates `.h5` model files for each fold.

- **`models_code/VGG16/`**:

  - **`centered_train_uncenter_test_VGG16.ipynb`**:
    - **Purpose**: Trains four independent VGG16 models using 4-fold cross-validation. In each fold, the model is trained on centered image data from three quadrants and tested on uncentered image data from the remaining quadrant. Input image size is typically 224x224.
    - **Usage**: Run in a Jupyter environment. Paths for `train_folder_Q1` to `train_folder_Q4` (for centered training data) and `test_folder_Q1` to `test_folder_Q4` (for uncentered test data) need to be set. Generates `.h5` model files for each fold.

- **`models_code/VGG19/`**:
  - **`VGG19_train_4_models_on_centered_data.ipynb`**:
    - **Purpose**: Trains four independent VGG19 models on centered data using 4-fold cross-validation.
    - **Usage**: Run in a Jupyter environment. `base_dir` (4-fold stratified data) needs to be set. Input image size is typically 224x224. Generates `.h5` model files for each fold.
  - **`VGG19_create_final_model.ipynb`**:
    - **Purpose**: Trains a final VGG19 model, potentially reserving a portion of data from each quadrant for validation and merging the rest for training.
    - **Usage**: Run in a Jupyter environment. `base_dir` needs to be set. Input image size is typically 224x224. Generates the final `.h5` model file.

**General Usage Instructions (for `.ipynb` files):**

1.  Ensure your Python environment has all necessary libraries installed (e.g., `tensorflow`, `keras`, `scikit-learn`, `matplotlib`, `numpy`, `pandas`, `torch`, `torchvision`, `skimage`, etc.).
2.  Start Jupyter Notebook or JupyterLab.
3.  Navigate to the appropriate directory (`DELIVERABLES/` or subfolders under `DELIVERABLES/models_code/`).
4.  Open the respective `.ipynb` file.
5.  Follow the instructions and execute the code cells sequentially. This typically involves loading data, preprocessing, feature extraction, model training, or evaluation.
6.  Verify that dataset paths and relevant constants are correctly configured in each notebook.

### 4. Project Data (`projectdata/`)

This folder contains additional data and resources required by the Shiny application and model training processes.

- **`custom_style.css`**: Custom Cascading Style Sheets (CSS) file for the Shiny application, defining its appearance.
- **`images/`**: Stores various image resources for the project. These datasets are categorized by their generation method and characteristics:
  - **`50_stratified4fold_1000per_seed3888`**:
    - **Source**: Generated by `4fold_strat_down_up_no_leakage.ipynb`.
    - **Description**: Contains 50x50 pixel cell-centered images, sampled from the original image (divided into 4 quadrants). Each quadrant aims for ~500 NonTumour, ~500 Tumour, and ~100 empty cell samples via selection, downsampling, or upsampling (image augmentation).
  - **`100_stratified4fold_1000per_seed3888`**:
    - **Source**: Generated by `4fold_strat_down_up_no_leakage.ipynb`.
    - **Description**: Contains 100x100 pixel cell-centered images, with similar sampling strategy and counts per quadrant as the 50x50 dataset.
  - **`uncentred_ternary_224_stratified4fold_1000per_seed3888`**:
    - **Source**: Generated by `uncentred_4fold_strat.ipynb`.
    - **Description**: Contains 224x224 pixel non-cell-centered images. Sampled from 4 quadrants, aiming for ~500 uncentered NonTumour, ~500 uncentered Tumour, and ~100 uncentered empty cell samples per quadrant.
  - **`uncentred_ternary_224_ALL`**:
    - **Source**: Generated by `uncentred_4fold_strat.ipynb`.
    - **Description**: Contains all valid 224x224 pixel tiles extracted from annotated regions of the original data, divided by the 4 quadrants. This dataset includes all qualifying tiles without specific sampling counts, primarily intended for comprehensive model evaluation.
- **`metadata_code/`**: Stores metadata for the original H&E stained tissue images, such as `GSM7780153_Post-Xenium_HE_Rep1.ome.tif` (note: this large file is not included in the repository and must be added manually) and cell coordinate files (`cbr.csv`, `41467_2023_43458_MOESM4_ESM.xlsx`).
- **`model_h5_files/`**: Used to store trained model weight files (typically in `.h5` or `.pth` format). These files are loaded by `app.py` or evaluation notebooks.

## Notes

- Before running any code, ensure that all necessary dependencies are installed. It is highly recommended to use a virtual environment to manage project dependencies.
- You may need to adjust file path configurations within the code according to your specific local setup.
- Model training can be time-consuming and may require significant computational resources (e.g., GPU).
