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

### 3. Aggregator Evaluation (`aggregator_evaluation/`)

This folder contains scripts and results for aggregating and evaluating the performance metrics of different models.

- **`Aggregated_metric_evaluation_results.ipynb`**:

  - **Purpose**: The main purpose of this Jupyter Notebook is to integrate and independently run all aggregator evaluation methods actually used in the Shiny application. After executing this Notebook, various performance metrics are calculated, unified, output, and stored in a CSV file named `aggregated_model_evaluation_results.csv`.
  - **Usage**: Open in a Jupyter environment and execute the Notebook cells sequentially. After execution, the relevant performance metrics will be calculated and saved to the `aggregated_model_evaluation_results.csv` file (usually generated in the same directory).

- **`aggregated_model_evaluation_results.csv`**:
  - **Purpose**: This is a CSV file generated by `Aggregated_metric_evaluation_results.ipynb`. It stores various performance metrics calculated through aggregator evaluation methods, such as accuracy, precision, recall, F1 score, etc., as well as potentially related model configurations or parameters.
  - **Usage**: This file is the output of `Aggregated_metric_evaluation_results.ipynb`. Its content can be used for subsequent analysis, visualization, or directly viewed using spreadsheet software or data analysis tools.

### 4. Model-Related Code (`models_code/`)

This folder contains Jupyter Notebooks for training and evaluating various image recognition models. Each subfolder corresponds to a specific model architecture and **includes an `outputs/` directory where the performance metrics for each cross-validation fold are stored.**

- **`models_code/SimpleCNN/`**:

  - **Purpose**: This directory contains notebooks for training and evaluating simple Convolutional Neural Network (CNN) models. The naming convention of the `.ipynb` files indicates the specifics of the training and testing data. For instance, "50" in the filename refers to the use of 50x50 datasets, while "100" refers to 100x100 datasets. These notebooks typically employ a 4-fold cross-validation strategy.
  - **Usage**: Run in a Jupyter environment. Ensure that the paths to the corresponding training and testing datasets are correctly set within each notebook. The models are trained and evaluated for each fold.

- **`models_code/HOGRGB_KNN_RF/`**:

  - **`hogrgb_knn_rf.ipynb`**:
    - **Purpose**: Performs binary classification of tumor vs. non-tumor using HOG-RGB features. First, HOG features for each color channel are extracted from 100x100 pixel centered data. These features are then fed into K-Nearest Neighbors (KNN) and Random Forest (RF) classifiers. A 4-fold cross-validation strategy is employed, and performance is evaluated on the same data. For KNN, GridSearchCV is used to determine the optimal number of neighbors `k`.
    - **Usage**: Run in a Jupyter environment. `IMAGE_DIR` (pointing to centered training data) needs to be set. It is preset in the code but may require manual adjustment if the file location changes or the file is not found. The notebook will compute HOG-RGB features, then train and evaluate both classifiers.

- **`models_code/InceptionV3/`**:

  - **`inceptionv3_train_4_models_on_centered_data.ipynb`**:
    - **Purpose**: Trains four independent InceptionV3 models on 100x100 pixel centered data using a 4-fold cross-validation method. In each fold, data from one quadrant is used as the test set, and data from the remaining three quadrants are merged for training. Four different models are obtained. This file is only for training and generating models; evaluation is done in `evaluate_model_on_uncentered.ipynb`.
    - **Usage**: Run in a Jupyter environment. `base_dir` needs to be set to point to the directory containing the 4-fold stratified data. It is preset in the code but may require manual adjustment if the file location changes or the file is not found. After running all code, .h5 model files for each fold will be generated.
  - **`inceptionv3_create_final_model.ipynb`**:
    - **Purpose**: Trains a final InceptionV3 model using 100x100 pixel centered data. First, 10% of the data from each quadrant is reserved as a final validation set. Then, the remaining 90% of data from all quadrants is merged to train this model.
    - **Usage**: Run in a Jupyter environment. `base_dir` needs to be set to point to the directory containing the 4-fold stratified data. It is preset in the code but may require manual adjustment if the file location changes or the file is not found. After running all code, the final .h5 model file will be generated.

- **`models_code/ResNet50/`**:

  - **`resnet50.ipynb`**:
    - **Purpose**: Trains four independent ResNet50 models on 100x100 pixel centered data using a 4-fold cross-validation method. In each fold, data from one quadrant is used as the test set, and data from the remaining three quadrants are merged for training. Four different models are obtained. This file is only for training and generating models; evaluation is done in `evaluate_model_on_uncentered.ipynb`.
    - **Usage**: Open in a Jupyter environment and execute notebook cells sequentially. `TRAIN_DATA_DIR` (pointing to centered training data) and `TEST_DATA_DIR` (pointing to uncentered test data) need to be set. These are preset in the code but may require manual adjustment if file locations change or files are not found. After running all code, .h5 model files for each fold will be generated.

- **`models_code/VGG16/`**:

  - **`centered_train_uncenter_test_VGG16.ipynb`**:
    - **Purpose**: Trains four independent VGG16 models on 100x100 pixel centered data using 4-fold cross-validation. In each fold, data from one quadrant is used as the test set, and data from the remaining three quadrants are merged for training. Four different models are obtained. This file is only for training and generating models; evaluation is done in `evaluate_model_on_uncentered.ipynb`.
    - **Usage**: Run in a Jupyter environment. Paths for `train_folder_Q1` to `train_folder_Q4` (for centered training data) and `test_folder_Q1` to `test_folder_Q4` (for uncentered test data) need to be set. Generates `.h5` model files for each fold.

- **`models_code/VGG19/`**:

  - **`VGG19_train_4_models_on_centered_data.ipynb`**:
    - **Purpose**: Trains four independent VGG19 models on 100x100 pixel centered data using a 4-fold cross-validation method. In each fold, data from one quadrant is used as the test set, and data from the remaining three quadrants are merged for training. Four different models are generated. This file is only for training and generating models; evaluation is done in `evaluate_model_on_uncentered.ipynb`.
    - **Usage**: Run in a Jupyter environment. `base_dir` needs to be set to point to the directory containing the 4-fold stratified data. It is preset in the code but may require manual adjustment if the file location changes or the file is not found. After running all code, .h5 model files for each fold will be generated.
  - **`VGG19_create_final_model.ipynb`**:
    - **Purpose**: Trains a final VGG19 model using 100x100 pixel centered data. Similar to the final model training for InceptionV3, this notebook may reserve a portion of data from each quadrant as a final validation set and then merge the remaining data to train this model.
    - **Usage**: Run in a Jupyter environment. `base_dir` needs to be set to point to the directory containing the 4-fold stratified data. It is preset in the code but may require manual adjustment if the file location changes or the file is not found. After running all code, the final .h5 model file will be generated.

- **`models_code/evaluate_model_on_uncentered.ipynb`**:

  - **Purpose**: This Jupyter Notebook is used to evaluate the performance of pre-trained models on 224x224 pixel uncentered image data. It can load specified model files and perform predictions and evaluations against the uncentered test dataset, thereby understanding the model's generalization ability on images closer to real-world scenarios (without pre-processing alignment).
  - **Usage**: Open and execute this notebook in a Jupyter environment. You need to ensure that the correct model file path (usually an `.h5` file) and the path to the uncentered test data are specified. The notebook will output relevant evaluation metrics such as accuracy, confusion matrix, etc.

    **Note**: Only the 4-fold cross-validation models for InceptionV3, ResNet50, VGG16, and VGG19 will be evaluated in this file. SimpleCNN and HOG_KNN_RF also complete their evaluations within their respective training model files.

**General Usage Instructions (for `.ipynb` files):**

1.  Ensure your Python environment has all necessary libraries installed (e.g., `tensorflow`, `keras`, `scikit-learn`, `matplotlib`, `numpy`, `pandas`, `torch`, `torchvision`, `skimage`, etc.).
2.  Start Jupyter Notebook or JupyterLab.
3.  Navigate to the appropriate directory (`DELIVERABLES/` or subfolders under `DELIVERABLES/models_code/`).
4.  Open the corresponding `.ipynb` file.
5.  Follow the instructions and execute the code cells in the notebook sequentially. Typically, you will need to load data, then perform pre-processing, feature extraction, model training, or evaluation.
6.  Ensure that dataset paths and relevant constants are correctly configured in the notebook.

### 5. Project Data (`projectdata/`)

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

- You may need to adjust file path configurations within the code according to your specific local setup.
- Model training can be time-consuming and may require significant computational resources.
