# DELIVERABLES Documentation

This folder contains the main deliverables of the project, including the Shiny application, Jupyter Notebooks for model training and evaluation, and other project data.

## Folder Structure

```
DELIVERABLES/
├── app.py
├── models_code/
│   ├── VGG16_train_4_models_on_centered_data.ipynb
│   ├── VGG19_create_final_model.ipynb
│   ├── VGG19_train_4_models_on_centered_data.ipynb
│   ├── evaluate_model_on_uncentered.ipynb
│   ├── inceptionv3_create_final_model.ipynb
│   └── inceptionv3_train_4_models_on_centered_data.ipynb
└── projectdata/
    ├── custom_style.css
    ├── images/
    ├── metadata_code/
    └── model_h5_files/
```

## File Descriptions

### 1. Shiny Application (`app.py`)

The `app.py` file is the Shiny application for cell identification.

**How to run the Shiny application:**

1.  Ensure you have installed the necessary Python packages (e.g., `shiny`, `pandas`, `numpy`, `tensorflow`, `Pillow`, etc.).
2.  In your terminal, navigate to the `DELIVERABLES` folder.
3.  Run the following command:
    ```bash
    shiny run app.py
    ```
    This will start a local server, and you can access the application in your web browser.

**Important: Prerequisites for running the Shiny application**

To ensure `app.py` runs successfully, please confirm that the following files and directory structures are configured as required within the `DELIVERABLES/projectdata/` path:

*   **Evaluation Image File (`EVALUATION_IMAGE_PATH`)**:
    *   The application expects this file at: `projectdata/metadata_code/GSM7780153_Post-Xenium_HE_Rep1.ome.tif`
    *   **Note**: This `.ome.tif` file is large and is **not included in the code repository**. You will need to **manually obtain this file and place it in the specified path**.

*   **Model Directory (`MODEL_DIR`)**:
    *   The application expects model files to be located at: `projectdata/model_h5_files/train_on_centered/`
    *   Please ensure this directory contains the required, pre-trained model files (usually in `.h5` format) for the application.

*   **Test Data Directory (`TEST_DATA_DIR`)**:
    *   The application expects test images to be located at: `projectdata/images/uncentred_ternary_224_ALL/`

*   **Sample Image Path (`SAMPLE_IMAGE_PATH`)**:
    *   The application expects a sample image at: `projectdata/images/Q1_quadrant.png`

*   **Model Files**:
    *   Please re-confirm that `projectdata/model_h5_files/` (especially the `train_on_centered` subdirectory) contains the correct `.h5` model files that can be loaded by the application.

### 2. Model-Related Code (`models_code/`)

This folder contains Jupyter Notebooks for training and evaluating different image recognition models.

*   **`VGG16_train_4_models_on_centered_data.ipynb`**:
    *   **Purpose**: Divides the original images into four quadrants, then trains four distinct models using the VGG16 architecture on centered data. Each model is trained using data from three of the four quadrants.
    *   **Usage**: Open in a Jupyter environment and execute the notebook cells sequentially. You will need to have the appropriate training dataset prepared.

*   **`VGG19_create_final_model.ipynb`**:
    *   **Purpose**: Combines data from all four quadrants to train a single, final model using the VGG19 architecture.
    *   **Usage**: Open in a Jupyter environment and execute the code. It may require loading previously saved model weights or data from the quadrant-specific training.

*   **`VGG19_train_4_models_on_centered_data.ipynb`**:
    *   **Purpose**: Divides the original images into four quadrants, then trains four distinct models using the VGG19 architecture on centered data. Each model is trained using data from three of the four quadrants.
    *   **Usage**: Similar to `VGG16_train_4_models_on_centered_data.ipynb`, run in a Jupyter environment.

*   **`evaluate_model_on_uncentered.ipynb`**:
    *   **Purpose**: Evaluates the performance of trained models on uncentered data.
    *   **Usage**: Open in a Jupyter environment, load the model you wish to evaluate and the corresponding uncentered test dataset, then execute the code.

*   **`inceptionv3_create_final_model.ipynb`**:
    *   **Purpose**: Combines data from all four quadrants to train a single, final model using the InceptionV3 architecture.
    *   **Usage**: Open in a Jupyter environment and execute the code. It may require loading previously saved model weights or data from the quadrant-specific training.

*   **`inceptionv3_train_4_models_on_centered_data.ipynb`**:
    *   **Purpose**: Divides the original images into four quadrants, then trains four distinct models using the InceptionV3 architecture on centered data. Each model is trained using data from three of the four quadrants.
    *   **Usage**: Similar to `VGG16_train_4_models_on_centered_data.ipynb`, run in a Jupyter environment.

**General Usage Instructions (for `.ipynb` files):**

1.  Ensure your Python environment has all necessary libraries installed (e.g., `tensorflow`, `keras`, `scikit-learn`, `matplotlib`, `numpy`, `pandas`, etc.).
2.  Start Jupyter Notebook or JupyterLab.
3.  Navigate to the `DELIVERABLES/models_code/` directory.
4.  Open the respective `.ipynb` file.
5.  Follow the instructions and execute code cells sequentially within the notebook. Typically, you will need to load data, then perform preprocessing, model training, or evaluation.
6.  Ensure that dataset paths are correctly configured within the notebooks.

### 3. Project Data (`projectdata/`)

This folder contains additional data and resources that may be required by the Shiny application and model training processes.

*   **`custom_style.css`**: Custom Cascading Style Sheet (CSS) file for the Shiny application, used to define its appearance.
*   **`images/`**: Stores image resources.
*   **`metadata_code/`**: Stores metadata for the original H&E stained tissue images.
*   **`model_h5_files/`**: Used to store trained model weight files (usually in `.h5` format). These files will be loaded by `app.py` or the evaluation notebooks.

## Notes

*   Before running any code, ensure you have installed all necessary dependencies. It is recommended to use a virtual environment to manage project dependencies.
*   You may need to adjust file path configurations in the code according to your specific setup.
*   Model training can be time-consuming and may require significant computational resources (e.g., a GPU).