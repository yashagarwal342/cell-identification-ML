# ----------------------------
# Import libraries
# ----------------------------
from shiny import App, ui, render, reactive
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import random
import gc
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    confusion_matrix, accuracy_score, balanced_accuracy_score, 
    f1_score, precision_score, recall_score
)
from PIL import ImageDraw, Image
import shutil
import os
import tensorflow.keras.backend as K
from tensorflow.keras.preprocessing import image  # Used for image to array conversion and preprocessing
from tensorflow.keras.models import load_model
import pickle
from collections import deque, defaultdict
import warnings
import hashlib
warnings.filterwarnings("ignore")
import pickle
import time

# ----------------------------
# Upload the image saving path
# ----------------------------
UPLOAD_SAVE_DIR = os.path.join("projectdata", "images", "uploads")

EVALUATION_IMAGE_PATH =  os.path.join("projectdata", "metadata_code", "GSM7780153_Post-Xenium_HE_Rep1.ome.tif") # path to the big .tif image of all cells
MODEL_DIR = os.path.join("projectdata", "model_h5_files", "train_on_centered")
TEST_DATA_DIR =  os.path.join("projectdata", "images", "uncentred_ternary_224_ALL") 
SAMPLE_IMAGE_PATH = os.path.join("projectdata", "images", "Q1_quadrant.png") # path to a sample image for testing

IMG_DIM = (224, 224)
BATCH_SIZE = 128 # will only affect batches for prediction I believe
IMG_SIZE = 224

CACHE_DIR = os.path.join("projectdata", "cache")
if not os.path.exists(CACHE_DIR):
    os.makedirs(CACHE_DIR)

# ----------------------------
# List of model names (for dropdown selection)
# ----------------------------
MODEL_NAMES = [
    "VGG19",
    "InceptionV3"
]

type_to_label_map = {
    'Empty': 0,
    'Non-Tumor': 1,
    'Tumor': 2
}
label_names = ["Empty", "Non-Tumor", "Tumor"]


QUADRANT_DICT = {"Q1": 1, "Q2": 2, "Q3": 3, "Q4": 4}

# key = (model_name, quadrant)，value = (X_ids, X_test, y_test, y_pred)
prediction_cache = {}

def get_prediction_cache_path(model_name, quadrant):
    return os.path.join(CACHE_DIR, f"pred_{model_name}_{quadrant}.pkl")

def get_heatmap_cache_path(model_name, quadrant):
    return os.path.join(CACHE_DIR, f"heatmap_{model_name}_{quadrant}.png")

def get_file_md5(filepath):
    """Calculate the md5 hash of the file"""
    hash_md5 = hashlib.md5()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()

def get_upload_pred_cache_path(model_name, file_md5):
    return os.path.join(CACHE_DIR, f"upload_pred_{model_name}_{file_md5}.pkl")

def get_upload_heatmap_cache_path(model_name, file_md5):
    return os.path.join(CACHE_DIR, f"upload_heatmap_{model_name}_{file_md5}.png")

# ----------------------------
# The image is cut into patches
# ----------------------------
def split_image_to_patches(img, patch_size=IMG_DIM):
    width, height = img.size
    patches, positions = [], []
    for y in range(0, height, patch_size[1]):
        for x in range(0, width, patch_size[0]):
            box = (x, y, x + patch_size[0], y + patch_size[1])
            patch = img.crop(box)
            patches.append(patch)
            positions.append((x, y))
    return patches, positions

# ----------------------------
# Preprocess all patches
# ----------------------------
def preprocess_patches(patches, preprocess_fn, target_size):
    processed = []
    for patch in patches:
        patch = patch.resize(target_size)
        arr = image.img_to_array(patch)
        arr = np.expand_dims(arr, axis=0)
        arr = preprocess_fn(arr)
        processed.append(arr[0])
    return np.array(processed)

# ----------------------------
# Restore the patch prediction to a heat map
# ----------------------------
def create_prediction_grid(positions, classes, img_size, patch_size):
    grid_w = img_size[0] // patch_size[0]
    grid_h = img_size[1] // patch_size[1]
    heatmap = np.zeros((grid_h, grid_w))
    for (x, y), cls in zip(positions, classes):
        row = y // patch_size[1]
        col = x // patch_size[0]
        heatmap[row, col] = cls
    return heatmap


def preprocess_image(img_path, target_size=(224, 224)):
    img = Image.open(img_path)
    img = img.resize(target_size)
    img_array = np.array(img, dtype=np.float32)
    img_array = img_array / 255.0

    img.close()
    del img # free up memory
    return img_array

def load_test_data(dir, quadrant, target_size=(224, 224)):
    """Load test data from a directory without separating by quadrants"""
    X_id = []
    X_test = []
    y_test = []


    quadrant_path = os.path.join(dir, quadrant)

    # Go through each cell type in the quadrant
    for cell_type in os.listdir(quadrant_path):
        cell_type_path = os.path.join(quadrant_path, cell_type)
        if os.path.isdir(cell_type_path):
            for filename in os.listdir(cell_type_path):
                if filename.endswith(".png") or filename.endswith(".jpg"):
                    img_path = os.path.join(cell_type_path, filename)
                    img = preprocess_image(img_path, target_size=target_size)

                    parts = filename.replace(".png", "").replace(".jpg", "").split("_")
                    grid_id = f"{parts[1]}_{parts[2]}"

                    X_id.append(grid_id)
                    X_test.append(img)
                    y_test.append(type_to_label_map[cell_type])

    return X_id, np.array(X_test), np.array(y_test)

def largest_tumour_mass(grid_IDs, labels):
    # expects grid IDs as [X_coord]_[Y_coord], and label as actual labels
    # does a BFS on grids
    # # Example input: list of (id, label) pairs
    # data = [
    #     ("1_1", "Tumor"),
    #     ("1_2", "Non-Tumor")
    # ]

    # convert to coordinate set, only containing tumour grids
    tumour_coords = set()
    for id_str, label in zip(grid_IDs, labels):
        if label == "Tumor":
            x, y = map(int, id_str.split("_"))
            tumour_coords.add((x, y))

    # BFS to find connected regions
    visited = set()
    directions = [(-1,0), (1,0), (0,-1), (0,1)]  # 4-connected neighborhood

    def bfs(start):
        queue = deque([start])
        region = [start]
        visited.add(start)

        while queue:
            x, y = queue.popleft()
            for dx, dy in directions:
                neighbor = (x + dx, y + dy)
                if neighbor in tumour_coords and neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(neighbor)
                    region.append(neighbor)
        return region

    # find the largest tumour region
    largest_region = []
    for coord in tumour_coords:
        if coord not in visited:
            region = bfs(coord)
            if len(region) > len(largest_region):
                largest_region = region

    return len(largest_region)

# HEATMAP GENERATION
def box_from_grid_id(grid_ids, image_path, quadrant):
    # this is only for use with training data grid_ids
    # grid ids were defined using entire image coords, BUT STARTING IN BOTTOM RIGHT!!
    # was a weird design chouce
    # so this code will determine pixel coords in original image
    # provides coords RELATIVE TO ENTIRE IMAGE
    print("generating coords for grid ids")
    Image.MAX_IMAGE_PIXELS = None  # Remove limit for large images
    img = Image.open(image_path)
    width, height = img.size
    x_mid, y_mid = width // 2, height // 2
    img.close()
    del img # free up memory

    left_margin = x_mid % IMG_SIZE
    top_margin = y_mid % IMG_SIZE

    coords = []
    for x, y in (map(int, grid_id.split("_")) for grid_id in grid_ids):
        left = left_margin + x * IMG_SIZE
        top = top_margin + y * IMG_SIZE
        coords.append((left, top, left + IMG_SIZE, top + IMG_SIZE))
    print("coords generated")
    return coords

def create_heatmap(image_path, coords, labels, output_path=None, show=True, quadrant=None):
    """
    Create a heatmap overlay on the original image based on grid predictions
    
    Args:
        image_path: Path to the original large image
        coords: List of bounding boxes (left, top, right, bottom) format i think, for each region
        labels: List of corresponding labels for each grid
        output_path: Optional path to save the output image
        show: Whether to display the image
        quadrant: string "Q1", "Q2", "Q3", or "Q4". if not None, will crop big image accordingly
    """
    print("creating heatmap")
    # Color mapping for labels
    color_map = {
        "Tumor": (1.0, 0.0, 0.0, 0.5),      # Red with 50% transparency
        "Non-Tumor": (0.0, 1.0, 0.0, 0.3),  # Green with 30% transparency
        "Empty": (0.0, 0.0, 1.0, 0.2)       # Blue with 20% transparency
    }
    
    # Load the original image
    Image.MAX_IMAGE_PIXELS = None  # Remove limit for large images
    img = Image.open(image_path)
    width, height = img.size

    # if quadrant is not None:
    #     if quadrant in ["Q1", "Q2", "Q3", "Q4"]:
    #         all_lefts = [box[0] for box in coords]
    #         all_tops = [box[1] for box in coords]
    #         all_rights = [box[2] for box in coords]
    #         all_bottoms = [box[3] for box in coords]

    #         min_x = max(min(all_lefts) - 1000, 0)
    #         min_y = max(min(all_tops) - 1000, 0)
    #         max_x = min(max(all_rights) + 1000, width)
    #         max_y = min(max(all_bottoms) + 1000, height)

    #         # Crop image
    #         img = img.crop((min_x, min_y, max_x, max_y)).copy()
    #         width, height = img.size

    #         # 👇 Shift coords so they align to cropped image
    #         coords = [(left - min_x, top - min_y, right - min_x, bottom - min_y) for (left, top, right, bottom) in coords]
    #     else:
    #         raise ValueError("Quadrant must be set appropriately, if not none (for whole image heatmap)!")
    # overlay = Image.new('RGBA', (width, height), (0, 0, 0, 0))

    # draw = ImageDraw.Draw(overlay)
    
    # # Draw colored rectangles for each grid
    # for (left, top, right, bottom), label in zip(coords, labels):
    #     if label in color_map:
    #         # Convert RGBA to PIL format (0-255)
    #         r, g, b, a = color_map[label]
    #         color = (int(r*255), int(g*255), int(b*255), int(a*255))
    #         draw.rectangle([left, top, right, bottom], fill=color)
    
    # # Convert original image to RGBA
    # if img.mode != 'RGBA':
    #     img = img.convert('RGBA')
      # # Composite the images
    # result = Image.alpha_composite(img, overlay)
    
    # crop if required and adjust coordinates
    if (quadrant != None):
        x_mid, y_mid = width // 2, height // 2
        quadrant_coords = [
            (0, 0, x_mid, y_mid),              # Q1
            (x_mid, 0, width, y_mid),          # Q2
            (0, y_mid, x_mid, height),         # Q3
            (x_mid, y_mid, width, height)      # Q4
        ]
        
        # Determine the quadrant offset
        crop_coords = None
        if (quadrant == "Q1"):
            crop_coords = quadrant_coords[0]
        elif (quadrant == "Q2"):
            crop_coords = quadrant_coords[1]
        elif (quadrant == "Q3"):
            crop_coords = quadrant_coords[2]
        elif (quadrant == "Q4"):
            crop_coords = quadrant_coords[3]
        else:
            raise ValueError("Quadrant must be set appropriately, if not none (for whole image heatmap)!")
        
        # Crop the image
        img = img.crop(crop_coords).copy()
        width, height = img.size
        
        # Adjust coordinates to match the cropped image
        # Subtract the offset from all coordinates
        offset_x, offset_y = crop_coords[0], crop_coords[1]
        coords = [(left - offset_x, top - offset_y, right - offset_x, bottom - offset_y) 
                 for (left, top, right, bottom) in coords]
    
    
    overlay = Image.new('RGBA', (width, height), (0, 0, 0, 0))

    draw = ImageDraw.Draw(overlay)
    
    # Draw colored rectangles for each grid
    for (left, top, right, bottom), label in zip(coords, labels):
        if label in color_map:
            # Convert RGBA to PIL format (0-255)
            r, g, b, a = color_map[label]
            color = (int(r*255), int(g*255), int(b*255), int(a*255))
            draw.rectangle([left, top, right, bottom], fill=color)
    
    # Convert original image to RGBA
    if img.mode != 'RGBA':
        img = img.convert('RGBA')
    
    # Composite the images
    result = Image.alpha_composite(img, overlay)
    
    

    # 🔁 Always use matplotlib to draw (so we can include legend in saved image)
    fig, ax = plt.subplots(figsize=(15, 15))
    ax.imshow(result)

    # Create legend
    legend_elements = [
        plt.Rectangle((0, 0), 1, 1, fc=color_map[label][:3], alpha=color_map[label][3], label=label)
        for label in color_map
    ]
    leg = ax.legend(handles=legend_elements, loc='lower center', bbox_to_anchor=(0.5, -0.1), ncol=3, fontsize=30)
    
    # 让 matplotlib 不自动压缩 legend 区域    
    leg.set_in_layout(True)

    # 用 constrained_layout 更智能保留空间
    fig.set_constrained_layout(True)

    ax.axis('off')
    ax.set_title('Prediction Heatmap')
    # plt.tight_layout()
    
    img.close()
    del img
    
    if output_path:
        fig.savefig(output_path, bbox_inches=None)
        
    
    if show:
        return plt.gcf() 
    
    plt.close(fig)
    return result

def create_comparison_heatmap(image_path, coords, true_labels, pred_labels, 
                                model_name, output_path=None, show=True, quadrant=None):
    """
    Create side-by-side heatmaps comparing true labels vs predicted labels
    
    Args:
        image_path: Path to the original large image
        coords: List of bounding boxes (left, top, right, bottom) format i think, for each region
        true_labels: List of true labels
        pred_labels: List of predicted labels
        model_name: Name of the model for title
        output_path: Optional path to save the output image
        show: Whether to display the image
        quadrant: string "Q1", "Q2", "Q3", or "Q4". if not None, will crop big image accordingly
    """
    plt.figure(figsize=(20, 10))
    
    # True labels heatmap
    plt.subplot(1, 2, 1)
    true_img = create_heatmap(image_path, coords, true_labels, show=False, quadrant=quadrant)
    plt.imshow(true_img)
    plt.title(f'True Labels', fontsize=30)
    plt.axis('off')
    
    # Predicted labels heatmap
    plt.subplot(1, 2, 2)
    pred_img = create_heatmap(image_path, coords, pred_labels, show=False, quadrant=quadrant)
    plt.imshow(pred_img)
    plt.title(f'Predicted Labels for ({model_name})', fontsize=30)
    plt.axis('off')
    
    # Create legend
    color_map = {
        "Tumor": (1.0, 0.0, 0.0, 0.5),
        "Non-Tumor": (0.0, 1.0, 0.0, 0.3),
        "Empty": (0.0, 0.0, 1.0, 0.2)
    }
    legend_elements = [plt.Rectangle((0, 0), 1, 1, fc=color_map[label][:3], 
                                        alpha=color_map[label][3], label=label) 
                        for label in color_map]
    plt.figlegend(handles=legend_elements, loc='lower center', bbox_to_anchor=(0.5, -0.1), ncol=3, fontsize=30)
    
    plt.suptitle(f'Comparison of True vs Predicted Labels for ({model_name})', fontsize=30)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
    
    if show:
        return plt.gcf()  # Return current figure
        

# ----------------------------
# Front-end UI construction
# ----------------------------
app_ui = ui.page_fluid(
ui.include_css("projectdata/custom_style.css"),
    ui.card(
        ui.h1("Applying Cell Classification Models to Whole Slide Tumour Prediction", class_="text-center display-4 fw-bold"),
        ui.h1("Image07, DATA3888, 2025, The University of Sydney", class_="text-center"),
        class_="bg-light border-0 shadow-sm"
    ),

    ui.navset_card_tab(

        # 📘 About page content
        ui.nav_panel("📖 About",
             ui.layout_columns(
                 ui.card(
                     ui.h2("📘 User Guide", class_="text-primary"),
                     ui.tags.div(
                        ui.tags.p("Welcome to the ML Classifier Dashboard! This tool allows you to use and explore predictive models for whole slide classification on histopathology (H&E Stained) images."),
                         ui.tags.ul(
                            ui.tags.li(
                                "Navigate to the ", ui.strong('Model Evaluation'), " tab to view the performance of cell classification models, when aggregated for whole slide tumour prediction."),
                             ui.tags.li(
                                "Here, we provide access to two predictive models (based on VGG19 and InceptionV3), which were strongest in our analysis, and perform comparably. You may choose to consider both of their predictions, for reliability."),
                            ui.tags.li("Click '🧹 Clear Cache' to remove cached results and force re-computation (", ui.strong('This may take a long time'), ")."),
                            ui.tags.li(
                                "Switch to the ", ui.strong('Whole Slide Prediction'), " tab to upload your own H&E stained image (.jpg, .jpeg, .png, .tif formats), and generate predictions in real time.")
                                
                         ),
                        ui.tags.p("Note: The dashboard uses pre-trained models and processes large images by dividing them into patches. All results are cached for faster repeated access.")

                     ),
                     class_="with-accent-blue"
                 ),

                 ui.card(
                     ui.h2("📊 Project Overview", class_="text-success"),
                     ui.tags.div(
                         ui.tags.ul(
                            ui.tags.li(
                                "Prior work produced a whole slide image of H&E stained breast cancer tissue, and labelled cells in this image."),
                            ui.tags.li(
                                "We divided the whole slide image into ", ui.strong('4 quadrants'), ", so we could train our models on 3 quadrants at a time while leaving the other for testing, and generate 4 sets of results."),
                            ui.tags.li(
                                "We trained predictive models for cell classification using the original data and transformed datasets, and chose the most performant classifiers for the ", ui.strong('aggregator'), " step, which is shown in this app."),
                            ui.tags.li([
                                "The aggregator spatially collates predictions across the whole-slide, producing:",
                                ui.tags.ul(
                                    ui.tags.li(ui.strong("Number and Proportion of Tumour Containing Regions")),
                                    ui.tags.li("Largest Region containing Tumours Cells (", ui.strong('Tumour Mass'), ")"),
                                    ui.tags.li(ui.strong("Heatmaps of Predictions")),
                                )
                            ]),
                            ui.tags.li([
                                "To evaluate the performance of our aggregator, we examined:",
                                ui.tags.ul(
                                    ui.tags.li("Accuracy, F1 Score, Average Errors across quadrants"),
                                    ui.tags.li("Confusion Matrix and Model Outputs for each quadrant")
                                )
                            ]),
                            ui.tags.li("This interactive app presents an effective platform for ", ui.strong('whole slide tumour prediction'), ", improving breast cancer diagnosis and screening, benefitting clinicians and patients.")
                         ),
                     ),
                     class_="with-accent-green"
                 ),
                 col_widths=[6, 6],
                 full_screen=True
             )
        ),

        # 🧠 Models page（Aggregator Evaluation）
        ui.nav_panel("🧠 Aggregator Evaluation",
            ui.accordion(
                ui.accordion_panel(
                    "Performance of all Candidate Models across All Quadrants",
                    ui.layout_columns(
                            ui.card(ui.h2("🎯 Metric Scatter Plots", class_="text-secondary text-center"),
                                    ui.p("Each point on the scatter plots below represents the predicted metric value against the true value, for one quadrant. The dashed lines below show perfect match between predicted and true."),
                                    ui.output_plot("scatter_plot"), full_screen=True, class_="with-accent-teal"),
                            ui.card(ui.h2("🏁 Metric Error Boxplots", class_="text-info text-center"),
                                    ui.p("RMSE, or root mean squared error, shows on average across all 4 quadrants, how wrong the models were on the aggregated metrics."),
                                    ui.output_plot("metric_error_plot"), full_screen=True, class_="with-accent-blue"),
                            ui.card(ui.h2("📦 Prediction Performance Boxplots", class_="text-success text-center"),
                                    ui.p("Accuracy below corresponds to how many of the models predictions were correct, and F1 score prioritises correctness of tumour predictions. Here we discuss predictions across the image (not aggregated metrics above)."),
                                    ui.output_plot("cv_metrics_boxplot"), full_screen=True, class_="with-accent-green"),
                        col_widths=[6, 6]
                    )
                ),
                ui.accordion_panel(
                    "Detailed Performance for Selected Model and Quadrant",
                    ui.layout_sidebar(
                        ui.sidebar(
                            ui.input_select("model_select", "Select Classifier", MODEL_NAMES),
                            ui.input_select("quadrant", "Select Quadrant", ["Q1", "Q2", "Q3", "Q4"]),
                            #ui.input_action_button("run_all_quadrants", "🚀 Run All Quadrants", class_="btn-success", width="100%"),
                            ui.hr(),
                            # ui.h5("📉 Model Confidence", class_="text-muted"),
                            # ui.output_text("error_display"),
                            # ui.hr(),
                            ui.input_action_button("clear_cache_btn", "🧹 Clear Cache", class_="btn-danger")
                        ),
                        ui.layout_columns(
                            # ui.card(ui.h4("🗺️ Heatmap from CV"), ui.output_plot("decision_boundary_plot"), full_screen=True),
                            # ui.card(ui.h4("📈 Performance Metrics"), ui.output_plot("cv_metrics_plot"), full_screen=True),
                            # ui.card(ui.h4("📊 Distributions"), ui.output_plot("cv_distribution_plot"), full_screen=True),
                            # ui.card(ui.h4("🧾 Outer Model Metrics"), ui.output_plot("percent_tumour_grids_plot"), full_screen=True),
                            # ui.card(ui.h4("🔬 Largest Tumour Mass"), ui.output_plot("largest_tumour_mass_plot"), full_screen=True),
                            # ui.card(ui.h4("🎯 Scatter Plot"), ui.output_plot("scatter_plot"), full_screen=True),
                            # ui.card(ui.h4("🏁 Performance"), ui.output_plot("metric_error_plot"), full_screen=True),
                            # ui.card(ui.h4("📦 CV Metric Boxplot"), ui.output_plot("cv_metrics_boxplot"), full_screen=True),

                            ui.card(ui.h2("🗺️ Heatmap from CV", class_="text-info text-center"),
                                    ui.output_plot("decision_boundary_plot"), full_screen=True, class_="with-accent-blue"),
                            ui.card(ui.h2("📈 Confusion Matrix", class_="text-success text-center"),
                                    ui.output_plot("cv_metrics_plot"), full_screen=True, class_="with-accent-green"),
                            ui.card(ui.h2("📊 Distribution of Grids", class_="text-primary text-center"),
                                    ui.output_plot("cv_distribution_plot"), full_screen=True, class_="with-accent-purple"),
                            ui.card(ui.h2("🧾 Percentage of Grids as Tumour", class_="text-warning text-center"),
                                    ui.output_plot("percent_tumour_grids_plot"), full_screen=True, class_="with-accent-orange"),
                            ui.card(ui.h2("🔬 Largest Tumour Mass", class_="text-danger text-center"),
                                    ui.output_plot("largest_tumour_mass_plot"), full_screen=True, class_="with-accent-red"),

                            col_widths=[6, 6]
                        )
                    )
                ),
                open = False
            )
        ),

        # ui.nav_panel("🖼️ Image Test (IN PROGRESS, DO NOT USE)",
        #     ui.div(
        #         ui.input_select("upload_model_select", "Select Classifier", MODEL_NAMES),
        #         ui.input_file("image_upload", "Upload an Image", accept=[".jpg", ".jpeg", ".png"]),
        #         ui.output_plot("upload_heatmap")  # 最简单的结构
        #     )
        # )
        #🖼️ Image Result page（Outer Model）
        ui.nav_panel("🖼️ Whole Slide Prediction",
            ui.layout_sidebar(
                ui.sidebar(
                    ui.input_select("upload_model_select", "Select Classifier", MODEL_NAMES),
                    ui.p("Both classifiers perform comparably - you may consider both of their predictions, for reliability."),
                    ui.input_file("image_upload", "Upload an Image", accept=[".jpg", ".jpeg", ".png", ".tif"]),
                    ui.output_text("show_uploaded_info"),
                    ui.hr(),
                    ui.input_action_button("clear_cache_btn_upload", "🧹 Clear Cache", class_="btn-danger")
                ),
                ui.layout_columns(
                    # ui.card(ui.h4("🧯 Full Image Heatmap"), ui.output_plot("upload_heatmap"), full_screen=True),
                    # ui.card(ui.h4("🧪 Prediction Distribution"), ui.output_plot("upload_distribution"), full_screen=True),
                    # ui.card(ui.h4("🧾 Tumour Cell Percentage"), ui.output_plot("upload_percent_tumour"), full_screen=True),
                    # ui.card(ui.h4("🧾 Largest Tumour Cell Mass"), ui.output_plot("upload_largest_mass"), full_screen=True),

                    ui.card(ui.h2("🌡️ Full Image Heatmap", class_="text-info text-center"),
                            ui.output_plot("upload_heatmap"), full_screen=True, class_="with-accent-blue"),
                    ui.card(ui.h2("📊 Prediction Distribution", class_="text-success text-center"),
                            ui.output_plot("upload_distribution"), full_screen=True, class_="with-accent-green"),
                    ui.card(ui.h2("📉 Tumour Cell Percentage", class_="text-primary text-center"),
                            ui.output_plot("upload_percent_tumour"), full_screen=True, class_="with-accent-orange"),
                    ui.card(ui.h2("🧬 Largest Tumour Cell Mass", class_="text-danger text-center"),
                            ui.output_plot("upload_largest_mass"), full_screen=True, class_="with-accent-red"),

                    col_widths=[6, 6]
                )
            )
        )
    )
)



# ----------------------------
# Back-end logic
# ----------------------------
def server(input, output, session):
    # TO FORCE populate the cache

    for model_name in MODEL_NAMES:
        print(f"🚀 Running all quadrant predictions for model: {model_name}")

        for quadrant in ["Q1", "Q2", "Q3", "Q4"]:
            pred_cache_path = get_prediction_cache_path(model_name, quadrant)
            if os.path.exists(pred_cache_path):
                print(f"✅ Cache exists for {quadrant}, skipping...")
                continue

            print(f"⚙️ Processing {quadrant}...")
            size = (299, 299) if model_name == "InceptionV3" else (224, 224)
            X_ids, X_test, y_test = load_test_data(TEST_DATA_DIR, quadrant, size)

            quadrant_index = QUADRANT_DICT[quadrant]
            if model_name == "VGG19 Train on Centered":
                get_model_name = lambda i: f"VGG19_fold_{i}.h5"
            elif model_name == "VGG19":
                get_model_name = lambda i: f"VGG19_fold_{i}.h5"
            elif model_name == "InceptionV3":
                get_model_name = lambda i: f"inceptionV3_Tanvi_fold_{i}.h5"
            else:
                print("❌ Unknown model name.")
                return

            model_path = os.path.join(MODEL_DIR, get_model_name(quadrant_index))
            if not os.path.exists(model_path):
                print(f"❌ Model file not found: {model_path}")
                continue

            model = load_model(model_path)
            y_prob = model.predict(X_test)
            y_pred = np.argmax(y_prob, axis=1)

            with open(pred_cache_path, 'wb') as f:
                pickle.dump((X_ids, X_test, y_test, y_pred), f)

            print(f"✅ Saved cache for {quadrant}")
            del model
            K.clear_session()
            gc.collect()

        print("🎉 All quadrants processed.")

    run_all_trigger = reactive.Value(0)

    upload_status = reactive.Value("📂 Please upload an image to begin inference.")

    # 🔴 TODO: Inner Model loading (reserve some model positions)
    @reactive.calc
    def get_model():
        name = input.model_select()
        quadrant = QUADRANT_DICT[input.quadrant()]

        model = None
        get_model_name = None

        # 🔴 TODO: Load and return the built-in models (such as CNN, VGG)
        if name == "VGG19 Train on Centered":
            # from tensorflow.keras.applications import VGG16
            # return VGG16(...) or load_model("vgg16_inner.h5")
            get_model_name = lambda i: f"VGG19_fold_{i}.h5"

        elif name == "VGG19":
            get_model_name = lambda i: f"VGG19_fold_{i}.h5"

        elif name == "InceptionV3":
            get_model_name = lambda i: f"inceptionV3_Tanvi_fold_{i}.h5"


        model_path = os.path.join(MODEL_DIR, get_model_name(quadrant))
        if os.path.exists(model_path):
            model = load_model(model_path)
        else:
            print(f"⚠️ Model does not exist: {model_path}")
            return None

        print(f"✅ Loaded model for {name}.")
        return model

    # The simulated dataset generated using sklearn
    @reactive.calc
    def dataset():
        size = (299, 299) if input.model_select() == "InceptionV3" else (224, 224)
        X_ids, X_test, y_test = load_test_data(TEST_DATA_DIR, input.quadrant(), size)
        return X_ids, X_test, y_test

    @reactive.calc
    def model_predictions():
        model_name = input.model_select()
        quadrant_key = input.quadrant()
        cache_key = (model_name, quadrant_key)
        pred_cache_path = get_prediction_cache_path(model_name, quadrant_key)

        # Read from the file cache first
        if os.path.exists(pred_cache_path):
            with open(pred_cache_path, 'rb') as f:
                X_ids, X_test, y_test, y_pred = pickle.load(f)
            print(f"⚡ Loaded predictions from file cache for {cache_key}")
            return X_ids, X_test, y_test, y_pred

        X_ids, X_test, y_test = dataset()
        model = get_model()
        y_prob = model.predict(X_test)
        y_pred = np.argmax(y_prob, axis=1)
        # 写入文件缓存
        with open(pred_cache_path, 'wb') as f:
            pickle.dump((X_ids, X_test, y_test, y_pred), f)
        print(f"✅ Saved predictions to file cache for {cache_key}")

        del model
        K.clear_session()
        gc.collect()

        return X_ids, X_test, y_test, y_pred

    @reactive.calc
    def aggregator():
        X_ids, _, y_test, y_pred = model_predictions()
        y_test_str = [label_names[i] for i in y_test]
        y_pred_str = [label_names[i] for i in y_pred]

        results = {
            'number_tumour_grids_true': [],
            'number_nontumour_grids_true': [],
            'number_empty_grids_true': [],
            'percent_tumour_true': [],
            'largest_mass_true': [],
            # pred val
            'number_tumour_grids_pred': [],
            'number_nontumour_grids_pred': [],
            'number_empty_grids_pred': [],
            'percent_tumour_pred': [],
            'largest_mass_pred': [],
        }

        results['number_tumour_grids_true'].append(y_test_str.count("Tumor"))
        results['number_nontumour_grids_true'].append(y_test_str.count("Non-Tumor"))
        results['number_empty_grids_true'].append(y_test_str.count("Empty"))
        results['percent_tumour_true'].append((y_test_str.count("Tumor")/len(y_test_str))*100)
        results['largest_mass_true'].append(largest_tumour_mass(X_ids, y_test_str))

        results['number_tumour_grids_pred'].append(y_pred_str.count("Tumor"))
        results['number_nontumour_grids_pred'].append(y_pred_str.count("Non-Tumor"))
        results['number_empty_grids_pred'].append(y_pred_str.count("Empty"))
        results['percent_tumour_pred'].append((y_pred_str.count("Tumor")/len(y_pred_str))*100)
        results['largest_mass_pred'].append(largest_tumour_mass(X_ids, y_pred_str))
        print(results)
        return results


    # 🔴 Clear cache logic
    @reactive.effect
    @reactive.event(input.clear_cache_btn)
    def clear_cache():
        import glob
        removed = 0
        for f in glob.glob(os.path.join(CACHE_DIR, '*')):
            try:
                os.remove(f)
                removed += 1
            except Exception as e:
                print(f"Failed to remove {f}: {e}")
        print(f"🧹 Cleared {removed} cache files.")
        session.send_custom_message('cache_cleared', {'removed': removed})

    # @reactive.effect
    # @reactive.event(input.run_all_quadrants)
    # def run_all_quadrants_cache():
    #     model_name = input.model_select()
    #     print(f"🚀 Running all quadrant predictions for model: {model_name}")

    #     for quadrant in ["Q1", "Q2", "Q3", "Q4"]:
    #         pred_cache_path = get_prediction_cache_path(model_name, quadrant)
    #         if os.path.exists(pred_cache_path):
    #             print(f"✅ Cache exists for {quadrant}, skipping...")
    #             continue

    #         print(f"⚙️ Processing {quadrant}...")
    #         size = (299, 299) if model_name == "InceptionV3" else (224, 224)
    #         X_ids, X_test, y_test = load_test_data(TEST_DATA_DIR, quadrant, size)

    #         quadrant_index = QUADRANT_DICT[quadrant]
    #         if model_name == "VGG19 Train on Centered":
    #             get_model_name = lambda i: f"VGG19_fold_{i}.h5"
    #         elif model_name == "VGG19":
    #             get_model_name = lambda i: f"VGG19_fold_{i}.h5"
    #         elif model_name == "InceptionV3":
    #             get_model_name = lambda i: f"inceptionV3_Tanvi_fold_{i}.h5"
    #         else:
    #             print("❌ Unknown model name.")
    #             return

    #         model_path = os.path.join(MODEL_DIR, get_model_name(quadrant_index))
    #         if not os.path.exists(model_path):
    #             print(f"❌ Model file not found: {model_path}")
    #             continue

    #         model = load_model(model_path)
    #         y_prob = model.predict(X_test)
    #         y_pred = np.argmax(y_prob, axis=1)

    #         with open(pred_cache_path, 'wb') as f:
    #             pickle.dump((X_ids, X_test, y_test, y_pred), f)

    #         print(f"✅ Saved cache for {quadrant}")
    #         del model
    #         K.clear_session()
    #         gc.collect()

    #     print("🎉 All quadrants processed.")

    #     # 🔁 Trigger plot refresh
    #     run_all_trigger.set(run_all_trigger.get() + 1)

# Display the cross-validation heat map (decision boundary)
    @output
    @render.plot
    def decision_boundary_plot():
        X_ids, _, y_test, y_pred = model_predictions()
        y_test_str = [label_names[i] for i in y_test]
        y_pred_str = [label_names[i] for i in y_pred]
        model_name = input.model_select()
        quadrant = input.quadrant()
        heatmap_cache_path = get_heatmap_cache_path(model_name, quadrant)
        # Read the heatmap from the file cache first
        if os.path.exists(heatmap_cache_path):
            img = plt.imread(heatmap_cache_path)
            plt.figure(figsize=(15, 15))
            plt.imshow(img)
            plt.axis('off')
            return plt.gcf()
        fig = create_comparison_heatmap(EVALUATION_IMAGE_PATH, box_from_grid_id(X_ids, EVALUATION_IMAGE_PATH, quadrant),
                                  y_test_str, y_pred_str, model_name, quadrant=quadrant, show=False)
        # Save the heatmap to the cache
        plt.savefig(heatmap_cache_path, dpi=150, bbox_inches='tight')
        
        img = plt.imread(heatmap_cache_path)
        plt.figure(figsize=(15, 15))
        plt.imshow(img)
        plt.axis('off')
        return plt.gcf()
    # Display the Accuracy/F1 distribution graph of cross-validation
    @output
    @render.plot
    def cv_metrics_plot():
        X_ids, _, y_test, y_pred = model_predictions()

        cm = confusion_matrix(y_test, y_pred)

        acc = accuracy_score(y_test, y_pred)
        bal_acc = balanced_accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='weighted')
        precision = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')

        plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=True,
                    xticklabels=label_names,
                    yticklabels=label_names)

        plt.xlabel("Predicted Label")
        plt.ylabel("True Label")
        plt.title(f"{input.model_select()} Confusion Matrix")

        metrics_text = (
            f"Accuracy: {acc:.2f}\n"
            f"Balanced Accuracy: {bal_acc:.2f}\n"
            f"F1 Score: {f1:.2f}\n"
            f"Precision: {precision:.2f}\n"
            f"Recall: {recall:.2f}"
        )

        plt.gcf().text(1.05, 0.5, metrics_text, fontsize=10, va='center')

    # Display the randomly generated distribution map (simulation)
    @output
    @render.plot
    def cv_distribution_plot():
        results = aggregator()

        # Prepare long-format data for seaborn
        data = {
            "Grid Type": ["Tumor", "Non-Tumor", "Empty"] * 2,
            "Count": [
                results["number_tumour_grids_true"][0], results['number_nontumour_grids_true'][0], results['number_empty_grids_true'][0],
                results['number_tumour_grids_pred'][0], results['number_nontumour_grids_pred'][0], results['number_empty_grids_pred'][0]
            ],
            "Source": ["True"] * 3 + ["Predicted"] * 3
        }

        df = pd.DataFrame(data)

        plt.figure(figsize=(8, 6))
        sns.barplot(data=df, x="Grid Type", y="Count", hue="Source")
        plt.title(f"Grid Type Distribution - True vs Predicted ({input.model_select()} on {input.quadrant()})")
        plt.ylabel("Grid Count")
        plt.tight_layout()

    # @output
    # @render.plot
    # def number_tumour_grids_plot():
    #     results = aggregator()


    #     data = {
    #         "Source": ["True", "Predicted"],
    #         "Number of Tumour Grids": [
    #             results["number_tumour_true"][0],
    #             results["number_tumour_pred"][0]
    #         ]
    #     }

    #     df = pd.DataFrame(data)

    #     plt.figure(figsize=(6, 5))
    #     sns.barplot(data=df, x="Source", y="Number of Tumour Grids")
    #     plt.title(f"Tumour Grid Number - True vs Predicted ({input.model_select()} on {input.quadrant()})")
    #     plt.ylabel("Number of Grid Regions")
    #     plt.ylim(0, 100)
    #     plt.tight_layout()

    @output
    @render.plot
    def percent_tumour_grids_plot():
        results = aggregator()


        data = {
            "Source": ["True", "Predicted"],
            "Percent Tumour Grids": [
                results["percent_tumour_true"][0],
                results["percent_tumour_pred"][0]
            ]
        }

        df = pd.DataFrame(data)

        plt.figure(figsize=(6, 5))
        sns.barplot(data=df, x="Source", y="Percent Tumour Grids")
        plt.title(f"Tumour Grid Percentage - True vs Predicted ({input.model_select()} on {input.quadrant()})")
        plt.ylabel("Percentage (%)")
        plt.ylim(0, 100)
        plt.tight_layout()

    @output
    @render.plot
    def largest_tumour_mass_plot():
        results = aggregator()

        data = {
            "Source": ["True Largest Mass", "Pred Largest Mass"],
            "Largest Tumour Mass": [
                results["largest_mass_true"][0],
                results["largest_mass_pred"][0]
            ]
        }

        df = pd.DataFrame(data)

        plt.figure(figsize=(6, 5))
        sns.barplot(data=df, x="Source", y="Largest Tumour Mass")
        plt.title(f"Largest Tumour Mass - True vs Predicted ({input.model_select()} on {input.quadrant()})")
        plt.ylabel("Mass Size (Grids)")
        plt.tight_layout()

    @output
    @render.plot
    def scatter_plot():
        # _ = run_all_trigger.get()

        # metrics = []
        # for model_name in MODEL_NAMES:
        #     for quadrant in sorted(QUADRANT_DICT.keys()):
        #         pred_cache_path = get_prediction_cache_path(model_name, quadrant)
        #         if not os.path.exists(pred_cache_path):
        #             continue

        #         with open(pred_cache_path, 'rb') as f:
        #             _, _, y_true, y_pred = pickle.load(f)

        #         metrics.append({
        #             "Model": model_name,
        #             "Quadrant": quadrant,
        #             "Accuracy": accuracy_score(y_true, y_pred),
        #             "F1 Score": f1_score(y_true, y_pred, average="weighted"), 
        #             "Precision": precision_score(y_true, y_pred, average="weighted"),
        #             "Recall": recall_score(y_true, y_pred, average="weighted")
        #         })

        # df = pd.DataFrame(metrics)
        # df_melted = df.melt(id_vars=["Model", "Quadrant"], var_name="Metric", value_name="Score")

        # fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        # sns.boxplot(data=df_melted, x="Metric", y="Score", hue="Model")
        # plt.title("Model Performance Comparison across Quadrants")
        # plt.xticks(rotation=45)
        # plt.legend(title="Model", bbox_to_anchor=(1.05, 1), loc='upper left')
        # plt.tight_layout()


        # Prepare data for scatter plots
        scatter_data = []
        for model_name in MODEL_NAMES:
            for quadrant in sorted(QUADRANT_DICT.keys()):
                pred_cache_path = get_prediction_cache_path(model_name, quadrant)
                if not os.path.exists(pred_cache_path):
                    continue

                with open(pred_cache_path, 'rb') as f:
                    X_ids, _, y_test, y_pred = pickle.load(f)
                
                # Convert to string labels
                y_test_str = [label_names[i] for i in y_test]
                y_pred_str = [label_names[i] for i in y_pred]

                # Calculate metrics
                percent_tumour_true = (y_test_str.count("Tumor") / len(y_test_str)) * 100
                percent_tumour_pred = (y_pred_str.count("Tumor") / len(y_pred_str)) * 100
                
                # Calculate largest mass for true and predicted
                largest_mass_true = largest_tumour_mass(X_ids, y_test_str)
                largest_mass_pred = largest_tumour_mass(X_ids, y_pred_str)

                # Store results
                scatter_data.append({
                    'model': model_name,
                    'quadrant': quadrant,
                    'percent_true': percent_tumour_true,
                    'percent_pred': percent_tumour_pred,
                    'mass_true': largest_mass_true,
                    'mass_pred': largest_mass_pred,
                    'count_true': y_test_str.count("Tumor"),
                    'count_pred': y_pred_str.count("Tumor")
                })

        # Create scatter plots
        fig, axes = plt.subplots(1, 3, figsize=(9, 3))

        # Plot for tumour count
        for model in MODEL_NAMES:
            model_data = [d for d in scatter_data if d['model'] == model]
            axes[2].scatter([d['count_true'] for d in model_data],
                           [d['count_pred'] for d in model_data],
                           label=model, alpha=0.7)

        max_val = max(max(d['count_true'] for d in scatter_data),
                      max(d['count_pred'] for d in scatter_data))
        axes[2].plot([0, max_val], [0, max_val], 'k--', alpha=0.5)
        axes[2].set_xlabel('True Number of Tumour Grids')
        axes[2].set_ylabel('Predicted Number of Tumour Grids')
        axes[2].set_title('Number Tumour Grids')
        axes[2].legend()

        # Plot for percent tumour
        for model in MODEL_NAMES:
            model_data = [d for d in scatter_data if d['model'] == model]
            axes[0].scatter([d['percent_true'] for d in model_data],
                           [d['percent_pred'] for d in model_data],
                           label=model, alpha=0.7)

        max_val = max(max(d['percent_true'] for d in scatter_data),
                      max(d['percent_pred'] for d in scatter_data))
        axes[0].plot([0, max_val], [0, max_val], 'k--', alpha=0.5)
        axes[0].set_xlabel('True Percent Tumour')
        axes[0].set_ylabel('Predicted Percent Tumour')
        axes[0].set_title('Percentage Tumour')
        axes[0].legend()

        # Plot for largest mass
        for model in MODEL_NAMES:
            model_data = [d for d in scatter_data if d['model'] == model]
            axes[1].scatter([d['mass_true'] for d in model_data],
                           [d['mass_pred'] for d in model_data],
                           label=model, alpha=0.7)

        max_val = max(max(d['mass_true'] for d in scatter_data),
                      max(d['mass_pred'] for d in scatter_data))
        axes[1].plot([0, max_val], [0, max_val], 'k--', alpha=0.5)
        axes[1].set_xlabel('True Largest Mass')
        axes[1].set_ylabel('Predicted Largest Mass (Grids)')
        axes[1].set_title('Largest Tumour Mass')
        axes[1].legend()

        plt.tight_layout()
        
        # results = aggregator()

        # fig, axes = plt.subplots(1, 3, figsize=(18, 6))

        # # Plot scatter for percent tumour
        # axes[0].scatter(results['percent_tumour_true'][0], results['percent_tumour_pred'][0],
        #             label='model_name', marker='o', s=80, alpha=0.7)
        # # Add perfect prediction line
        # max_val = max(results['percent_tumour_true'][0], results['percent_tumour_pred'][0])
        # axes[0].plot([0, max_val], [0, max_val], 'k--', alpha=0.5)
        # axes[0].set_xlabel('True Percent Tumour')
        # axes[0].set_ylabel('Predicted Percent Tumour')
        # axes[0].set_title('True vs Predicted Percent Tumour')
        # axes[0].legend()

        # # Plot scatter for largest mass
        # axes[1].scatter(results['largest_mass_true'][0], results['largest_mass_pred'][0],
        #             label='InceptionV3', marker='o', s=80, alpha=0.7)

        # # Add perfect prediction line
        # max_val = max(results['largest_mass_true'][0], results['largest_mass_pred'][0])
        # axes[1].plot([0, max_val], [0, max_val], 'k--', alpha=0.5)
        # axes[1].set_xlabel('True Largest Mass')
        # axes[1].set_ylabel('Predicted Largest Mass')
        # axes[1].set_title('True vs Predicted Largest Tumour Mass')
        # axes[1].legend()

        # # Plot scatter for number of tumour grids
        # axes[2].scatter(results['number_tumour_grids_true'][0], results['number_tumour_grids_pred'][0],
        #             label='InceptionV3', marker='o', s=80, alpha=0.7)
        # # Add perfect prediction line
        # max_val = max(results['number_tumour_grids_true'][0], results['number_tumour_grids_pred'][0])
        # axes[2].plot([0, max_val], [0, max_val], 'k--', alpha=0.5)
        # axes[2].set_xlabel('True Number of Tumour Grids')
        # axes[2].set_ylabel('Predicted Number of Tumour Grids')
        # axes[2].set_title('True vs Predicted Number of Tumour Grids')
        # axes[2].legend()

        # plt.tight_layout()

    @output
    @render.plot
    def metric_error_plot():

        # Get all cached predictions and calculate RMSEs for each model and quadrant
        rmse_data = []
        for model_name in MODEL_NAMES:
            for quadrant in sorted(QUADRANT_DICT.keys()):
                pred_cache_path = get_prediction_cache_path(model_name, quadrant)
                if not os.path.exists(pred_cache_path):
                    continue

                with open(pred_cache_path, 'rb') as f:
                    X_ids, _, y_test, y_pred = pickle.load(f)

                # Convert to string labels
                y_test_str = [label_names[i] for i in y_test]
                y_pred_str = [label_names[i] for i in y_pred]

                # Calculate metrics
                percent_true = (y_test_str.count("Tumor") / len(y_test_str)) * 100
                percent_pred = (y_pred_str.count("Tumor") / len(y_pred_str)) * 100
                
                largest_mass_true = largest_tumour_mass(X_ids, y_test_str)
                largest_mass_pred = largest_tumour_mass(X_ids, y_pred_str)
                
                count_true = y_test_str.count("Tumor")
                count_pred = y_pred_str.count("Tumor")

                # Calculate RMSE
                rmse_percent = np.sqrt((percent_true - percent_pred)**2)
                rmse_mass = np.sqrt((largest_mass_true - largest_mass_pred)**2) 
                rmse_count = np.sqrt((count_true - count_pred)**2)

                rmse_data.append({
                    'Model': model_name,
                    'Quadrant': quadrant,
                    'Metric': 'Percent Tumour',
                    'RMSE': rmse_percent
                })
                rmse_data.append({
                    'Model': model_name, 
                    'Quadrant': quadrant,
                    'Metric': 'Largest Mass',
                    'RMSE': rmse_mass
                })
                rmse_data.append({
                    'Model': model_name,
                    'Quadrant': quadrant, 
                    'Metric': 'Tumour Count',
                    'RMSE': rmse_count
                })

        # Create boxplots
        df = pd.DataFrame(rmse_data)
        fig, axes = plt.subplots(1, 3, figsize=(9, 3))

        for i, metric in enumerate(['Percent Tumour', 'Largest Mass', 'Tumour Count']):
            sns.boxplot(data=df[df['Metric'] == metric], y='RMSE', hue='Model', ax=axes[i])
            axes[i].set_title(f'RMSE for {metric}')
            
            if (metric == 'Percent Tumour'):
                axes[i].set_ylabel(f'RMSE (%)')
            elif (metric == 'Largest Mass'):
                axes[i].set_ylabel(f'RMSE (Grids)')
            elif (metric == 'Tumour Count'):
                axes[i].set_ylabel(f'RMSE (Grids)')
            else:
                axes[i].set_ylabel('RMSE')
            if i > 0:  # Only show legend for first plot
                axes[i].get_legend().remove()

        plt.suptitle('RMSE Comparison Across Models and Metrics')
        plt.tight_layout()
        
        # results = aggregator()  # DataFrame

        # percent_true = results['percent_tumour_true'][0]
        # percent_pred = results['percent_tumour_pred'][0]

        # mass_true = results['largest_mass_true'][0]
        # mass_pred = results['largest_mass_pred'][0]

        # count_true = results['number_tumour_grids_true'][0]
        # count_pred = results['number_tumour_grids_pred'][0]

        # # Calculate RMSE
        # rmse_percent = np.sqrt((percent_true - percent_pred)**2)
        # rmse_mass = np.sqrt((mass_true - mass_pred)**2)
        # rmse_tumour_count = np.sqrt((count_true - count_pred)**2)

        # # Prepare the data
        # data = {
        #     "Metric": ["Percent Tumour", "Largest Mass", "Tumour Grid Count"],
        #     "RMSE": [rmse_percent, rmse_mass, rmse_tumour_count]
        # }
        # df = pd.DataFrame(data)

        # # Draw a picture
        # plt.figure(figsize=(7, 5))
        # sns.barplot(data=df, x="Metric", y="RMSE")
        # plt.title(f"Model RMSE Comparison ({input.model_select()})")
        # plt.ylabel("RMSE")
        # plt.tight_layout()

    @output
    @render.text
    def error_display():
        return "Estimated model error range: ±XX.XX%\n"

    #🔴 TODO: Outer Model loading and prediction (The Patch cutting logic has been completed and is awaiting integration into the model.）

    def get_upload_agg_map(filepath, model_name):
        file_md5 = get_file_md5(filepath)
        pred_cache_path = get_upload_pred_cache_path(model_name, file_md5)
        heatmap_cache_path = get_upload_heatmap_cache_path(model_name, file_md5)

        # 如果聚合和热力图都存在，直接返回
        if os.path.exists(pred_cache_path) and os.path.exists(heatmap_cache_path):
            with open(pred_cache_path, 'rb') as f:
                agg_map = pickle.load(f)
            return agg_map, heatmap_cache_path

        # 否则需要做预测
        model_info = upload_model()
        if model_info is None:
            raise ValueError("Model not loaded.")
        model, preprocess, target_size = model_info

        Image.MAX_IMAGE_PIXELS = None
        big_image = Image.open(filepath)
        width, height = big_image.size
        grids = [
            (i - IMG_SIZE, j - IMG_SIZE, i, j)
            for i in range(width, 0, -IMG_SIZE)
            for j in range(height, 0, -IMG_SIZE)
            if i - IMG_SIZE >= 0 and j - IMG_SIZE >= 0
        ]
        big_image.close()
        del big_image
        gc.collect()

        all_preds = []
        total_batches = (len(grids) + BATCH_SIZE - 1) // BATCH_SIZE
        print(f"Total batches: {total_batches}")

        start_time = time.time()
        for batch_idx, start in enumerate(range(0, len(grids), BATCH_SIZE)):
            end = min(start + BATCH_SIZE, len(grids))
            batch_boxes = grids[start:end]

            big_image = Image.open(filepath)
            batch_images = [
                preprocess(np.array(big_image.crop(box).resize(target_size)))
                for box in batch_boxes
            ]
            big_image.close()
            del big_image
            gc.collect()

            X_batch = np.stack(batch_images).astype("float32")
            y_probs = model.predict(X_batch, batch_size=BATCH_SIZE, verbose=0)
            y_pred_batch = np.argmax(y_probs, axis=1)
            all_preds.extend(y_pred_batch)

            # ✅ ETA tracking and status update
            elapsed = time.time() - start_time
            avg_time = elapsed / (batch_idx + 1)
            eta = avg_time * (total_batches - batch_idx - 1)
            status = f"⏳ Progress: {batch_idx + 1}/{total_batches} batches | ETA: {eta:.1f} sec"
            print(status)

        all_preds_str = [label_names[i] for i in all_preds]
        number_tumour_grids_pred = all_preds_str.count("Tumor")
        number_nontumour_grids_pred = all_preds_str.count("Non-Tumor")
        number_empty_grids_pred = all_preds_str.count("Empty")
        percent_tumour_pred = (number_tumour_grids_pred / len(all_preds_str)) * 100 if all_preds_str else 0
        largest_mass_pred = largest_tumour_mass([f"{(box[0] // IMG_SIZE)}_{(box[1] // IMG_SIZE)}" for box in grids],
                                                all_preds_str)
        agg_map = {
            'number_tumour_grids_pred': number_tumour_grids_pred,
            'number_nontumour_grids_pred': number_nontumour_grids_pred,
            'number_empty_grids_pred': number_empty_grids_pred,
            'percent_tumour_pred': percent_tumour_pred,
            'largest_mass_pred': largest_mass_pred
        }
        with open(pred_cache_path, 'wb') as f:
            pickle.dump(agg_map, f)
        # 预测完顺便生成热力图
        if not os.path.exists(heatmap_cache_path):
            create_heatmap(filepath, grids, all_preds_str, output_path=heatmap_cache_path, show=False)


        return agg_map, heatmap_cache_path

    @reactive.effect
    @reactive.event(input.image_upload)
    def process_uploaded_image():
        if input.image_upload():
            filepath = input.image_upload()[0]['datapath']
            model_name = input.upload_model_select()
            try:
                # 执行缓存检查和推理逻辑
                _ = get_upload_agg_map(filepath, model_name)
                print("✅ Uploaded image processed and cached.")
            except Exception as e:
                print(f"❌ Error processing uploaded image: {e}")

    @output
    @render.plot
    def upload_heatmap():
        if input.image_upload():
            filepath = input.image_upload()[0]['datapath']

        else:
            filepath = SAMPLE_IMAGE_PATH
            
        model_name = input.upload_model_select()
        file_md5 = get_file_md5(filepath)
        agg_map, heatmap_cache_path = get_upload_agg_map(filepath, model_name)

        if os.path.exists(heatmap_cache_path):
            img = plt.imread(heatmap_cache_path)
            fig, ax = plt.subplots(figsize=(15, 15))
            ax.imshow(img)
            ax.axis('off')
            ax.set_title("Prediction Heatmap")
            return plt.gcf()

    @output
    @render.plot
    def upload_distribution():
        if input.image_upload():
            filepath = input.image_upload()[0]['datapath']

        else:
            filepath = SAMPLE_IMAGE_PATH
        
        model_name = input.upload_model_select()
        agg_map, _ = get_upload_agg_map(filepath, model_name)
        print(agg_map)
        data = {
            "Grid Type": ["Tumor", "Non-Tumor", "Empty"],
            "Count": [
                agg_map["number_tumour_grids_pred"],
                agg_map["number_nontumour_grids_pred"],
                agg_map["number_empty_grids_pred"]
            ]
        }
        df = pd.DataFrame(data)
        plt.figure(figsize=(6, 5))
        sns.barplot(data=df, x="Grid Type", y="Count")
        plt.title("Distribution of Predicted Grids (Upload)")
        plt.ylabel("Grid Count")

    @output
    @render.plot
    def upload_percent_tumour():
        if input.image_upload():
            filepath = input.image_upload()[0]['datapath']
        else:
            filepath = SAMPLE_IMAGE_PATH

        model_name = input.upload_model_select()
        agg_map, _ = get_upload_agg_map(filepath, model_name)
        percent = agg_map.get('percent_tumour_pred', 0)
        plt.figure(figsize=(4, 5))
        plt.bar(['Predicted Tumour %'], [percent], color='tomato')
        plt.ylim(0, 100)
        plt.ylabel('Percentage (%)')
        plt.title('Predicted Percentage of Tumour Grids')
        for i, v in enumerate([percent]):
            plt.text(i, v + 2, f"{v:.1f}%", ha='center', va='bottom', fontsize=12)

    @output
    @render.plot
    def upload_largest_mass():
        if input.image_upload():
            filepath = input.image_upload()[0]['datapath']
        else:
            filepath = SAMPLE_IMAGE_PATH
            
        model_name = input.upload_model_select()
        agg_map, _ = get_upload_agg_map(filepath, model_name)
        mass = agg_map.get('largest_mass_pred', 0)
        plt.figure(figsize=(4, 5))
        plt.bar(['Predicted Largest Mass (Number of Grids)'], [mass], color='seagreen')
        plt.ylabel('Mass Size (grids)')
        plt.title('Predicted Largest Tumour Mass')
        for i, v in enumerate([mass]):
            plt.text(i, v + 1, f"{v}", ha='center', va='bottom', fontsize=12)

    # The information will be displayed after uploading the picture
    @output
    @render.text
    def show_uploaded_info():
        if input.image_upload():
            fileinfo = input.image_upload()[0]
            saved_name = fileinfo['name']
            temp_path = fileinfo['datapath']
            size_mb = fileinfo['size'] / (1024 ** 2)
            if not os.path.exists(UPLOAD_SAVE_DIR):
                os.makedirs(UPLOAD_SAVE_DIR)
            final_save_path = os.path.join(UPLOAD_SAVE_DIR, saved_name)
            try:
                shutil.copy(temp_path, final_save_path)
                # Temp Path: {temp_path} \n
                return f"""✅ Uploaded File Info: \n
                                File Name: {saved_name} \n
                                Size: {size_mb:.2f} MB \n 
                                Saved To: {final_save_path}
                            """
            except Exception as e:
                return f"❌ Error saving file: {e}"
        return "📂 Please upload an image to see info and save it."

    @output
    @render.plot
    def cv_metrics_boxplot():
        _ = run_all_trigger.get()

        metrics = []
        for model_name in MODEL_NAMES:
            for quadrant in sorted(QUADRANT_DICT.keys()):
                pred_cache_path = get_prediction_cache_path(model_name, quadrant)
                if not os.path.exists(pred_cache_path):
                    continue

                with open(pred_cache_path, 'rb') as f:
                    _, _, y_true, y_pred = pickle.load(f)

                metrics.append({
                    "Model": model_name,
                    "Quadrant": quadrant,
                    "Accuracy": accuracy_score(y_true, y_pred),
                    "F1 Score": f1_score(y_true, y_pred, average="weighted")#, 
                    #"Precision": precision_score(y_true, y_pred, average="weighted"),
                    #"Recall": recall_score(y_true, y_pred, average="weighted")
                })

        df = pd.DataFrame(metrics)
        df_melted = df.melt(id_vars=["Model", "Quadrant"], var_name="Metric", value_name="Score")

        plt.figure(figsize=(9, 3))
        sns.boxplot(data=df_melted, x="Metric", y="Score", hue="Model")
        plt.title("Model Performance Comparison across Quadrants")
        plt.xticks(rotation=45)
        plt.legend(title="Model", bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        # _ = run_all_trigger.get()
    
        # #model_name = input.model_select()
        # metrics = []
        # for model_name in MODEL_NAMES:
        #     for quadrant in sorted(QUADRANT_DICT.keys()):
        #         pred_cache_path = get_prediction_cache_path(model_name, quadrant)
        #         if not os.path.exists(pred_cache_path):
        #             continue

        #         with open(pred_cache_path, 'rb') as f:
        #             _, _, y_true, y_pred = pickle.load(f)

        #         metrics.append({
        #             "Quadrant": quadrant,
        #             "Accuracy": accuracy_score(y_true, y_pred),
        #             "F1 Score": f1_score(y_true, y_pred, average="weighted"),
        #             "Precision": precision_score(y_true, y_pred, average="weighted"),
        #             "Recall": recall_score(y_true, y_pred, average="weighted")
        #         })

        # df = pd.DataFrame(metrics)
        # df_melted = df.melt(id_vars=["Quadrant"], var_name="Metric", value_name="Score")

        # plt.figure(figsize=(8, 6))
        # sns.boxplot(data=df_melted, x="Metric", y="Score")
        # plt.title(f"{model_name} Performance across Quadrants")
        # plt.tight_layout()

    @reactive.calc
    def upload_model():
        name = input.upload_model_select()
        model = None

        if name == "VGG19":
            get_model_name = lambda i: f"VGG19_fold_{i}.h5"

        elif name == "InceptionV3":
            get_model_name = lambda i: f"inceptionV3_Tanvi_fold_{i}.h5"

        model_path = os.path.join(MODEL_DIR, get_model_name(
            1))  # TODO: use the first fold for now, should be changed to the final model
        if os.path.exists(model_path):
            model = load_model(model_path)
        else:
            print(f"⚠️ Model not exists: {model_path}")
            return None

        print(f"✅ Loaded model for {name}.")
        return model, lambda x: x / 255.0, (224, 224)  # CNN(50x50) Occupy position


app = App(app_ui, server)