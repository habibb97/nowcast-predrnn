# -*- coding: utf-8 -*-
"""
Created on Thu Oct 24 16:14:47 2024

@author: Habib
"""

import os
import sys
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import rasterio

from core.models.model_factory import Model
from core.data_provider import datasets_factory
from core.utils import preprocess

import cartopy.crs as ccrs
import cartopy.feature as cfeature

import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
import geopandas as gpd
from scipy.ndimage import median_filter
from scipy.ndimage import gaussian_filter
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import matplotlib.image as mpimg

import re
from datetime import datetime, timedelta
import warnings

warnings.filterwarnings("ignore")


class Configs:
    def __init__(self):
        self.is_training = 0
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_name = "predrnn"
        self.pretrained_model = "./pretrained/radar.ckpt-80000"
        self.input_folder = os.getenv(
            "INPUT_FOLDER", "./raw/slicing/resampled"
        )  # Ambil dari environment variable
        self.img_width = 1008
        self.img_height = 1008
        self.img_channel = 1
        self.input_length = 10
        self.total_length = 20
        self.num_hidden = "128,128,128,128"
        self.filter_size = 3
        self.stride = 1
        self.layer_norm = 0
        self.patch_size = 8
        self.batch_size = 1
        self.reverse_input = 1
        self.scheduled_sampling = 1
        self.reverse_scheduled_sampling = 0
        self.sampling_stop_iter = 100
        self.sampling_start_value = 1.0
        self.sampling_changing_rate = 0.00002
        self.display_interval = 1
        self.test_interval = 1
        self.snapshot_interval = 1
        self.num_save_samples = 1
        self.save_dir = "./checkpoints"
        self.gen_frm_dir = os.getenv(
            "OUTPUT_FOLDER", "./results"
        )  # Ambil dari environment variable
        self.save_output = 1
        self.lr = 0.0003
        self.beta1 = 0.9


def load_geotiff_images(folder_path):
    images = []
    image_paths = []

    # Get all files with .tif or .tiff extension in the folder
    for file_name in os.listdir(folder_path):
        if file_name.lower().endswith((".tif", ".tiff")):
            image_paths.append(os.path.join(folder_path, file_name))

    # Sort files to ensure consistent order
    image_paths.sort()

    # Read each GeoTIFF file
    for img_path in image_paths:
        print(f"Reading file: {img_path}")
        with rasterio.open(img_path) as dataset:
            img_array = dataset.read(1)  # Read the first band
            img_array[img_array > 200] = 200
            img_array[img_array < 1] = 0
            images.append(img_array)

    images = np.stack(images, axis=0)  # Shape: [sequence_length, img_height, img_width]
    return images


def min_max_normalize(images, min_value=0, max_value=100):
    images = np.nan_to_num(images)
    images = images / 100

    return images


def min_max_denormalize(images, min_value=0, max_value=200):
    images = images * 100
    return images


def preprocess_geotiff_images(images, configs):
    # If img_channel = 1, add channel dimension
    if configs.img_channel == 1:
        images = images[
            ..., np.newaxis
        ]  # Shape: [sequence_length, img_height, img_width, 1]
    elif configs.img_channel > 1:
        pass  # Adjust if using more than 1 channel

    return images.astype(np.float32)


def save_geotiff(data_array, output_path, reference_image):
    with rasterio.open(reference_image) as src:
        profile = src.profile
        # Update profile to ensure compatibility
        profile.update(dtype=rasterio.float32, count=1, nodata=0)
        with rasterio.open(output_path, "w", **profile) as dst:
            dst.write(data_array.astype(rasterio.float32), 1)


def plot(
    data,
    min_lon,
    max_lon,
    min_lat,
    max_lat,
    lon_grid,
    lat_grid,
    configs,
    kabupaten,
    propinsi,
    logo_path,
    legend_patches,
    datetime_str,
    cmap,
    norm,
    delta_time,  # Waktu prediksi yang benar
):
    # Create figure and axis (ukuran lebih kecil)
    fig, ax = plt.subplots(
        figsize=(8, 6), subplot_kw={"projection": ccrs.PlateCarree()}
    )
    ax.set_extent([min_lon, max_lon, min_lat, max_lat])

    # # Add features
    # ax.add_feature(cfeature.BORDERS, linewidth=0.5)
    # ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
    # ax.add_feature(cfeature.LAND, facecolor="lightgray")
    # ax.add_feature(cfeature.OCEAN, facecolor="lightblue")

    # Baca file peta statis (misal screenshot OSM)
    peta_img = plt.imread(r"E:\predrnn-pytorch\shp_lite\peta_osm.png")

    # Tampilkan peta di bawah data
    ax.imshow(
        peta_img,
        extent=[min_lon, max_lon, min_lat, max_lat],
        transform=ccrs.PlateCarree(),
        origin="upper",
    )

    # Plot data raster
    mesh = ax.pcolormesh(
        lon_grid,
        lat_grid,
        data * 200,
        transform=ccrs.PlateCarree(),
        cmap=cmap,
        norm=norm,
        shading="auto",
        alpha=0.7,
    )
    # # Plot shapefiles
    # kabupaten.plot(ax=ax, edgecolor="black", linewidth=0.5, facecolor="none")
    # propinsi.plot(ax=ax, edgecolor="black", linewidth=0.8, facecolor="none")

    # Add text
    ax.text(
        min_lon + 0.1,
        max_lat - 0.1,
        "Prediksi Curah Hujan",
        fontsize=7,
        fontweight="bold",
    )
    ax.text(min_lon, min_lat - 0.1, "Model: PRED-RNN", fontsize=7)
    ax.text(
        min_lon,
        min_lat - 0.2,
        "Sumber data: Tim analisis radar satelit BMKG",
        fontsize=7,
    )
    ax.text(
        max_lon, min_lat - 0.1, f"Inisial: { datetime_str} UTC", fontsize=7, ha="right"
    )
    ax.text(
        max_lon, min_lat - 0.2, f"Prediksi: {delta_time} UTC", fontsize=7, ha="right"
    )

    # Tambahkan legenda
    legend = ax.legend(
        handles=legend_patches,
        title="Legenda",
        loc="lower left",
        fontsize=6,
        borderpad=1,
    )
    ax.add_artist(legend)

    # Colorbar
    cbar = plt.colorbar(mappable=mesh, ax=ax, shrink=0.8, pad=0.03, alpha=1)
    cbar.ax.tick_params(labelsize=5)
    cbar.set_label("mm")

    # Tambahkan logo di pojok kiri atas
    logo = mpimg.imread(logo_path)
    imagebox = OffsetImage(logo, zoom=0.05)  # Sesuaikan zoom untuk ukuran kecil
    ab = AnnotationBbox(
        imagebox, (max_lon - 0.3, max_lat - 0.3), frameon=False, transform=ax.transData
    )
    ax.add_artist(ab)

    # Add title
    # plt.title(f"Predicted Frame {iter}")

    output_path = os.path.join(configs.gen_frm_dir, datetime_str)
    os.makedirs(output_path, exist_ok=True)

    # Save dan tampilkan plot
    plot_output_path = os.path.join(output_path, f"jabar_ai_{delta_time}.png")

    plt.savefig(plot_output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main():
    # Get configurations
    configs = Configs()

    kabupaten = gpd.read_file("./shp_lite/Kabupaten.shp")
    propinsi = gpd.read_file("./shp_lite/Propinsi.shp")
    logo_path = "./shp_lite/logo_BMKG.png"

    # Create result directory if it doesn't exist
    if not os.path.exists(configs.gen_frm_dir):
        os.makedirs(configs.gen_frm_dir)

    # Move model to the appropriate device
    device = torch.device(configs.device)

    # Read and process GeoTIFF data
    input_folder = configs.input_folder  # Replace with your input folder path
    images = load_geotiff_images(input_folder)

    # Ensure enough images
    input_length = configs.input_length  # 10
    total_length = configs.total_length  # 30
    if images.shape[0] < input_length:
        raise ValueError(
            f"Not enough images. Found {images.shape[0]}, required {input_length}"
        )

    # Use the first 'input_length' images as input
    input_images = images[:input_length]

    # Ensure image dimensions
    img_height, img_width = input_images.shape[1], input_images.shape[2]
    configs.img_width = img_width
    configs.img_height = img_height

    # Initialize model after setting image dimensions
    model = Model(configs)
    model.load(configs.pretrained_model)
    model.network.eval()
    model.network.to(device)

    # Normalize images with Min-Max Scaler
    min_value = 0
    max_value = 100
    input_images = min_max_normalize(
        input_images, min_value=min_value, max_value=max_value
    )

    # Process images to match model input shape
    input_images = preprocess_geotiff_images(
        input_images, configs
    )  # Shape: [input_length, img_height, img_width, img_channel]

    # Create frames_tensor with shape [batch_size, total_length, img_height, img_width, img_channel]
    frames_tensor = np.zeros(
        (
            configs.batch_size,
            configs.total_length,
            img_height,
            img_width,
            configs.img_channel,
        ),
        dtype=np.float32,
    )

    # Copy input images into frames_tensor
    frames_tensor[0, :input_length] = input_images

    # Optionally, replicate the last input frame to fill the rest
    for t in range(input_length, total_length):
        frames_tensor[0, t] = input_images[-1]

    # Preprocessing data (reshape into patches)
    frames_tensor = preprocess.reshape_patch(frames_tensor, configs.patch_size)
    frames_tensor = torch.from_numpy(frames_tensor).float().to(device)  # Ensure float32

    # Create real_input_flag
    real_input_flag = np.zeros(
        (
            configs.batch_size,
            configs.total_length - configs.input_length,
            configs.img_height // configs.patch_size,
            configs.img_width // configs.patch_size,
            configs.patch_size**2 * configs.img_channel,
        )
    ).astype(np.float32)
    real_input_flag[:, : configs.input_length - 1, :, :] = 1.0
    real_input_flag = torch.from_numpy(real_input_flag).float().to(device)

    # Perform prediction
    with torch.no_grad():
        output = model.network(frames_tensor, real_input_flag)
        output_data = output[0]  # Extract next_frames from the tuple
        output_data = output_data.cpu().numpy()

    # Postprocessing data
    output_data = preprocess.reshape_patch_back(output_data, configs.patch_size)

    # Get predicted frames (after input_length)
    pred_frames = output_data[:, configs.input_length - 1 :, :, :, :]
    pred_frames[pred_frames <= 0.001] = np.nan

    # Latitude and Longitude bounds
    raster_min_lat, raster_max_lat = -7, -5
    raster_min_lon, raster_max_lon = 105, 108
    min_lon, max_lon = 105, 108
    min_lat, max_lat = -7, -5

    # Create latitude and longitude arrays
    latitudes = np.linspace(
        raster_max_lat, raster_min_lat, img_height
    )  # Ensure correct order
    longitudes = np.linspace(raster_min_lon, raster_max_lon, img_width)

    # Create coordinate meshgrid
    lon_grid, lat_grid = np.meshgrid(longitudes, latitudes)
    reference_image = os.path.join(input_folder, os.listdir(input_folder)[0])

    last_date = os.listdir(input_folder)[-1]

    # Hilangkan ekstensi supaya regex lebih aman
    base_name = os.path.splitext(last_date)[0]

    # Cari pola tanggal dan jam (YYYYMMDD.ZHHMM)
    pattern = r"(\d{8})Z?(\d{4})"

    match = re.search(pattern, base_name)

    if match:
        # Ambil tanggal dan waktu dari hasil match
        date_string = match.group(1)
        time_string = match.group(2)
        datetime_str = date_string + time_string
        datetime_obj = datetime.strptime(datetime_str, "%Y%m%d%H%M")
    else:
        raise ValueError("Tanggal atau jam dan menit tidak ditemukan dalam filename")
    # Gabungkan tanggal dan waktu menjadi format datetime

    # Create legend
    # Define colormap
    levels = [1.5, 2, 5, 10, 15, 20, 25, 30, 50, 100]
    # list(range(0, 110, 10))
    colors = [
        # "#BDF2BA",
        "#B2F2A4",
        # "#88F487",
        "#68F422",
        "#A4EE1B",
        "#F2F220",
        # "#EFD216",
        "#EBA91C",
        "#ED8E1D",
        "#EA661F",
        "#EE251E",
        "#E719B5",
    ]
    levels_reduced = [
        "Hujan ringan",
        "Hujan sedang",
        "Hujan lebat",
        "Hujan sangat Lebat",
        "Hujan ekstrim",
    ]
    colors_reduced = ["#A4EE1B", "#F2F220", "#EBA91C", "#EE251E", "#E719B5"]
    legend_patches = [
        mpatches.Patch(color=color, label=level)
        for color, level in zip(colors_reduced, levels_reduced)
    ]
    #         label=f"{levels_reduced[i]} - {levels_reduced[i+1]} mm",
    #     )
    #     for i in range(len(levels_reduced) - 1)
    # ]
    cmap = mcolors.ListedColormap(colors[: len(levels) - 1])
    cmap.set_bad(color="none")  # Warna untuk NaN
    norm = mcolors.BoundaryNorm(levels, cmap.N)

    # Save predicted frames as GeoTIFF and plot
    for i in range(pred_frames.shape[1]):

        delta_time = datetime_obj + timedelta(minutes=(i + 1) * 10)
        delta_time = delta_time.strftime("%Y-%m-%d %H:%M:%S")
        delta_time = delta_time.replace(":", "").replace(" ", "").replace("-", "")
        frame = pred_frames[0, i, :, :, 0]  # Get first batch and first channel

        # Ensure values are within [0, 1]
        frame = np.clip(frame, 0, 1)

        # Save frame as GeoTIFF

        output_path = os.path.join(configs.gen_frm_dir, datetime_str)
        os.makedirs(output_path, exist_ok=True)

        # Use one of the input files as reference

        save_geotiff(
            frame,
            os.path.join(output_path, f"predicted_frame_{delta_time}.tiff"),
            reference_image=reference_image,
        )

        # Prepare data for plotting
        data = frame  # Data is already in [0, 1]
        data = median_filter(data, size=5)
        data = gaussian_filter(data, sigma=1)
        plot(
            data,
            min_lon,
            max_lon,
            min_lat,
            max_lat,
            lon_grid,
            lat_grid,
            configs,
            kabupaten,
            propinsi,
            logo_path,
            legend_patches,
            datetime_str,
            cmap,
            norm,
            delta_time,  # Waktu prediksi yang benar
        )

        print(f"Prediksi {delta_time}. disimpan di folder:", output_path)

    acc_frame30 = (
        pred_frames[0, 0, :, :, 0]
        + (pred_frames[0, 1, :, :, 0] - pred_frames[0, 0, :, :, 0])
        + (pred_frames[0, 2, :, :, 0] - pred_frames[0, 1, :, :, 0])
        # + pred_frames[0, 3, :, :, 0]
        # + pred_frames[0, 4, :, :, 0]
        # + pred_frames[0, 5, :, :, 0]
    )
    acc_frame30 = np.clip(acc_frame30, 0, 1)
    # acc_frame30 = gaussian_filter(acc_frame30, sigma=1.2)
    output_path30 = os.path.join(output_path, "accumulation_30.tiff")
    save_geotiff(acc_frame30, output_path30, reference_image=reference_image)
    delta_time = datetime_obj + timedelta(minutes=30)
    delta_time = delta_time.strftime("%Y-%m-%d %H:%M:%S")
    delta_time = delta_time.replace(":", "").replace(" ", "").replace("-", "")

    plot(
        acc_frame30,
        min_lon,
        max_lon,
        min_lat,
        max_lat,
        lon_grid,
        lat_grid,
        configs,
        kabupaten,
        propinsi,
        logo_path,
        legend_patches,
        datetime_str,
        cmap,
        norm,
        f"Akumulasi 30 menit {delta_time}",
    )
    print(f"Prediksi Akumulasi 30 menit. disimpan di folder:", output_path30)

    acc_frame60 = (
        pred_frames[0, 2, :, :, 0]
        + (pred_frames[0, 3, :, :, 0] - pred_frames[0, 2, :, :, 0])
        + (pred_frames[0, 4, :, :, 0] - pred_frames[0, 3, :, :, 0])
        + (pred_frames[0, 5, :, :, 0] - pred_frames[0, 4, :, :, 0])
    )
    acc_frame60 = np.clip(acc_frame60, 0, 1)
    # acc_frame30 = gaussian_filter(acc_frame30, sigma=1.2)
    output_path60 = os.path.join(output_path, "accumulation_60.tiff")
    save_geotiff(acc_frame60, output_path60, reference_image=reference_image)
    delta_time = datetime_obj + timedelta(minutes=60)
    delta_time = delta_time.strftime("%Y-%m-%d %H:%M:%S")
    delta_time = delta_time.replace(":", "").replace(" ", "").replace("-", "")

    plot(
        acc_frame60,
        min_lon,
        max_lon,
        min_lat,
        max_lat,
        lon_grid,
        lat_grid,
        configs,
        kabupaten,
        propinsi,
        logo_path,
        legend_patches,
        datetime_str,
        cmap,
        norm,
        f"Akumulasi 1 Jam {delta_time}",
    )
    print(f"Prediksi Akumulasi 1 Jam. disimpan di folder:", output_path60)
    return pred_frames, input_images


if __name__ == "__main__":
    output, input_data = main()
