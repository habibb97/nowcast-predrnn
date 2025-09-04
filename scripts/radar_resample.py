import rasterio
from rasterio.enums import Resampling
from rasterio.warp import reproject
import glob
import os
import re
import numpy as np
from datetime import datetime

input_folder = r"E:\predrnn-pytorch\raw\slicing"
output_folder = os.path.join(input_folder, "resampled")
os.makedirs(output_folder, exist_ok=True)


def extract_datetime_from_filename(file_path):
    """
    Ambil timestamp dari nama file:
      - Cari angka 14 digit (YYYYMMDDHHMMSS)
      - Cari angka 12 digit (YYYYMMDDHHMM)
      - Cari angka 8 digit  (YYYYMMDD, jam default 0000)
    Jika tidak ditemukan, fallback ke mtime file.
    """
    fname = os.path.basename(file_path)
    m = re.search(r"(\d{14}|\d{12}|\d{8})", fname)
    if m:
        s = m.group(1)
        if len(s) == 14:
            date_str = s[:8]
            time_str = s[8:12]
        elif len(s) == 12:
            date_str = s[:8]
            time_str = s[8:12]
        else:  # 8 digit
            date_str = s
            time_str = "0000"
        return date_str, time_str
    else:
        # fallback: gunakan waktu modifikasi file
        ts = os.path.getmtime(file_path)
        dt = datetime.fromtimestamp(ts)
        return dt.strftime("%Y%m%d"), dt.strftime("%H%M")


# Parameter target
dst_height = 1008
dst_width = 1008
resampling_method = (
    Resampling.bilinear
)  # ganti ke Resampling.nearest jika data kategori

for file_path in glob.glob(os.path.join(input_folder, "*.tif*")):
    try:
        with rasterio.open(file_path) as src:
            data = src.read(1)
            bounds = src.bounds

            # Hitung transform baru supaya bounds tetap sama
            new_transform = rasterio.transform.from_bounds(
                bounds.left,
                bounds.bottom,
                bounds.right,
                bounds.top,
                dst_width,
                dst_height,
            )

            # Buat array tujuan
            dst_dtype = np.dtype(src.dtypes[0])
            dst_data = np.empty((dst_height, dst_width), dtype=dst_dtype)

            # Reproject / resample
            reproject(
                source=data,
                destination=dst_data,
                src_transform=src.transform,
                src_crs=src.crs,
                dst_transform=new_transform,
                dst_crs=src.crs,
                resampling=resampling_method,
            )

            # Ambil tanggal/waktu dari nama file atau fallback ke mtime
            date_str, time_str = extract_datetime_from_filename(file_path)

            # Format nama output TANPA titik sebelum Z
            new_filename = f"H09_B13_Indonesia_{date_str}Z{time_str}.cor.tiff"
            output_path = os.path.join(output_folder, new_filename)

            # Update profile & simpan (overwrite kalau sudah ada)
            profile = src.profile.copy()
            profile.update(
                {"height": dst_height, "width": dst_width, "transform": new_transform}
            )

            with rasterio.open(output_path, "w", **profile) as dst:
                dst.write(dst_data, 1)

            print(f"Resampled: {file_path} -> {output_path}")

    except Exception as e:
        print(f"ERROR processing {file_path}: {e}")
