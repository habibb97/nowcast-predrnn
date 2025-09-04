import glob
import numpy as np
import rasterio
from rasterio.enums import Resampling

# Ambil semua file dalam direktori
paths = [glob.glob(f"/home/ubuntu/model-cuaca/dataset/{num:02d}/*/*.tiff") for num in range(1, 12, 2)]
paths = [item for sublist in paths for item in sublist[:500]]
paths.sort()

# Periksa apakah ada file yang ditemukan
batch_size = len(paths)
if batch_size == 0:
    raise ValueError("Tidak ada file yang ditemukan di path yang diberikan.")

# Inisialisasi batch array
batch = np.zeros((batch_size, 1, 2350, 2350), dtype=np.float64)

# Zeros untuk padding vertikal
padding_vert = np.zeros((750, 2350), dtype=np.float64)

for i, file in enumerate(paths):
    with rasterio.open(file) as src:
        # Baca data dan resize kalau perlu
        if src.height != 851 or src.width != 2351:
            data = src.read(
                out_shape=(1, 851, 2351),  # 3D shape
                resampling=Resampling.bilinear
            ).astype(np.float64)
        else:
            data = src.read(1).astype(np.float64)
            data = np.expand_dims(data, axis=0)  # supaya bentuk (1, h, w)

        # Preprocessing umum
        data = np.nan_to_num(data)
        data = np.flip(data, axis=1)  # flip vertical
        data = data[:, :-1, :-1]  # potong jadi (1, 850, 2350)
        data = np.where(data > 100, 100, data)

        # padding atas bawah
        data = np.vstack((padding_vert, data[0], padding_vert))  # now shape (2350, 2350)
        data = np.expand_dims(data, axis=0)  # shape (1, 2350, 2350)

        # Simpan ke batch
        batch[i] = data

        print(f'{file} telah disimpan')

# Simpan ke .npy
print(f'Jumlah data {batch_size}, terakhir: {paths[-1]}')
np.save("/home/ubuntu/model-cuaca/train-data/satelit_full_dataset.npy", batch)
