import os
import numpy as np
import rasterio
from rasterio.warp import calculate_default_transform, reproject, Resampling
from rasterio.transform import from_origin

# Fungsi untuk melakukan cropping dan pemrosesan data GeoTIFF
def process_geotiff_files(tiff_folder, output_folder, min_lat, max_lat, min_lon, max_lon):
    # Loop untuk mencari file-file GeoTIFF
    for root, dirs, files in os.walk(tiff_folder):
        print(f"Searching in: {root}")  # Debug: Menampilkan folder yang sedang diproses
        for file in files:
            if file.endswith('.tiff') or file.endswith('.tif'):
                tiff_file = os.path.join(root, file)
                print(f"Found file: {tiff_file}")  # Debug: Menampilkan file TIFF yang ditemukan
                
                try:
                    with rasterio.open(tiff_file) as dataset:
                        # Membaca data raster (band pertama)
                        data = dataset.read(1)
                        transform = dataset.transform
                        
                        # Mendapatkan grid latitude dan longitude dari data raster
                        lon = np.linspace(transform[2], transform[2] + transform[0] * dataset.width, dataset.width)
                        lat = np.linspace(transform[5], transform[5] + transform[4] * dataset.height, dataset.height)
                        
                        # Memastikan grid lat/lon menurun untuk latitude (sesuai dengan sistem koordinat GeoTIFF)
                        if lat[0] < lat[-1]:
                            lat = lat[::-1]
                            data = np.flipud(data)  # Membalik data jika grid lat terbalik
                        
                        # Mask untuk memfilter data berdasarkan batas lat/lon yang diberikan
                        mask_lat = (lat >= min_lat) & (lat <= max_lat)
                        mask_lon = (lon >= min_lon) & (lon <= max_lon)
                        
                        # Memotong data sesuai dengan mask lat/lon
                        if not mask_lat.any() or not mask_lon.any():
                            print(f"No data in bounds for {tiff_file}. Skipping.")
                            continue
                            
                        data_cropped = data[np.ix_(mask_lat, mask_lon)]
                        
                        # Mendapatkan transform baru untuk data yang telah dipotong
                        new_transform = from_origin(transform[2] + transform[0] * mask_lon.argmin(), 
                                                     transform[5] + transform[4] * mask_lat.max(), 
                                                     transform[0], transform[4])
                        
                        # Menyimpan data yang telah dipotong sebagai file GeoTIFF
                        output_file = os.path.join(output_folder, file)
                        print(f"Saving cropped data to: {output_file}")  # Debug: Menampilkan lokasi penyimpanan
                        
                        with rasterio.open(
                                output_file,
                                'w',
                                driver='GTiff',
                                height=data_cropped.shape[0],
                                width=data_cropped.shape[1],
                                count=1,
                                dtype=data_cropped.dtype,
                                crs=dataset.crs,
                                transform=new_transform) as dst:
                            dst.write(data_cropped, 1)

                except Exception as e:
                    print(f"Error reading file {tiff_file}: {e}")

# Pengaturan folder input dan parameter lat/lon yang diinginkan
tiff_folder = r'E:\coorection\12'
output_folder = r'E:\coorection\12_satelit'
# min_lat, max_lat = -9, -4
# min_lon, max_lon = 103, 110

min_lon, min_lat, max_lon, max_lat = 104.37442251240476, -8.420261421103787, 108.92083624434126, -3.918950999999998
# Memproses file GeoTIFF
process_geotiff_files(tiff_folder, output_folder, min_lat, max_lat, min_lon, max_lon)
