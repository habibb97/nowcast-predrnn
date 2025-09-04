# -*- coding: utf-8 -*-
"""
Created on Mon Mar 17 13:43:31 2025

@author: Habib
"""

from ftplib import FTP
from datetime import datetime, timedelta
import os
import re
import sys

# Konfigurasi FTP
FTP_HOST = '202.90.199.64'
FTP_USER = 'metpublik1'
FTP_PASS = '7fX[53W#2Z'
LOCAL_DOWNLOAD_PATH = '/scratch/prefect/habib/input'
INPUT_FOLDER = LOCAL_DOWNLOAD_PATH

# Buat folder input (jika sudah ada, lewati)
try:
    os.makedirs(INPUT_FOLDER)
    print(f"Folder dibuat: {INPUT_FOLDER}")
except FileExistsError:
    print(f"Folder sudah ada: {INPUT_FOLDER}")

# Buat koneksi ke FTP
ftp = FTP(FTP_HOST)
ftp.login(FTP_USER, FTP_PASS)

# Cek koneksi berhasil
print(ftp.getwelcome())

# Ambil tanggal hari ini dan konversi ke path FTP
today = datetime.now().strftime('%Y/%m/%d')
remote_path = f'/himawari6/others/hima_corrected/{today}/'

# Coba masuk ke direktori tanggal saat ini, fallback ke hari sebelumnya jika gagal
try:
    ftp.cwd(remote_path)
except Exception as e:
    print(f'Failed to access directory {remote_path}: {e}')
    # Coba fallback ke direktori kemarin jika hari ini tidak ada
    yesterday = (datetime.now() - timedelta(days=1)).strftime('%Y/%m/%d')
    remote_path = f'/himawari6/others/hima_corrected/{yesterday}/'
    try:
        ftp.cwd(remote_path)
        print(f'Fallback to directory: {remote_path}')
    except Exception as e:
        print(f'Failed to access fallback directory {remote_path}: {e}')
        ftp.quit()
        sys.exit(1)

# Dapatkan daftar file di direktori
try:
    files = ftp.nlst()
except Exception as e:
    print(f'Failed to list files: {e}')
    ftp.quit()
    sys.exit(1)

# Pola regex untuk ekstraksi datetime dari nama file
pattern = re.compile(r'H09_B13_Indonesia_(\d{8})\.Z(\d{4})\.cor\.tiff')

# Filter file yang sesuai dengan pola dan ambil datetime-nya
file_list = []
for file in files:
    match = pattern.match(file)
    if match:
        date_str = match.group(1) + match.group(2)  # Gabungkan tanggal dan waktu
        try:
            date_obj = datetime.strptime(date_str, '%Y%m%d%H%M')
            file_list.append((file, date_obj))
        except ValueError as e:
            print(f"Skipping invalid date format in file {file}: {e}")

# Urutkan file berdasarkan datetime terbaru
file_list.sort(key=lambda x: x[1], reverse=True)

# Ambil 10 file terbaru
latest_files = file_list[:10]

# Jika tidak ada file yang ditemukan, tampilkan pesan
if not latest_files:
    print("No matching files found.")
    ftp.quit()
    sys.exit(0)

# Download file ke folder input
for file, date_obj in latest_files:
    local_file = os.path.join(INPUT_FOLDER, file)
    try:
        with open(local_file, 'wb') as f:
            ftp.retrbinary(f'RETR {file}', f.write)
        print(f'Downloaded: {file}')
    except Exception as e:
        print(f'Failed to download {file}: {e}')

# Tutup koneksi FTP
ftp.quit()
print('Download selesai.')
