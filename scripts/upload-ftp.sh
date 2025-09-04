#!/bin/bash

# Aktifkan virtual environment

# Cari folder terakhir yang dibuat di ~/model-cuaca/results/
LATEST_FOLDER=$(ls -td ~/model-cuaca/results/*/ | head -1)

# Cek apakah folder ditemukan
if [ -d "$LATEST_FOLDER" ]; then
    # Ambil nama folder terakhir (tanpa path)
    #FOLDER_NAME=$(basename "$LATEST_FOLDER")

    echo "Uploading $LATEST_FOLDER to FTP..."

    # Gunakan lftp untuk upload
    lftp -u "user@datacuaca,user@datacuaca2024" 202.90.199.252 <<EOF
    set ssl:verify-certificate no
    mirror -R "$LATEST_FOLDER" "/cuaca/jabar/img/"
    bye
EOF

    echo "Berhasil upload ke /cuaca/jabar/img/ !"
else
    echo "No folder found to upload."
fi
