import os.path
import datetime
import numpy as np
from skimage.metrics import structural_similarity as compare_ssim
from core.utils import preprocess
import lpips
import torch
import rasterio

loss_fn_alex = lpips.LPIPS(net='alex')

# Nilai min_global dan max_global (sesuaikan dengan data Anda)
min_global = 0.0  # Contoh nilai minimum global
max_global = 200.0  # Contoh nilai maksimum global

def normalize(data):
    """Normalisasi data menggunakan min_global dan max_global."""
    return (data - min_global) / (max_global - min_global + 1e-8)

def denormalize(data):
    """Denormalisasi data menggunakan min_global dan max_global."""
    return data * (max_global - min_global + 1e-8) + min_global

def compute_psnr(pred, target, data_range):
    """
    Menghitung PSNR antara prediksi dan target.

    Parameters:
    - pred: array numpy, data prediksi
    - target: array numpy, data target asli
    - data_range: float, rentang data maksimum (max - min)

    Returns:
    - psnr_value: float, nilai PSNR dalam desibel (dB)
    """
    mse = np.mean((pred - target) ** 2)
    if mse == 0:
        return float('inf')
    psnr_value = 20 * np.log10(data_range / np.sqrt(mse))
    return psnr_value

def train(model, ims, real_input_flag, configs, itr):
    # Normalisasi input sebelum pelatihan
    ims_normalized = normalize(ims)

    cost = model.train(ims_normalized, real_input_flag)
    if configs.reverse_input:
        ims_rev = np.flip(ims_normalized, axis=1).copy()
        cost += model.train(ims_rev, real_input_flag)
        cost = cost / 2

    if itr % configs.display_interval == 0:
        print(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'), 'itr: ' + str(itr))
        print('training loss: ' + str(cost))

def test(model, test_input_handle, configs, itr):
    print(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'), 'test...')
    test_input_handle.begin(do_shuffle=False)
    res_path = os.path.join(configs.gen_frm_dir, str(itr))
    os.mkdir(res_path)
    avg_mse = 0
    batch_id = 0
    img_mse, ssim, psnr = [], [], []
    lp = []

    for i in range(configs.total_length - configs.input_length):
        img_mse.append(0)
        ssim.append(0)
        psnr.append(0)
        lp.append(0)

    # Reverse schedule sampling
    if configs.reverse_scheduled_sampling == 1:
        mask_input = 1
    else:
        mask_input = configs.input_length

    real_input_flag = np.zeros(
    (configs.batch_size,
     configs.total_length - mask_input - 1,
     configs.img_height // configs.patch_size,
     configs.img_width // configs.patch_size,
     configs.patch_size ** 2 * configs.img_channel))


    if configs.reverse_scheduled_sampling == 1:
        real_input_flag[:, :configs.input_length - 1, :, :] = 1.0

    while not test_input_handle.no_batch_left():
        batch_id += 1
        test_ims = test_input_handle.get_batch()
        # Normalisasi input sebelum inferensi
        test_ims_normalized = normalize(test_ims)

        test_dat = preprocess.reshape_patch(test_ims_normalized, configs.patch_size)
        test_ims = test_ims[:, :, :, :, :configs.img_channel]
        img_gen = model.test(test_dat, real_input_flag)

        img_gen = preprocess.reshape_patch_back(img_gen, configs.patch_size)
        output_length = configs.total_length - configs.input_length
        img_out = img_gen[:, -output_length:]

        # Denormalisasi output model
        img_out_denormalized = denormalize(img_out)

        # Data asli (ground truth) untuk perhitungan metrik
        test_ims_denormalized = test_ims

        # MSE per frame
        for i in range(output_length):
            x = test_ims_denormalized[:, i + configs.input_length, :, :, :]
            gx = img_out_denormalized[:, i, :, :, :]
            mse = np.square(x - gx).sum()
            img_mse[i] += mse
            avg_mse += mse

            # Calculate LPIPS
            img_x = np.zeros([configs.batch_size, 3, configs.img_width, configs.img_width])
            if configs.img_channel == 3:
                img_x[:, 0, :, :] = x[:, :, :, 0]
                img_x[:, 1, :, :] = x[:, :, :, 1]
                img_x[:, 2, :, :] = x[:, :, :, 2]
            else:
                img_x[:, 0, :, :] = x[:, :, :, 0]
                img_x[:, 1, :, :] = x[:, :, :, 0]
                img_x[:, 2, :, :] = x[:, :, :, 0]
            img_x = torch.FloatTensor(img_x)
            img_gx = np.zeros([configs.batch_size, 3, configs.img_width, configs.img_width])
            if configs.img_channel == 3:
                img_gx[:, 0, :, :] = gx[:, :, :, 0]
                img_gx[:, 1, :, :] = gx[:, :, :, 1]
                img_gx[:, 2, :, :] = gx[:, :, :, 2]
            else:
                img_gx[:, 0, :, :] = gx[:, :, :, 0]
                img_gx[:, 1, :, :] = gx[:, :, :, 0]
                img_gx[:, 2, :, :] = gx[:, :, :, 0]
            img_gx = torch.FloatTensor(img_gx)
            lp_loss = loss_fn_alex(img_x, img_gx)
            lp[i] += torch.mean(lp_loss).item()

            real_frm = x  # Data asli dalam skala asli
            pred_frm = gx  # Prediksi model dalam skala asli

            # Compute PSNR per sample in batch
            psnr_batch = []
            for b in range(configs.batch_size):
                pred = pred_frm[b]
                target = real_frm[b]
                psnr_value = compute_psnr(pred, target, data_range=max_global - min_global)
                psnr_batch.append(psnr_value)
            # Rata-rata PSNR untuk batch ini
            psnr_avg = np.mean(psnr_batch)
            psnr[i] += psnr_avg

            # Compute SSIM
            for b in range(configs.batch_size):
                # Squeeze the images to remove any singleton dimensions
                pred_frm_squeezed = np.squeeze(pred_frm[b])
                real_frm_squeezed = np.squeeze(real_frm[b])

                # Set win_size dynamically based on the squeezed image size
                win_size = min(pred_frm_squeezed.shape[0], pred_frm_squeezed.shape[1], 7)  # Ensure the win_size is valid

                # Compare SSIM between the squeezed predicted and real frames
                score = compare_ssim(
                    real_frm_squeezed,
                    pred_frm_squeezed,
                    data_range=max_global - min_global,
                    multichannel=False,
                    win_size=win_size
                )
                ssim[i] += score

        # Save prediction examples as GeoTIFF
        if batch_id <= configs.num_save_samples:
            path = os.path.join(res_path, str(batch_id))
            os.mkdir(path)
            for i in range(configs.total_length):
                name = 'gt' + str(i + 1) + '.tif'  # Save as GeoTIFF
                file_name = os.path.join(path, name)
                img_gt = test_ims_denormalized[0, i, :, :, :].astype(np.float32)  # Data dalam skala asli

                # Rearrange dimensions for rasterio
                if img_gt.ndim == 3 and img_gt.shape[2] == 1:
                    img_gt = img_gt[:, :, 0]
                    img_gt = np.expand_dims(img_gt, axis=0)
                elif img_gt.ndim == 3 and img_gt.shape[2] == 3:
                    img_gt = img_gt.transpose(2, 0, 1)
                else:
                    img_gt = np.expand_dims(img_gt, axis=0)

                with rasterio.open(
                    file_name,
                    'w',
                    driver='GTiff',
                    height=img_gt.shape[1],
                    width=img_gt.shape[2],
                    count=img_gt.shape[0],
                    dtype=img_gt.dtype
                ) as dst:
                    dst.write(img_gt)

            for i in range(output_length):
                name = 'pd' + str(i + 1 + configs.input_length) + '.tif'  # Save as GeoTIFF
                file_name = os.path.join(path, name)
                img_pd = img_out_denormalized[0, i, :, :, :].astype(np.float32)  # Data dalam skala asli

                # Rearrange dimensions for rasterio
                if img_pd.ndim == 3 and img_pd.shape[2] == 1:
                    img_pd = img_pd[:, :, 0]
                    img_pd = np.expand_dims(img_pd, axis=0)
                elif img_pd.ndim == 3 and img_pd.shape[2] == 3:
                    img_pd = img_pd.transpose(2, 0, 1)
                else:
                    img_pd = np.expand_dims(img_pd, axis=0)

                with rasterio.open(
                    file_name,
                    'w',
                    driver='GTiff',
                    height=img_pd.shape[1],
                    width=img_pd.shape[2],
                    count=img_pd.shape[0],
                    dtype=img_pd.dtype
                ) as dst:
                    dst.write(img_pd)

        test_input_handle.next()

    avg_mse = avg_mse / (batch_id * configs.batch_size)
    print('mse per seq: ' + str(avg_mse))
    for i in range(configs.total_length - configs.input_length):
        print(img_mse[i] / (batch_id * configs.batch_size))

    ssim = np.asarray(ssim, dtype=np.float32) / (configs.batch_size * batch_id)
    print('ssim per frame: ' + str(np.mean(ssim)))
    for i in range(configs.total_length - configs.input_length):
        print(ssim[i])

    psnr = np.asarray(psnr, dtype=np.float32) / batch_id
    print('psnr per frame: ' + str(np.mean(psnr)))
    for i in range(configs.total_length - configs.input_length):
        print(psnr[i])

    lp = np.asarray(lp, dtype=np.float32) / batch_id
    print('lpips per frame: ' + str(np.mean(lp)))
    for i in range(configs.total_length - configs.input_length):
        print(lp[i])
