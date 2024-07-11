import os
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter
from scipy.stats import pearsonr
import pickle as pkl

home_directory = os.path.expanduser('~') + '/'
Ophthal_dir = home_directory + 'Ophthal/'

patient_id = 'f840f8f7cce203f8772bee75a0d91750646bd8c82dcd207b4fde48647284dee4/'
visit_hash = '66eef770603e053058bf8b52fedf6a3a22db63ad2d5eeb68a3095c931f476f1e/'
oct_imgs_path = Ophthal_dir + patient_id + '/macOCT/' + visit_hash
out_dir = home_directory + 'retFig/save_figs/'

def load_oct_image(oct_imgs_path, idx_start=0, idx_end=61):
    oct_imgs = []
    for i in range(idx_start, idx_end):
        img_path = oct_imgs_path + f'oct-{i:03d}.png'
        oct_img = Image.open(img_path)
        oct_imgs.append(oct_img)
    return oct_imgs

def mse(imageA, imageB):
    err = np.sum((np.array(imageA).astype("float") - np.array(imageB).astype("float")) ** 2)
    err /= float(np.prod(np.array(imageA).shape))
    err = np.sqrt(err)
    return err

def ssim(imageA, imageB, K1=0.01, K2=0.03, win_size=7, sigma=1.5):
    """
    Compute the SSIM between two images.
    
    Args:
        imageA (numpy array): The first image.
        imageB (numpy array): The second image.
        K1 (float): Constant to stabilize the division with weak denominator.
        K2 (float): Constant to stabilize the division with weak denominator.
        win_size (int): The size of the Gaussian filter window.
        sigma (float): The standard deviation of the Gaussian filter.
    
    Returns:
        float: The SSIM index between imageA and imageB.
    """
    
    mu1 = gaussian_filter(imageA, sigma=sigma)
    mu2 = gaussian_filter(imageB, sigma=sigma)
    
    # Compute variances and covariance
    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2
    
    sigma1_sq = gaussian_filter(imageA ** 2, sigma=sigma) - mu1_sq
    sigma2_sq = gaussian_filter(imageB ** 2, sigma=sigma) - mu2_sq
    sigma12 = gaussian_filter(imageA * imageB, sigma=sigma) - mu1_mu2
    
    # Constants for SSIM
    C1 = (K1 * 255) ** 2
    C2 = (K2 * 255) ** 2
    
    # Compute SSIM
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    
    return ssim_map.mean()


oct_imgs = load_oct_image(oct_imgs_path)
oct_imgs = [np.array(oct_img).astype("float") for oct_img in oct_imgs]
print(mse(oct_imgs[0], oct_imgs[1]))
print(ssim(oct_imgs[0], oct_imgs[1]))

# mse_pairwise = [[] for i in range(len(oct_imgs))]
# for i in range(len(oct_imgs)):
#     for j in range(len(oct_imgs)):
#         mse_pairwise[i].append(mse(oct_imgs[i], oct_imgs[j]))



# Precompute shape and product of shape
shape = np.array(oct_imgs[0]).shape
prod_shape = float(np.prod(shape))

# Vectorized pairwise MSE computation
oct_imgs_stack = np.stack(oct_imgs)  # Shape: (num_images, height, width)
mse_pairwise = np.zeros((len(oct_imgs), len(oct_imgs)))
ssim_pairwise = np.zeros((len(oct_imgs), len(oct_imgs)))
for i in range(len(oct_imgs)):
    diffs = oct_imgs_stack - oct_imgs[i]  # Shape: (num_images, height, width)
    squared_diffs = np.square(diffs)  # Shape: (num_images, height, width)
    sum_squared_diffs = np.sum(squared_diffs, axis=(1, 2))  # Shape: (num_images,)
    mse_pairwise[i] = np.sqrt(sum_squared_diffs / prod_shape)
# Calculate absolute frame index distance and corresponding RMSE values
frame_distances = []
rmse_values = []

for i in range(len(oct_imgs)):
    for j in range(len(oct_imgs)):
        if i == j:
            continue
        frame_distances.append(abs(i - j))
        rmse_values.append(mse_pairwise[i, j])


# Pearson correlation coefficient
# Calculate Pearson correlation coefficient
pearson_corr, _ = pearsonr(frame_distances, rmse_values)
# Perform linear regression
slope, intercept = np.polyfit(frame_distances, rmse_values, 1)
line_x = np.array(frame_distances)
line_y = slope * line_x + intercept
plt.plot(line_x, line_y, color='green', label=f'Fit line: y={slope:.2f}x+{intercept:.2f}', alpha=0.7, linestyle='--')

# Create scatter plot with color based on RMSE values
sc = plt.scatter(frame_distances, rmse_values, c=rmse_values, cmap='Blues', alpha=0.7)
plt.title(f'Frame similarity across OCT volume (Pearson r = {pearson_corr:.3f})', fontsize=10)

plt.xlabel('Absolute frame index distance', fontsize=10)
plt.ylabel('Rooted mean square error (RMSE)', fontsize=10)

# plt.imshow(mse_pairwise, cmap='Blues', vmin=12)
# plt.title('Frame similarity across OCT volume', fontsize=10)
# plt.xlabel('Frame index', fontsize=10)
# plt.ylabel('Frame index', fontsize=10)

cbar = plt.colorbar(orientation='vertical')
# Add title for colorbar
cbar.ax.set_xlabel('RMSE', fontsize=10)
cbar.ax.xaxis.set_label_position('top')
cbar.ax.xaxis.set_ticks_position('top')
plt.tight_layout()
plt.savefig(out_dir + 'mse_pairwise_scatter.png')
plt.savefig(out_dir + 'mse_pairwise_scatter.pdf', dpi=300)

if os.path.exists(out_dir + 'ssim_pairwise_scatter.pkl'):
    with open(out_dir + 'ssim_pairwise_scatter.pkl', 'rb') as f:
        ssim_pairwise = pkl.load(f)
        ssim_values, frame_distances = ssim_pairwise['ssim_values'], ssim_pairwise['frame_distances']

else:

    import time
    start = time.time()
    for i in range(len(oct_imgs)):
        
        for j in range(len(oct_imgs)):
            ssim_pairwise[i, j] = ssim(oct_imgs[i], oct_imgs[j])
        print('i', i, time.time() - start)

    # Calculate absolute frame index distance and corresponding SSIM values
    ssim_values = []

    for i in range(len(oct_imgs)):
        for j in range(len(oct_imgs)):
            if i == j:
                continue
            ssim_values.append(ssim_pairwise[i, j])
    with open(out_dir + 'ssim_pairwise_scatter.pkl', 'wb') as f:
        pkl.dump({'ssim_values': ssim_values, 'frame_distances': frame_distances}, f)

# Pearson correlation coefficient for SSIM
pearson_corr_ssim, _ = pearsonr(frame_distances, ssim_values)

# Perform linear regression for SSIM
slope_ssim, intercept_ssim = np.polyfit(frame_distances, ssim_values, 1)
line_y_ssim = slope_ssim * line_x + intercept_ssim

# Create scatter plot with color based on SSIM values
plt.clf()
sc_ssim = plt.scatter(frame_distances, ssim_values, c=ssim_values, cmap='Blues', alpha=0.99)
plt.plot(line_x, line_y_ssim, color='orange', label=f'Fit line: y={slope_ssim:.2f}x+{intercept_ssim:.2f}', alpha=0.7, linestyle='--')
plt.title(f'Frame similarity across OCT volume (Pearson r = {pearson_corr_ssim:.3f})', fontsize=10)
plt.xlabel('Absolute frame index distance', fontsize=10)
plt.ylabel('Structural similarity index (SSIM)', fontsize=10)
# plt.legend()
cbar_ssim = plt.colorbar(sc_ssim, orientation='vertical')
cbar_ssim.ax.set_xlabel('SSIM', fontsize=10)
cbar_ssim.ax.xaxis.set_label_position('top')
cbar_ssim.ax.xaxis.set_ticks_position('top')

plt.tight_layout()
plt.savefig(out_dir + 'ssim_pairwise_scatter.png')
plt.savefig(out_dir + 'ssim_pairwise_scatter.pdf', dpi=300)