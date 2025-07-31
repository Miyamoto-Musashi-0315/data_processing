from math import e
import os
import h5py
import numpy as np
import cv2
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from skimage.metrics import structural_similarity as ssim

def load_h5_images(h5_path):
    """Load images from .h5 file."""
    with h5py.File(h5_path, 'r') as f:
        images = f['data'][()] #[:]  # Shape: (24, 3, 2280, 3420)

    if images.ndim == 4:
        return images.transpose(0, 2, 3, 1)
    else:
        return images


def load_camera_params(npz_path):
    """Load stereo camera parameters from .npz file."""
    data = np.load(npz_path)
    params = {        
        'P1': data['P1'], 'P2': data['P2'],
        'baseline': data['baseline'], 'fB': data['fB']
    }

    params['K_new'] = params['P1'][:, :3]
    params['K_inv'] = np.linalg.inv(params['K_new'])
    params['T'] = np.array([params['baseline'], 0, 0])
    return params

def compute_grad_error(I_left, eps=1e-4):
    I_gray = cv2.cvtColor(I_left, cv2.COLOR_BGR2GRAY).astype(np.float32)
    # sobel gradients
    gx = cv2.Sobel(I_gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(I_gray, cv2.CV_32F, 0, 1, ksize=3)
    G = np.abs(gx) + np.abs(gy)
    E_tex = 1.0 / (G + eps)
    return E_tex #texture error



def px_to_camera(depth_map, K_inv, uv1=None):
    """
    Convert pixel coordinates to 3D camera coordinates for the same view,
    scaled to match depth.
    """
    if uv1 is None:
        H, W = depth_map.shape
        u, v = np.meshgrid(np.arange(W), np.arange(H), indexing='xy')
        uv1 = np.stack([u, v, np.ones_like(u)], axis=-1).reshape(-1, 3)  # (H*W, 3)

    # Camera coordinates: X_c = Z_c * K^-1 * [u,v,1]
    Z_c = depth_map.ravel()    
    X_c = Z_c[:,None] * (K_inv @ uv1.T).T  # (H*W, 3)
    return X_c

def project_to_view(X_one, P_two):
    """Reproject first camera frame 3D points to second camera 2D image view."""
    length = X_one.shape[0]
    X_one_hom = np.hstack([X_one, np.ones((length, 1))])  # (H*W, 4)
    x_two = (P_two @ X_one_hom.T).T  # (H*W, 3)
    
    # Handle division by zero and invalid values
    z_vals = x_two[:, 2]
    valid_mask = np.abs(z_vals) > 1e-6  # avoid division by very small numbers
    
    result = np.full((x_two.shape[0], 2), np.nan)
    result[valid_mask] = x_two[valid_mask, :2] / z_vals[valid_mask, None]
    
    return result


def depth_projection_errors(D_left, x_right, K_inv, P_one, T, fx_B, error_types=['px', 'depth']):
    H, W = D_left.shape
    u_l, v_l = np.meshgrid(np.arange(W), np.arange(H), indexing='xy')
    x_left = np.stack([u_l, v_l], axis=-1).reshape(-1, 2)

    # Split u_R and v_R
    u_r = x_right[..., 0].astype(np.float32).reshape(H, W)
    v_r = x_right[..., 1].astype(np.float32).reshape(H, W)    
    ur_vr_1 = np.stack([u_r, v_r, np.ones_like(u_r)], axis=-1).reshape(-1, 3)

    D_right_prime = fx_B / (u_l-u_r)
    X_c_right_prime1 = px_to_camera(D_right_prime, K_inv, ur_vr_1)
    X_c_right_prime_left = X_c_right_prime1 - T

    x_left_2d_prime = project_to_view(X_c_right_prime_left, P_one)

    # Warp left-depth map in the same left view but at right image pixel location
    # D_left_prime_uv = cv2.remap(D_left.astype(np.float32), u_r, v_r, interpolation=cv2.INTER_LINEAR,
    #               borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    # X_c_right_prime = px_to_camera(D_left_prime_uv, K_inv, ur_vr_1)
    # x_left_2d_prime = project_to_view(X_c_right_prime, P_one)
    
    errors = []
    if 'px' in error_types:
        errors.append(np.abs(x_left_2d_prime - x_left).sum(-1).astype(np.float32).reshape(H, W))
    # if 'depth' in error_types:
    #     errors.append(np.abs(D_left - D_left_prime_uv))

    total_error = np.sum(errors, axis=0)
    return np.nan_to_num(total_error, nan=500, posinf=500,neginf=500)


def photometric_errors(I_L, I_R, x_right, error_types=['l1', 'l2', 'ssim']):
    """
    Computes L1 and L2 error maps between I_L and I_R(x_right), per pixel.

    I_L: HxWx3 left image
    I_R: HxWx3 right image
    x_right: HxWx2 of (u_R, v_R) reprojected coordinates in right image
    """
    H_img, W_img, _ = I_L.shape
    
    # Handle case where x_right comes from a different resolution depth map
    if x_right.shape[0] != H_img * W_img:
        # Calculate actual depth map dimensions
        total_pixels = x_right.shape[0]
        
        # Find the original depth map dimensions by trying common aspect ratios
        for h_depth in range(1, int(np.sqrt(total_pixels)) + 1):
            if total_pixels % h_depth == 0:
                w_depth = total_pixels // h_depth
                if abs(h_depth / w_depth - H_img / W_img) < 0.1:  # similar aspect ratio
                    break
        
        # Reshape to depth map dimensions
        x_right_reshaped = x_right.reshape(h_depth, w_depth, 2)
        
        # Resize to match image dimensions
        u_R = cv2.resize(x_right_reshaped[..., 0].astype(np.float32), (W_img, H_img), interpolation=cv2.INTER_LINEAR)
        v_R = cv2.resize(x_right_reshaped[..., 1].astype(np.float32), (W_img, H_img), interpolation=cv2.INTER_LINEAR)
    else:
        # Direct reshape if dimensions match
        u_R = x_right[..., 0].astype(np.float32).reshape(H_img, W_img)
        v_R = x_right[..., 1].astype(np.float32).reshape(H_img, W_img)

    # Ensure the coordinate maps are in the correct format for cv2.remap
    u_R = u_R.astype(np.float32)
    v_R = v_R.astype(np.float32)
    
    # Handle NaN values by setting them to invalid coordinates
    nan_mask = np.isnan(u_R) | np.isnan(v_R)
    u_R[nan_mask] = -1  # Invalid coordinates that will be ignored
    v_R[nan_mask] = -1
    
    valid = (
        (u_R >= 0) & (u_R < W_img) &
        (v_R >= 0) & (v_R < H_img)
    )

    # Warp each channel using remap
    I_R_warped = cv2.remap(I_R.astype(np.uint8), u_R, v_R, interpolation=cv2.INTER_LINEAR,
                  borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    
    errors = []
    if 'l1' in error_types:
        l1_error = np.mean(np.abs(I_L - I_R_warped), axis=-1)
        l1_error[~valid] = np.nan  # Mark invalid regions
        errors.append(l1_error)
    if 'l2' in error_types:
        l2_error = np.mean((I_L - I_R_warped)**2, axis=-1)
        l2_error[~valid] = np.nan  # Mark invalid regions
        errors.append(l2_error)
    if 'ssim' in error_types:        
        ssim_error = photometric_error_ssim(I_L, I_R_warped)
        ssim_error[~valid] = np.nan  # Mark invalid regions
        errors.append(ssim_error)

    total_error = np.sum(errors, axis=0)
    return total_error

def photometric_error_ssim(I_L, I_R_warped):
    """
    Returns 1 - SSIM index per pixel (approximate perceptual error)
    """
    # Convert to grayscale
    I_L_gray = cv2.cvtColor(I_L, cv2.COLOR_BGR2GRAY)
    I_R_gray = cv2.cvtColor(I_R_warped, cv2.COLOR_BGR2GRAY)

    ssim_map = ssim(I_L_gray, I_R_gray, data_range=255, full=True)[1]  # Returns (mean SSIM score, full map)
    error_map = 1.0 - ssim_map  # dissimilarity

    return error_map


if __name__ == "__main__":
    # Load parameters
    rootdir = "/content/drive/MyDrive/Scene-5/f-28.0mm/a-1.27mm/"
    params_path = os.path.join(rootdir, 'stereocal_results_f28.0mm_a1.27mm','stereocal_params.npz')
    depth_map_path = os.path.join(rootdir, 'rectified_depths_left' ,'depth_anything_v2_all_depths_packed.h5')
    pattern_info_path = os.path.join(rootdir, 'stereocal_results_f28.0mm_a1.27mm' 'pattern_info.json')
    rectified_left_path = os.path.join(rootdir, 'stereocal_results_f28.0mm_a1.27mm', "rectified", "rectified_lefts.h5")
    rectified_right_path = rectified_left_path.replace("lefts", "rights")

    image_index = 14
    params = load_camera_params(params_path)
    print(params)
    K_inv = params['K_inv']
    T = params['T']
    P1, P2 = params['P1'], params['P2']
    
    depth_map = load_h5_images(depth_map_path)[image_index]
    rectified_left = load_h5_images(rectified_left_path)[image_index]
    rectified_right = load_h5_images(rectified_right_path)[image_index]

    print(f"\n image {image_index}: " + \
            f"rectified_left: {rectified_left.shape}, " + \
            f"rectified_right: {rectified_right.shape}, " + \
            f"depth_map: {depth_map.shape}")
    
    # Convert left image regular pixel meshgrid to left camera 3D points,
    # scaled to match depth.
    X_c_left = px_to_camera(depth_map, K_inv)
    
    # Project left camera 3D points to right camera 2D image view
    x_right_2d = project_to_view(X_c_left, P2)

    # Compute photometric errors between I_L_{(u,v):regular pixel meshgrid}
    # and I_R_{(u',v'):coordinates of right image obtained by projecting left image (u,v)}
    err1 = photometric_errors(rectified_left, rectified_right, x_right_2d, ['l1'])
    err2 = photometric_error_ssim(rectified_left, rectified_right)

    err_left = err1 + err2    

    # Calculate mean error only over valid (non-NaN) pixels
    valid_pixels = ~np.isnan(err_left)
    mean_error = np.nanmean(err_left[valid_pixels]) if np.any(valid_pixels) else np.nan
    
    print(f"Mean photometric error: {mean_error:.2f} over {np.sum(valid_pixels)} valid pixels.")
    print(f"Total pixels: {err_left.size}, Valid pixels: {np.sum(valid_pixels)} ({100*np.sum(valid_pixels)/err_left.size:.1f}%)")
