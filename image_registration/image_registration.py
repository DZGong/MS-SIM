"""
This script performs image registration and analysis on a set of images.
It includes functions to load images, normalize them, detect and compute keypoints,
match descriptors, compute homography matrices, warp images, and calculate various metrics
such as RMSE, SSIM, and PSNR. The script also includes functions to visualize keypoints and matches,
visualize warped images, overlay and difference images, and plot metrics between image stacks.

The image registration if preformed by solving for a homography matrix basis.
The homography matrix is then applied to the images to align them.
Homography matrices solved for with the calibration beads can then be applied to new images.
"""

#%% Imports
import cv2
import numpy as np
import os
import tifffile as tiff
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim, mean_squared_error


#%% 1. Load Images
def load_z_stack(image_path):
    """Load the z-stack from a TIFF file."""
    with tiff.TiffFile(image_path) as tif:
        z_stack = [page.asarray() for page in tif.pages]
    return z_stack


def load_images(directory):
    """Load image paths and z-stacks from a directory."""
    image_paths = [os.path.join(directory, filename) for filename in os.listdir(directory) if filename.endswith('.tif')]
    
    if not image_paths:
        raise ValueError("No TIFF images found in the directory.")
    
    z_stacks = [load_z_stack(image_path) for image_path in image_paths]
    return z_stacks, image_paths


#%% 2. Image Processing Functions
def normalize_and_convert_to_uint8_single(image):
    """Normalize an image to the range [0, 255] and convert it to uint8."""
    image_normalized = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX)
    return image_normalized.astype(np.uint8)


def detect_and_compute_keypoints(image, detector):
    """Detect keypoints and compute descriptors using a specified detector."""
    keypoints, descriptors = detector.detectAndCompute(image, None)
    return keypoints, descriptors


def match_descriptors(descriptors1, descriptors2, distance_metric=cv2.NORM_L2, cross_check=True):
    """Match descriptors using the specified distance metric."""
    bf = cv2.BFMatcher(distance_metric, crossCheck=cross_check)
    matches = bf.match(descriptors1, descriptors2)
    return sorted(matches, key=lambda x: x.distance)


def filter_matches_by_distance(matches, keypoints1, keypoints2, max_distance=100):
    """Filter matches by the distance between keypoints."""
    filtered_matches = [
        match for match in matches
        if np.linalg.norm(np.array(keypoints1[match.queryIdx].pt) - np.array(keypoints2[match.trainIdx].pt)) < max_distance
    ]
    return filtered_matches


def compute_homography(filtered_matches, keypoints1, keypoints2):
    """Compute the homography matrix based on the filtered matches."""
    if len(filtered_matches) >= 4:
        src_pts = np.float32([keypoints1[m.queryIdx].pt for m in filtered_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([keypoints2[m.trainIdx].pt for m in filtered_matches]).reshape(-1, 1, 2)
        H, mask = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC, 5.0)
        return H, mask
    else:
        print("Not enough matches are found to compute a reliable homography.")
        return None, None


def warp_image(image, H, shape):
    """Warp an image using a homography matrix."""
    return cv2.warpPerspective(image, H, shape)


#%% 3. Visualization Functions
def visualize_keypoints_and_matches(image1, keypoints1, image2, keypoints2, matches, title):
    """Visualize keypoints and matches between two images."""
    img_matches = cv2.drawMatches(image1, keypoints1, image2, keypoints2, matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    plt.figure(figsize=(15, 5))
    plt.title(title)
    plt.imshow(img_matches)
    plt.axis('off')
    plt.show()


def visualize_warped_image(image1, image2, warped_image):
    """Visualize the original images and the warped image."""
    plt.figure(figsize=(18, 6))

    plt.subplot(1, 3, 1)
    plt.title('Plane 1 Slice 1')
    plt.imshow(image1, cmap='gray')
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.title('Plane 2 Slice 1')
    plt.imshow(image2, cmap='gray')
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.title('Warped Plane 2 Slice 1')
    plt.imshow(warped_image, cmap='gray')
    plt.axis('off')

    plt.tight_layout()
    plt.show()


def overlay_and_difference_images(image1, warped_image):
    """Overlay and compute the difference between the original and warped images."""
    overlay = cv2.addWeighted(image1, 0.5, warped_image, 0.5, 0)
    difference = cv2.absdiff(image1, warped_image)

    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    plt.title('Overlay of Plane 1 and Warped Plane 2')
    plt.imshow(overlay, cmap='gray')
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.title('Difference between Plane 1 and Warped Plane 2')
    plt.imshow(difference, cmap='gray')
    plt.axis('off')

    plt.tight_layout()
    plt.show()


#%% 4. Alignment Functions
def align_and_return_transform(plane1, plane2):
    """Align two images using SIFT and homography."""
    sift = cv2.SIFT_create()
    keypoints1, descriptors1 = detect_and_compute_keypoints(plane1, sift)
    keypoints2, descriptors2 = detect_and_compute_keypoints(plane2, sift)
    matches = match_descriptors(descriptors1, descriptors2)
    filtered_matches = filter_matches_by_distance(matches, keypoints1, keypoints2, max_distance=10)
    H, _ = compute_homography(filtered_matches, keypoints1, keypoints2)

    warped_plane2 = None
    if H is not None:
        height, width = plane1.shape
        warped_plane2 = warp_image(plane2, H, (width, height))

    return warped_plane2, (filtered_matches, keypoints1, keypoints2, H)


def align_stacks_to_previous_slices(plane1_stack, plane2_stack, plane3_stack, offset=7):
    """Align each slice in the stack to the slice `offset` positions above or below it.

    The resulting aligned stacks will have the same length as `plane2_stack`,
    with zero images (black images) added to `aligned_plane1_stack` and `aligned_plane3_stack` as needed.
    """
    aligned_plane1_stack, aligned_plane3_stack = [], []
    all_info_plane1, all_info_plane3 = [], []

    height, width = plane2_stack[0].shape

    for i in range(len(plane2_stack)):
        # Handling alignment for plane1_stack
        if i >= len(plane2_stack) - offset:
            # Prepend black images for indices where plane1 does not have slices to align
            aligned_plane1_stack.append(np.zeros((height, width), dtype=np.uint8))
            all_info_plane1.append(None)
        else:
            # Align slices within the valid range for plane1_stack
            aligned_plane1_slice, info_plane1 = align_and_return_transform(plane2_stack[i + offset], plane1_stack[i])
            aligned_plane1_stack.append(aligned_plane1_slice)
            all_info_plane1.append(info_plane1)

        # Handling alignment for plane3_stack
        if i < offset:
            # Append black images for indices beyond the length of plane3_stack
            aligned_plane3_stack.append(np.zeros((height, width), dtype=np.uint8))
            all_info_plane3.append(None)
        else:
            # Align slices within the valid range for plane3_stack
            aligned_plane3_slice, info_plane3 = align_and_return_transform(plane2_stack[i - offset], plane3_stack[i])
            aligned_plane3_stack.append(aligned_plane3_slice)
            all_info_plane3.append(info_plane3)

    return (
        (np.array(aligned_plane1_stack), plane2_stack, np.array(aligned_plane3_stack)),
        all_info_plane1,
        all_info_plane3
    )


#%% 5. Metrics and Displacement Calculation
def calculate_rmse(image1, image2):
    """Calculate the Root Mean Square Error (RMSE) between two images."""
    return np.sqrt(mean_squared_error(image1, image2))


def calculate_ssim(image1, image2):
    """Calculate the Structural Similarity Index (SSIM) between two images."""
    return ssim(image1, image2, data_range=255)


def calculate_psnr(image1, image2):
    """Calculate the Peak Signal-to-Noise Ratio (PSNR) between two images."""
    return cv2.PSNR(image1, image2)


def calculate_metrics(image1, image2):
    """Calculate RMSE, SSIM, and PSNR between two images."""
    rmse = calculate_rmse(image1, image2)
    ssim_value = calculate_ssim(image1, image2)
    psnr_value = calculate_psnr(image1, image2)
    return rmse, ssim_value, psnr_value


def compute_metrics_between_stacks(stack_a, stack_b, offset=7, compare_ahead=True):
    """Compute metrics between two image stacks."""
    metrics_results = []
    for i in range(len(stack_a)):
        try:
            if compare_ahead:
                image_a = stack_a[i]
                image_b = stack_b[i + offset]
            else:
                if i < offset:
                    continue
                image_a = stack_a[i]
                image_b = stack_b[i - offset]
                
            rmse, ssim_value, psnr_value = calculate_metrics(image_a, image_b)
            metrics_results.append({
                'slice_index_a': i,
                'slice_index_b': i + offset if compare_ahead else i - offset,
                'rmse': rmse,
                'ssim': ssim_value,
                'psnr': psnr_value
            })
        except IndexError:
            continue
    
    return metrics_results


def plot_metrics(metrics, stack_name_a, stack_name_b):
    """Plot RMSE, SSIM, and PSNR between two image stacks."""
    indices_a = [result['slice_index_a'] for result in metrics]
    indices_b = [result['slice_index_b'] for result in metrics]
    rmses = [result['rmse'] for result in metrics]
    ssims = [result['ssim'] for result in metrics]
    psnrs = [result['psnr'] for result in metrics]

    plt.figure(figsize=(14, 8))

    plt.subplot(3, 1, 1)
    plt.plot(indices_b, rmses, 'o-', label=f'{stack_name_a} vs {stack_name_b} RMSE')
    plt.xlabel('Slice Index')
    plt.ylabel('RMSE')
    plt.title(f'RMSE between {stack_name_a} and {stack_name_b}')
    plt.legend()

    plt.subplot(3, 1, 2)
    plt.plot(indices_b, ssims, 'o-', label=f'{stack_name_a} vs {stack_name_b} SSIM')
    plt.xlabel('Slice Index')
    plt.ylabel('SSIM')
    plt.title(f'SSIM between {stack_name_a} and {stack_name_b}')
    plt.ylim(0, 1)
    plt.legend()

    plt.subplot(3, 1, 3)
    plt.plot(indices_b, psnrs, 'o-', label=f'{stack_name_a} vs {stack_name_b} PSNR')
    plt.xlabel('Slice Index')
    plt.ylabel('PSNR (dB)')
    plt.title(f'PSNR between {stack_name_a} and {stack_name_b}')
    plt.legend()

    plt.tight_layout()
    plt.show()


#%% 6. Main Function to Run the Analysis
def main():
    """Main function to run the image alignment and analysis."""
    # Set up the directories and load images
    directory = os.path.join('data', 'raw_data')
    z_stacks, image_paths = load_images(directory)
    plane1_stack, plane2_stack, plane3_stack = z_stacks

    # Normalize the plane1, plane2, and plane3 stacks
    plane1_stack_normalized = [normalize_and_convert_to_uint8_single(slice) for slice in plane1_stack]
    plane2_stack_normalized = [normalize_and_convert_to_uint8_single(slice) for slice in plane2_stack]
    plane3_stack_normalized = [normalize_and_convert_to_uint8_single(slice) for slice in plane3_stack]

    # Align the stacks
    (aligned_plane1_stack, aligned_plane2_stack, aligned_plane3_stack), info_plane1, info_plane3 = align_stacks_to_previous_slices(
        plane1_stack_normalized, plane2_stack_normalized, plane3_stack_normalized
    )

    # Compute and plot metrics between aligned stacks and the original stack
    metrics_stack1_stack2 = compute_metrics_between_stacks(aligned_plane1_stack, plane2_stack_normalized)
    plot_metrics(metrics_stack1_stack2, "Aligned Stack1", "Stack2")

    metrics_stack2_stack3 = compute_metrics_between_stacks(aligned_plane3_stack, plane2_stack_normalized, compare_ahead=False)
    plot_metrics(metrics_stack2_stack3, "Aligned Stack3", "Stack2")


#%% 7. Apply the metrics without alignment
def view_metrics_without_alignment():
    """View the metrics between the original stacks."""
    # Set up the directories and load images
    directory = os.path.join('data', 'beads')
    z_stacks, image_paths = load_images(directory)
    plane1_stack, plane2_stack, plane3_stack = z_stacks

    # Normalize the plane1, plane2, and plane3 stacks
    plane1_stack_normalized = [normalize_and_convert_to_uint8_single(slice) for slice in plane1_stack]
    plane2_stack_normalized = [normalize_and_convert_to_uint8_single(slice) for slice in plane2_stack]
    plane3_stack_normalized = [normalize_and_convert_to_uint8_single(slice) for slice in plane3_stack]

    # Align the stacks
    (aligned_plane1_stack, aligned_plane2_stack, aligned_plane3_stack), info_plane1, info_plane3 = align_stacks_to_previous_slices(
        plane1_stack_normalized, plane2_stack_normalized, plane3_stack_normalized
    )

    # Compare metrics for unaligned vs aligned stacks (plane1 vs plane2)
    metrics_unaligned, metrics_aligned = compare_metrics_aligned_vs_unaligned(
        plane1_stack_normalized, plane2_stack_normalized,
        aligned_plane1_stack, plane2_stack_normalized
    )
    plot_comparison_metrics(metrics_unaligned, metrics_aligned, "Stack1", "Stack2")

    # Compare metrics for unaligned vs aligned stacks (plane3 vs plane2)
    metrics_unaligned, metrics_aligned = compare_metrics_aligned_vs_unaligned(
        plane3_stack_normalized, plane2_stack_normalized,
        aligned_plane3_stack, plane2_stack_normalized, compare_ahead=False
    )
    plot_comparison_metrics(metrics_unaligned, metrics_aligned, "Stack3", "Stack2")


def compare_metrics_aligned_vs_unaligned(unaligned_stack_a, unaligned_stack_b, aligned_stack_a, aligned_stack_b, offset=7, compare_ahead=True):
    """Compare metrics between unaligned and aligned stacks."""
    # Compute metrics for unaligned stacks
    metrics_unaligned = compute_metrics_between_stacks(unaligned_stack_a, unaligned_stack_b, offset=offset, compare_ahead=compare_ahead)

    # Compute metrics for aligned stacks
    metrics_aligned = compute_metrics_between_stacks(aligned_stack_a, aligned_stack_b, offset=offset, compare_ahead=compare_ahead)

    return metrics_unaligned, metrics_aligned


def plot_comparison_metrics(metrics_unaligned, metrics_aligned, stack_name_a, stack_name_b):
    """Plot RMSE, SSIM, and PSNR for unaligned and aligned stacks."""
    indices_b_unaligned = [result['slice_index_b'] for result in metrics_unaligned]
    indices_b_aligned = [result['slice_index_b'] for result in metrics_aligned]

    rmses_unaligned = [result['rmse'] for result in metrics_unaligned]
    rmses_aligned = [result['rmse'] for result in metrics_aligned]

    ssims_unaligned = [result['ssim'] for result in metrics_unaligned]
    ssims_aligned = [result['ssim'] for result in metrics_aligned]

    psnrs_unaligned = [result['psnr'] for result in metrics_unaligned]
    psnrs_aligned = [result['psnr'] for result in metrics_aligned]

    plt.figure(figsize=(14, 8))

    # RMSE Comparison
    plt.subplot(3, 1, 1)
    plt.plot(indices_b_unaligned, rmses_unaligned, 'o--', label=f'Unaligned {stack_name_a} vs {stack_name_b} RMSE')
    plt.plot(indices_b_aligned, rmses_aligned, 'o-', label=f'Aligned {stack_name_a} vs {stack_name_b} RMSE')
    plt.xlabel('Slice Index')
    plt.ylabel('RMSE')
    plt.title(f'RMSE: Unaligned vs Aligned {stack_name_a} and {stack_name_b}')
    plt.legend()

    # SSIM Comparison
    plt.subplot(3, 1, 2)
    plt.plot(indices_b_unaligned, ssims_unaligned, 'o--', label=f'Unaligned {stack_name_a} vs {stack_name_b} SSIM')
    plt.plot(indices_b_aligned, ssims_aligned, 'o-', label=f'Aligned {stack_name_a} vs {stack_name_b} SSIM')
    plt.xlabel('Slice Index')
    plt.ylabel('SSIM')
    plt.ylim(0, 1)
    plt.title(f'SSIM: Unaligned vs Aligned {stack_name_a} and {stack_name_b}')
    plt.legend()

    # PSNR Comparison
    plt.subplot(3, 1, 3)
    plt.plot(indices_b_unaligned, psnrs_unaligned, 'o--', label=f'Unaligned {stack_name_a} vs {stack_name_b} PSNR')
    plt.plot(indices_b_aligned, psnrs_aligned, 'o-', label=f'Aligned {stack_name_a} vs {stack_name_b} PSNR')
    plt.xlabel('Slice Index')
    plt.ylabel('PSNR (dB)')
    plt.title(f'PSNR: Unaligned vs Aligned {stack_name_a} and {stack_name_b}')
    plt.legend()

    plt.tight_layout()
    plt.show()


# %% 9. Apply the homography matrix transformations to new images
def apply_transformations_to_new_images(new_plane_stack, H_matrices):
    """Apply the stored homography matrices to a new set of images."""
    transformed_stack = []

    height, width = new_plane_stack[0].shape

    for i, H in enumerate(H_matrices):
        if H is not None:
            transformed_image = warp_image(new_plane_stack[i], H, (width, height))
        else:
            transformed_image = np.zeros((height, width), dtype=np.uint8)  # Add black image if no transformation
        transformed_stack.append(transformed_image)

    return np.array(transformed_stack)


# %% Main
if __name__ == "__main__":
    main()

# %%
