import cv2 as cv
import numpy as np
from pso import pso

# ----------------- Enhancement Utilities -----------------

def apply_histogram_equalization(image):
    ycrcb = cv.cvtColor(image, cv.COLOR_RGB2YCrCb)
    ycrcb[:, :, 0] = cv.equalizeHist(ycrcb[:, :, 0])
    return cv.cvtColor(ycrcb, cv.COLOR_YCrCb2RGB)

def rghs(image):
    lab = cv.cvtColor(image, cv.COLOR_RGB2Lab)
    L, A, B = cv.split(lab)

    L = L.astype(np.float32)
    L = (L - np.min(L)) / (np.max(L) - np.min(L)) * 255
    L = L.astype(np.uint8)

    lab = cv.merge((L, A, B))
    return cv.cvtColor(lab, cv.COLOR_Lab2RGB)

def gaussian_blur(image):
    return cv.GaussianBlur(image, (3, 3), 0)

def unsharp_masking(image):
    blurred = cv.GaussianBlur(image, (9, 9), 10.0)
    return cv.addWeighted(image, 1.5, blurred, -0.5, 0)

def neutralize_image(image):
    mean = np.mean(image, axis=(0, 1))
    mean_gray = np.mean(mean)
    scale = mean_gray / (mean + 1e-5)
    image = np.clip(image * scale, 0, 255)
    return image.astype(np.uint8)

def Stretching(image):
    min_val = np.min(image, axis=(0, 1), keepdims=True)
    max_val = np.max(image, axis=(0, 1), keepdims=True)
    stretched = (image - min_val) * (255.0 / (max_val - min_val + 1e-5))
    return np.clip(stretched, 0, 255).astype(np.uint8)

# ----------------- Metrics -----------------

def compute_entropy(image):
    gray = cv.cvtColor(image, cv.COLOR_RGB2GRAY)
    hist = cv.calcHist([gray], [0], None, [256], [0, 256])
    hist = hist.ravel() / hist.sum()
    hist = hist[hist > 0]
    return -np.sum(hist * np.log2(hist))

def compute_contrast(image):
    gray = cv.cvtColor(image, cv.COLOR_RGB2GRAY)
    return np.std(gray)

def compute_cci(image):
    R, G, B = image[:, :, 0], image[:, :, 1], image[:, :, 2]
    rg = np.abs(R - G)
    yb = np.abs(0.5 * (R + G) - B)
    std_rg = np.std(rg)
    std_yb = np.std(yb)
    mean_rg = np.mean(rg)
    mean_yb = np.mean(yb)
    return np.sqrt(std_rg**2 + std_yb**2) + 0.3 * np.sqrt(mean_rg**2 + mean_yb**2)

def compute_iqi(image):
    gray = cv.cvtColor(image, cv.COLOR_RGB2GRAY)
    mean = np.mean(gray)
    std = np.std(gray)
    return 0 if mean == 0 else std / mean

# ----------------- PSO-Based Enhancement -----------------

def enhanced_image(image, weights):
    neutral = neutralize_image(image)
    stretched = Stretching(neutral)
    unsharpened = unsharp_masking(stretched)
    enhanced = (weights[0] * neutral + weights[1] * stretched + weights[2] * unsharpened)
    return np.clip(enhanced, 0, 255).astype(np.uint8)

def objective_function_factory(image):
    def objective(weights):
        if np.sum(weights) == 0:
            return float("inf")
        weights = weights / np.sum(weights)
        enhanced = enhanced_image(image, weights)
        contrast = compute_contrast(enhanced)
        entropy = compute_entropy(enhanced)
        return -(contrast + entropy)  # Maximize both (minimize negative sum)
    return objective

def pso_image(image):
    func = objective_function_factory(image)
    params = {
        "wmax": 0.9,
        "wmin": 0.4,
        "c1": 2,
        "c2": 2,
    }
    gbest = pso(func, max_iter=20, num_particles=10, dim=3, vmin=0, vmax=1, params=params)
    weights = gbest["position"]
    weights = weights / np.sum(weights)
    return enhanced_image(image, weights)

# ----------------- NUCE Pipeline -----------------

def NUCE(image):
    image = gaussian_blur(image)
    return pso_image(image)
