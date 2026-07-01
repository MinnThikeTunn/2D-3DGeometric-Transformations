import cv2
import numpy as np
import matplotlib.pyplot as plt

def create_test_grid(size=(400, 400)):
    """Generates a white geometric grid on a black canvas for clear structural visibility."""
    grid = np.zeros((size[0], size[1], 3), dtype=np.uint8)
    # Draw a bounding rectangle inside to track edge modifications clearly
    cv2.rectangle(grid, (100, 100), (300, 300), (0, 255, 0), 2)
    # Draw grid crosslines
    for i in range(100, 350, 50):
        cv2.line(grid, (i, 100), (i, 300), (255, 255, 255), 1)
        cv2.line(grid, (100, i), (300, i), (255, 255, 255), 1)
    # Label a marker in an asymmetric location to easily detect reflections/flips
    cv2.putText(grid, 'F', (120, 180), cv2.FONT_HERSHEY_SIMPLEX, 2.5, (255, 255, 0), 4)
    return grid

def run_lecture_2_2_transformations(img):
    """
    Implements the exact 2D Homogeneous Coordinate transformations from Lecture 2.2:
    - Basic: Translation, Scaling, Rotation
    - Other: Reflections (X, Y, Origin, y=x) & Shearing (X-direction, Y-direction)
    """
    rows, cols, _ = img.shape
    results = {}

    # --- Part 1: Basic Transformations ---

    # 1. Translation: x' = x + tx, y' = y + ty
    tx, ty = 40, -30
    M_trans = np.float32([[1, 0, tx],
                          [0, 1, ty]])
    results['1. Translation'] = cv2.warpAffine(img, M_trans, (cols, rows))

    # 2. Scaling (Relative to Origin): x' = x * sx, y' = y * sy
    sx, sy = 1.3, 0.7
    M_scale = np.float32([[sx, 0,  0],
                          [0,  sy, 0]])
    results['2. Scaling'] = cv2.warpAffine(img, M_scale, (cols, rows))

    # 3. Rotation (Around Origin): x' = x*cosθ - y*sinθ, y' = x*sinθ + y*cosθ
    theta = np.radians(20)  # Rotate 20 degrees counter-clockwise
    c, s = np.cos(theta), np.sin(theta)
    M_rot = np.float32([[c, -s, 0],
                        [s,  c, 0]])
    results['3. Rotation'] = cv2.warpAffine(img, M_rot, (cols, rows))

    # --- Part 2: Other Transformations (Reflection & Shear) ---

    # 4. Reflection across X-axis: x' = x, y' = -y
    # Since OpenCV coordinate origin is top-left, we offset by height (rows) to keep it visible
    M_reflect_x = np.float32([[1,  0, 0],
                              [0, -1, rows]])
    results['4. Reflection X-axis'] = cv2.warpAffine(img, M_reflect_x, (cols, rows))

    # 5. Reflection across Y-axis: x' = -x, y' = y
    # Offset by width (cols) to keep the flipped image within the frame
    M_reflect_y = np.float32([[-1, 0, cols],
                              [ 0, 1, 0]])
    results['5. Reflection Y-axis'] = cv2.warpAffine(img, M_reflect_y, (cols, rows))

    # 6. Reflection across Origin: x' = -x, y' = -y
    M_reflect_orig = np.float32([[-1,  0, cols],
                                 [ 0, -1, rows]])
    results['6. Reflection Origin'] = cv2.warpAffine(img, M_reflect_orig, (cols, rows))

    # 7. Reflection across Diagonal Line y = x: x' = y, y' = x
    M_reflect_yx = np.float32([[0, 1, 0],
                               [1, 0, 0]])
    results['7. Reflection y=x'] = cv2.warpAffine(img, M_reflect_yx, (cols, rows))

    # 8. X-Direction Shear: x' = x + shx * y, y' = y
    shx = 0.25
    M_shear_x = np.float32([[1, shx, 0],
                            [0, 1,   0]])
    results['8. X-Shear'] = cv2.warpAffine(img, M_shear_x, (cols, rows))

    # 9. Y-Direction Shear: x' = x, y' = y + shy * x
    shy = 0.20
    M_shear_y = np.float32([[1,   0, 0],
                            [shy, 1, 0]])
    results['9. Y-Shear'] = cv2.warpAffine(img, M_shear_y, (cols, rows))

    return results

if __name__ == '__main__':
    # Try reading 'test_image.jpg', fallback to the asymmetric letter grid if missing
    image_path = 'test_image.jpg'
    img_bgr = cv2.imread(image_path)

    if img_bgr is None:
        print(f"Notice: '{image_path}' not detected. Constructing lecture asymmetric canvas.")
        img_rgb = create_test_grid()
    else:
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    # Execute full processing pipelines for Lecture 2.2
    transformations = run_lecture_2_2_transformations(img_rgb)

    # Setup visualization grid for all 10 states (Original + 9 Transformations)
    titles = ['Original Canvas'] + list(transformations.keys())
    images = [img_rgb] + list(transformations.values())

    plt.figure(figsize=(16, 12))
    for i in range(len(images)):
        plt.subplot(2, 5, i + 1)
        plt.imshow(images[i])
        plt.title(titles[i], fontsize=11, fontweight='bold')
        plt.axis('on')
        plt.grid(True, linestyle=':', alpha=0.6)

    plt.suptitle("2D Geometric Transformations )", fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.show()
