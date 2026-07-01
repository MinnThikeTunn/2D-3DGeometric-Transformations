import cv2
import numpy as np
import matplotlib.pyplot as plt

def create_test_grid(size=(400, 400)):
    """Generates a white grid image on a black background for clear geometric visualization."""
    grid = np.zeros((size[0], size[1], 3), dtype=np.uint8)
    # Draw a bounding square inside to see boundaries clearly
    cv2.rectangle(grid, (50, 50), (350, 350), (0, 255, 0), 2)
    # Draw inner grid lines
    for i in range(50, 350, 50):
        cv2.line(grid, (i, 50), (i, 350), (255, 255, 255), 1)
        cv2.line(grid, (50, i), (350, i), (255, 255, 255), 1)
    # Label a marker
    cv2.putText(grid, 'UIT CV', (70, 120), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 0), 3)
    return grid

def run_2d_transformations(img):
    """
    Implements the 2D coordinate transformation hierarchy from Slide 6:
    Translation, Rigid (Euclidean), Similarity, Affine, and Projective.
    """
    rows, cols, _ = img.shape
    results = {}

    # 1. 2D Translation Matrix [I | t] (2 DoF) - Preserves Orientation
    # x' = x + tx, y' = y + ty
    tx, ty = 40, 30
    M_translation = np.float32([[1, 0, tx],
                                [0, 1, ty]])
    results['Translation'] = cv2.warpAffine(img, M_translation, (cols, rows))

    # 2. 2D Rigid / Euclidean Transformation [R | t] (3 DoF) - Preserves Lengths
    # Matrix format: [[cos(theta), -sin(theta), tx], [sin(theta), cos(theta), ty]]
    theta = np.radians(15)  # Rotate 15 degrees
    c, s = np.cos(theta), np.sin(theta)
    M_rigid = np.float32([[c, -s, 20],
                          [s,  c, 10]])
    results['Rigid (Euclidean)'] = cv2.warpAffine(img, M_rigid, (cols, rows))

    # 3. 2D Similarity Transformation [sR | t] (4 DoF) - Preserves Angles
    # Matrix format: [[a*cos(theta), -a*sin(theta), tx], [a*sin(theta), a*cos(theta), ty]]
    scale = 0.75
    M_similarity = np.float32([[scale * c, -scale * s, 60],
                               [scale * s,  scale * c, 40]])
    results['Similarity'] = cv2.warpAffine(img, M_similarity, (cols, rows))

    # 4. 2D Affine Transformation [A] (6 DoF) - Preserves Parallelism
    # Matrix format: [[a, b, c], [d, e, f]]
    # Map from 3 points to 3 points
    pts_src = np.float32([[50, 50], [200, 50], [50, 200]])
    pts_dst = np.float32([[70, 60], [180, 80], [60, 230]])
    M_affine = cv2.getAffineTransform(pts_src, pts_dst)
    results['Affine'] = cv2.warpAffine(img, M_affine, (cols, rows))

    # 5. 2D Projective Transformation / Homography [H] (8 DoF) - Preserves Straight Lines
    # Matrix format: 3x3 matrix [[a, b, c], [d, e, f], [g, h, 1]]
    pts_src_4 = np.float32([[50, 50], [350, 50], [50, 350], [350, 350]])
    pts_dst_4 = np.float32([[80, 70], [320, 100], [40, 320], [380, 280]])
    M_projective = cv2.getPerspectiveTransform(pts_src_4, pts_dst_4)
    results['Projective'] = cv2.warpPerspective(img, M_projective, (cols, rows))

    return results

def compute_3d_transformations():
    """
    Computes pure analytical Homogeneous Matrix Operations for 3D graphics
    as stated in Slides 7 to 10 using homogeneous point processing.
    """
    print("\n--- Analytical 3D Transformations Demonstration ---")
    # Define a target 3D point P = [x, y, z, 1]^T in Homogeneous Coordinates
    P = np.array([10.0, 20.0, 30.0, 1.0])
    print(f"Original 3D point P: {P[:3]}")

    # 1. 3D Translation Matrix (Slide 7) - FIXED SYNTAX
    tx, ty, tz = 5, -10, 15
    T_3d = np.array([[1, 0, 0, tx],
                     [0, 1, 0, ty],
                     [0, 0, 1, tz],
                     [0, 0, 0, 1]])
    P_translated = T_3d @ P
    print(f"After 3D Translation: {P_translated[:3]}")

    # 2. 3D Scaling Matrix (Slide 8)
    sx, sy, sz = 1.5, 2.0, 0.5
    S_3d = np.array([[sx,  0,  0, 0],
                     [ 0, sy,  0, 0],
                     [ 0,  0, sz, 0],
                     [ 0,  0,  0, 1]])
    P_scaled = S_3d @ P
    print(f"After 3D Scaling: {P_scaled[:3]}")

    # 3. 3D Rotations (Slides 9 & 10)
    alpha = np.radians(45)  # 45 degrees
    ca, sa = np.cos(alpha), np.sin(alpha)

    # Rotation about Z-axis
    R_z = np.array([[ca, -sa,  0,  0],
                    [sa,  ca,  0,  0],
                    [ 0,   0,  1,  0],
                    [ 0,   0,  0,  1]])

    # Rotation about Y-axis
    R_y = np.array([[ ca,  0, sa,  0],
                    [  0,  1,  0,  0],
                    [-sa,  0, ca,  0],
                    [  0,  0,  0,  1]])

    # Rotation about X-axis
    R_x = np.array([[1,   0,   0,  0],
                    [0,  ca, -sa,  0],
                    [0,  sa,  ca,  0],
                    [0,   0,   0,  1]])

    print(f"After Rotation around X-axis: {(R_x @ P)[:3]}")
    print(f"After Rotation around Y-axis: {(R_y @ P)[:3]}")
    print(f"After Rotation around Z-axis: {(R_z @ P)[:3]}")

if __name__ == '__main__':
    # Try reading an image, fallback to a clean generated canvas grid if path isn't provided
    image_path = 'test_image.jpg'
    img_bgr = cv2.imread(image_path)

    if img_bgr is None:
        print(f"Notice: '{image_path}' not detected. Constructing lecture template canvas structure.")
        img_rgb = create_test_grid()
    else:
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    # Execute 2D processing pipelines
    transformations_2d = run_2d_transformations(img_rgb)

    # Plotting 2D Hierarchy Results Side-by-Side
    titles = ['Original Canvas'] + list(transformations_2d.keys())
    images = [img_rgb] + list(transformations_2d.values())

    plt.figure(figsize=(15, 10))
    for i in range(len(images)):
        plt.subplot(2, 3, i + 1)
        plt.imshow(images[i])
        plt.title(titles[i], fontsize=12, fontweight='bold')
        plt.axis('on')
        plt.grid(True, linestyle='--', alpha=0.5)

    plt.suptitle("3D Geometric Transformations", fontsize=16, fontweight='bold')
    plt.tight_layout()

    # Print out analytical verification matrices for 3D objects
    compute_3d_transformations()

    # Show the structural visual plots
    plt.show()
