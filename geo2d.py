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
    Returns: (images_dict, matrices_dict, params_dict)
    """
    rows, cols, _ = img.shape
    results = {}
    matrices = {}
    params = {}

    # --- Part 1: Basic Transformations ---

    # 1. Translation: x' = x + tx, y' = y + ty
    tx, ty = 40, -30
    M_trans = np.float32([[1, 0, tx],
                          [0, 1, ty]])
    results['1. Translation'] = cv2.warpAffine(img, M_trans, (cols, rows))
    matrices['1. Translation'] = M_trans
    params['1. Translation'] = f'tx={tx}, ty={ty}'

    # 2. Scaling (Relative to Origin): x' = x * sx, y' = y * sy
    sx, sy = 1.3, 0.7
    M_scale = np.float32([[sx, 0,  0],
                          [0,  sy, 0]])
    results['2. Scaling'] = cv2.warpAffine(img, M_scale, (cols, rows))
    matrices['2. Scaling'] = M_scale
    params['2. Scaling'] = f'sx={sx}, sy={sy}'

    # 3. Rotation (Around Origin): x' = x*cosθ - y*sinθ, y' = x*sinθ + y*cosθ
    theta = np.radians(20)  # Rotate 20 degrees counter-clockwise
    c, s = np.cos(theta), np.sin(theta)
    M_rot = np.float32([[c, -s, 0],
                        [s,  c, 0]])
    results['3. Rotation'] = cv2.warpAffine(img, M_rot, (cols, rows))
    matrices['3. Rotation'] = M_rot
    params['3. Rotation'] = f'θ=20°'

    # --- Part 2: Other Transformations (Reflection & Shear) ---

    # 4. Reflection across X-axis: x' = x, y' = -y
    # Since OpenCV coordinate origin is top-left, we offset by height (rows) to keep it visible
    M_reflect_x = np.float32([[1,  0, 0],
                              [0, -1, rows]])
    results['4. Reflection X-axis'] = cv2.warpAffine(img, M_reflect_x, (cols, rows))
    matrices['4. Reflection X-axis'] = M_reflect_x
    params['4. Reflection X-axis'] = 'y\' = -y'

    # 5. Reflection across Y-axis: x' = -x, y' = y
    # Offset by width (cols) to keep the flipped image within the frame
    M_reflect_y = np.float32([[-1, 0, cols],
                              [ 0, 1, 0]])
    results['5. Reflection Y-axis'] = cv2.warpAffine(img, M_reflect_y, (cols, rows))
    matrices['5. Reflection Y-axis'] = M_reflect_y
    params['5. Reflection Y-axis'] = 'x\' = -x'

    # 6. Reflection across Origin: x' = -x, y' = -y
    M_reflect_orig = np.float32([[-1,  0, cols],
                                 [ 0, -1, rows]])
    results['6. Reflection Origin'] = cv2.warpAffine(img, M_reflect_orig, (cols, rows))
    matrices['6. Reflection Origin'] = M_reflect_orig
    params['6. Reflection Origin'] = 'x\' = -x, y\' = -y'

    # 7. Reflection across Diagonal Line y = x: x' = y, y' = x
    M_reflect_yx = np.float32([[0, 1, 0],
                               [1, 0, 0]])
    results['7. Reflection y=x'] = cv2.warpAffine(img, M_reflect_yx, (cols, rows))
    matrices['7. Reflection y=x'] = M_reflect_yx
    params['7. Reflection y=x'] = 'x\' = y, y\' = x'

    # 8. X-Direction Shear: x' = x + shx * y, y' = y
    shx = 0.25
    M_shear_x = np.float32([[1, shx, 0],
                            [0, 1,   0]])
    results['8. X-Shear'] = cv2.warpAffine(img, M_shear_x, (cols, rows))
    matrices['8. X-Shear'] = M_shear_x
    params['8. X-Shear'] = f'shx={shx}'

    # 9. Y-Direction Shear: x' = x, y' = y + shy * x
    shy = 0.20
    M_shear_y = np.float32([[1,   0, 0],
                            [shy, 1, 0]])
    results['9. Y-Shear'] = cv2.warpAffine(img, M_shear_y, (cols, rows))
    matrices['9. Y-Shear'] = M_shear_y
    params['9. Y-Shear'] = f'shy={shy}'

    return results, matrices, params

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
    transformations, matrices, params = run_lecture_2_2_transformations(img_rgb)

    # Setup visualization grid for all 10 states (Original + 9 Transformations)
    titles = ['Original Canvas'] + list(transformations.keys())
    images = [img_rgb] + list(transformations.values())
    param_texts = ['Original'] + list(params.values())

    # Create figure with 3x4 grid layout
    fig = plt.figure(figsize=(20, 14))
    axes = []

    # Plot all images
    for i in range(len(images)):
        ax = plt.subplot(3, 4, i + 1)
        ax.imshow(images[i])
        ax.set_title(titles[i], fontsize=12, fontweight='bold', color='#333333', pad=15)
        ax.set_xlabel(param_texts[i], fontsize=9, color='#666666', style='italic')
        ax.axis('on')
        ax.grid(True, linestyle=':', alpha=0.3, color='#cccccc')
        axes.append(ax)

    plt.suptitle("2D Geometric Transformations (Interactive)", fontsize=18, fontweight='bold', color='#222222')
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    # Add matplotlib toolbar
    from matplotlib.widgets import Button
    from matplotlib import rcParams

    # Global state for interaction (using list to avoid nonlocal issues)
    highlighted_idx = [-1]
    status_text = fig.text(0.02, 0.02, '', fontsize=10, color='#444444', 
                           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # Mouse hover event - display coordinates
    def on_hover(event):
        if event.inaxes in axes:
            idx = axes.index(event.inaxes)
            x, y = event.xdata, event.ydata
            if x is not None and y is not None:
                status_text.set_text(f'Hover: {titles[idx]} | Coordinates: ({x:.1f}, {y:.1f})')
            else:
                status_text.set_text(f'Hover: {titles[idx]}')
        else:
            status_text.set_text('')

    # Click event - show transformation matrix
    def on_click(event):
        if event.inaxes in axes:
            idx = axes.index(event.inaxes)
            highlighted_idx[0] = idx
            
            # Reset all edge colors
            for ax in axes:
                for spine in ax.spines.values():
                    spine.set_edgecolor('black')
                    spine.set_linewidth(1)
            
            # Highlight selected subplot
            for spine in axes[idx].spines.values():
                spine.set_edgecolor('red')
                spine.set_linewidth(3)
            
            # Display matrix info
            if idx == 0:
                print("\n=== Original Canvas ===")
                print("No transformation applied")
            else:
                title = titles[idx]
                matrix = matrices[title]
                print(f"\n=== {title} ===")
                print(f"Parameters: {params[title]}")
                print("Transformation Matrix:")
                print(matrix)
            
            fig.canvas.draw()

    # Keyboard shortcuts
    def on_key(event):
        if event.key == 'n' or event.key == 'right':
            highlighted_idx[0] = (highlighted_idx[0] + 1) % len(axes) if highlighted_idx[0] >= 0 else 0
            on_click(type('obj', (object,), {'inaxes': axes[highlighted_idx[0]]}))
        elif event.key == 'p' or event.key == 'left':
            highlighted_idx[0] = (highlighted_idx[0] - 1) % len(axes) if highlighted_idx[0] >= 0 else len(axes) - 1
            on_click(type('obj', (object,), {'inaxes': axes[highlighted_idx[0]]}))
        elif event.key == 's':
            fig.savefig('transformations.png', dpi=150, bbox_inches='tight')
            print("Figure saved as 'transformations.png'")
        elif event.key == 'r':
            # Reset view
            for ax in axes:
                for spine in ax.spines.values():
                    spine.set_edgecolor('black')
                    spine.set_linewidth(1)
            highlighted_idx[0] = -1
            fig.canvas.draw()
            print("View reset")

    # Connect event handlers
    fig.canvas.mpl_connect('motion_notify_event', on_hover)
    fig.canvas.mpl_connect('button_press_event', on_click)
    fig.canvas.mpl_connect('key_press_event', on_key)

    print("\n=== Interactive Controls ===")
    print("• Hover over images to see coordinates")
    print("• Click on an image to see its transformation matrix")
    print("• Press 'n' or → to highlight next transformation")
    print("• Press 'p' or ← to highlight previous transformation")
    print("• Press 's' to save the figure")
    print("• Press 'r' to reset the view")
    print("• Use matplotlib toolbar for zoom/pan\n")

    plt.show()
