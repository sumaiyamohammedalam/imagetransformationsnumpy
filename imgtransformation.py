import numpy as np
import matplotlib.pyplot as plt
import codecademylib3

heart_img = np.array([[255,0,0,255,0,0,255],
              [0,255/2,255/2,0,255/2,255/2,0],
          [0,255/2,255/2,255/2,255/2,255/2,0],
          [0,255/2,255/2,255/2,255/2,255/2,0],
              [255,0,255/2,255/2,255/2,0,255],
                  [255,255,0,255/2,0,255,255],
                  [255,255,255,0,255,255,255]])

# This is a helper function that makes it easy for you to show images!
def show_image(image, name_identifier):
    plt.imshow(image, cmap="gray")
    plt.title(name_identifier)
    plt.show()

# -----------------------------
# 1. Show heart image
# -----------------------------
show_image(heart_img, "Heart Image")

# -----------------------------
# 3. Invert colors
# -----------------------------
inverted_heart_img = 255 - heart_img
show_image(inverted_heart_img, "Inverted Heart Image")

# -----------------------------
# 4. Rotate heart (transpose)
# -----------------------------
rotated_heart_img = heart_img.T
show_image(rotated_heart_img, "Rotated Heart Image")

# -----------------------------
# 6. Random Image
# -----------------------------
random_img = np.random.randint(0, 255, (7, 7))
show_image(random_img, "Random Image")

# -----------------------------
# 7. Solve for x in random_img * x = heart_img
# -----------------------------
# Use np.linalg.solve if possible, or np.linalg.pinv for pseudo-inverse
x = np.linalg.pinv(random_img) @ heart_img
show_image(x, "x")

# -----------------------------
# 8. Reconstruct heart image
# -----------------------------
solved_heart_img = random_img @ x
show_image(solved_heart_img, "Solved Heart Image")
