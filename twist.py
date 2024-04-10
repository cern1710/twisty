import cv2
import numpy as np
import time

def twist(img_path, strength=1.0, radius=100,
          center=(138,171.6), decay_scale=0.9):
    """Applies a swirl/twirl effect to image with fine-tuning parameters.

    Parameters:
      img_path: Path to the input image.
      strength: Intensity of the twist effect.
      radius: Effective radius of the twist effect.
      center: Tuple (x, y) indicating the center of the twist effect.
      decay_scale: Scaling factor for the decay function.
    """
    img = cv2.imread(img_path)
    h, w, _ = img.shape
    cx, cy = center if center else ((w // 2), (h // 2))

    # Generate coord grids and find their distances and angles from the center
    xx, yy = np.meshgrid(np.arange(w), np.arange(h))
    r = np.sqrt((xx - cx)**2 + (yy - cy)**2)
    theta = np.arctan2(yy - cy, xx - cx)

    # Quadratic decay to center
    decay = ((radius - r * decay_scale) / radius)**2

    # Apply swirl / twirl effect
    theta_new = theta + strength * decay
    theta_new[r > radius] = theta[r > radius]

    # Convert polar coordinates back to Cartesian and apply map coordinates
    xx_new = r * np.cos(theta_new) + cx
    yy_new = r * np.sin(theta_new) + cy
    map_x = np.clip(xx_new, 0, w - 1).astype(np.float32)
    map_y = np.clip(yy_new, 0, h - 1).astype(np.float32)

    # Remap the image pixels
    twisted_img = cv2.remap(img, map_x, map_y, cv2.INTER_CUBIC)
    return twisted_img

if __name__ == "__main__":
    img = twist('face.jpeg', strength=17.05, radius=140,
                center=(139.9, 172.0),
                decay_scale=0.995)
    cv2.imwrite('unswirled_face.jpeg', img)