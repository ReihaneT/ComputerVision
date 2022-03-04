import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines


def line_detection_hough(image, edge_image, threshold=30):
  height, width = np.shape(edge_image)
  height_half, width_half = height / 2, width / 2
  d = np.sqrt(height**2 + width**2)
  thetas = np.arange(0, 180, step=1)
  rhos = np.arange(-d, d, step=1)
  cos_thetas = np.cos(np.deg2rad(thetas))
  sin_thetas = np.sin(np.deg2rad(thetas))
  matrix = np.zeros((len(rhos), len(thetas)))
  cv2.imshow("original image", image)
  cv2.imshow("edge",edge_image)
  figure = plt.figure(figsize=(12, 12))

  subplot1 = figure.add_subplot(1, 2, 1)
  subplot1.set_facecolor((0, 0, 0))
  subplot2 = figure.add_subplot(1, 2, 2)
  subplot2.imshow(image)
  for y in range(height):
    for x in range(width):
      if edge_image[y][x] != 0:
        edge_point = [y - height_half, x - width_half]
        ys, xs = [], []
        for theta_idx in range(len(thetas)):
          rho = (edge_point[1] * cos_thetas[theta_idx]) + (edge_point[0] * sin_thetas[theta_idx])
          theta = thetas[theta_idx]
          a=np.abs(rhos - rho)
          rho_idx = np.argmin(a)
          matrix[rho_idx][theta_idx] += 1
          ys.append(rho)
          xs.append(theta)
        subplot1.plot(xs, ys, color="red", alpha=0.05)

  for y in range(matrix.shape[0]):
    for x in range(matrix.shape[1]):
      if matrix[y][x] > threshold:
        rho = rhos[y]
        theta = thetas[x]
        a = np.cos(np.deg2rad(theta))
        b = np.sin(np.deg2rad(theta))
        x0 = (a * rho) + width_half
        y0 = (b * rho) + height_half
        x1 = int(x0 + 1000 * (-b))
        y1 = int(y0 + 1000 * (a))
        x2 = int(x0 - 1000 * (-b))
        y2 = int(y0 - 1000 * (a))
        subplot1.plot([theta], [rho], marker='*', color="white")
        subplot2.add_line(mlines.Line2D([x1, x2], [y1, y2]))
  subplot1.invert_yaxis()
  subplot1.invert_xaxis()

  subplot1.title.set_text("Hough Space")
  subplot2.title.set_text("Detected Lines")
  plt.show()




image = cv2.imread("hough1.png")
edge_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
edge_image = cv2.GaussianBlur(edge_image, (3, 3), 1)
edge_image = cv2.Canny(edge_image, 73, 150)
line_detection_hough(image, edge_image)
cv2.waitKey()