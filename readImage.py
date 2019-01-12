import cv2 as cv
import numpy as np
from skimage.segmentation import slic
from skimage.segmentation import mark_boundaries
from skimage.util import img_as_float
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import color_conv


def read_images():
    original_image = cv.imread("samples/hair_res.bmp", 0)  # open colour image
    im = np.zeros((original_image.shape[0], original_image.shape[1], 3))
    im[:, :, 0] = original_image
    im[:, :, 1] = original_image
    im[:, :, 2] = original_image
    cv.imwrite("samples/hair_res1.bmp", im)
    np.set_printoptions(threshold=np.nan)

    segments = slic(im, n_segments=300, sigma=2)
    fig = plt.figure("Superpixels -- %d segments" % 300)
    ax = fig.add_subplot(1, 1, 1)
    print(segments)
    ax.imshow(mark_boundaries(im, segments))
    plt.axis("off")
    plt.show()
    feature_vec = np.zeros((len(np.unique(segments)), 3))  # creating feature vector
    yiq = color_conv.rgb2yiq(im)
    for (j, segValue) in enumerate(np.unique(segments)):
        y, i, q = cv.split(yiq)
        feature_vec[j][0] = np.mean(y[segments == segValue])  # adding r values
        feature_vec[j][1] = np.mean(i[segments == segValue])  # adding g values
        feature_vec[j][2] = np.mean(q[segments == segValue])  # adding b values

    plt.scatter(feature_vec[:, 0], feature_vec[:, 1], label='True Position')
    plt.show()

    clusters = KMeans(n_clusters=12, random_state=0).fit(feature_vec)
    print(len(clusters.labels_))
    for i in range(len(clusters.labels_)):
        for ii in range(segments.shape[0]):
            for ij in range(segments.shape[1]):
                if segments[ii, ij] == i:
                    segments[ii, ij] = clusters.labels_[i]

    plt.scatter(feature_vec[:, 0], feature_vec[:, 1], c=clusters.labels_, cmap='rainbow')
    print(segments)
    fig = plt.figure("Superpixels -- %d segments" % 300)
    ax = fig.add_subplot(1, 1, 1)
    ax.imshow(mark_boundaries(im, segments))
    plt.axis("off")
    plt.show()


if __name__ == '__main__':
    read_images()
