import cv2

def read_images():
    original_image = cv2.imread("samples/herseyim.bmp")
    original_imagebaby = cv2.imread("samples/baby.bmp")
    original_image1 = cv2.imread("samples/herseyimmarked.bmp")

    cv2.imwrite("samples/herseyim.bmp",original_image)
    cv2.imwrite("samples/herseyim_marked.bmp",original_image1)

    print(original_image.shape)
    print(original_imagebaby.shape)

if __name__ == '__main__':
    read_images()