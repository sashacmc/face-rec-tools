import cv2


def read_image(image_file, max_size):
    image = cv2.imread(image_file)

    height, width, col = image.shape

    if height > width:
        scale = max_size / height
    else:
        scale = max_size / width

    if scale < 1:
        image = cv2.resize(image, (int(width * scale), int(height * scale)))

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    return image
