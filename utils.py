import cv2
def resize_image(image, width=None, height=None, inter=cv2.INTER_AREA):
    # check if the width and height is specified
    if width is None and height is None:
        return image

    # initialize the dimension of the image and grab the
    # width and height of the image
    dimension = None
    (h, w) = image.shape[:2]

    # calculate the ratio of the height and
    # construct the new dimension
    if height is not None:
        ratio = height / float(h)
        dimension = (int(w * ratio), height)
    else:
        ratio = width / float(w)
        dimension = (width, int(h * ratio))

    # resize the image
    resized_image = cv2.resize(image, dimension, interpolation=inter)

    return resized_image