import glob
from random import randint
import numpy as np
import cv2
import chainer


class PreprocessedDataset(chainer.dataset.DatasetMixin):
    def __init__(self, path, crop_size):
        self._path = glob.glob(path)
        self._crop_size = crop_size
        self._length = len(self._path)

    def __len__(self):
        return self._length

    def get_example(self, i):
        raise NotImplementedError

    @staticmethod
    def tochainer(cv2_image):
        return cv2_image.astype(np.float32).transpose((2, 0, 1)) 


class PhotoDataset(PreprocessedDataset):
    def __init__(self, path, crop_size=224):
        super().__init__(path, crop_size)

    def get_example(self, i):
        while True:
            try:
                image = cv2.imread(self._path[i], cv2.IMREAD_COLOR)
                assert image is not None
                h, w, _ = image.shape
                assert (w >= self._crop_size and h >= self._crop_size)
                break
            except AssertionError:
                i = (i + np.random.randint(1, self._length)) % self._length
        crop_w = randint(0, w - self._crop_size)
        crop_h = randint(0, h - self._crop_size)
        image = image[crop_h: crop_h + self._crop_size, crop_w: crop_w + self._crop_size]
        return self.tochainer(image)


class ImageDataset(PreprocessedDataset):
    def __init__(self, path, crop_size=224):
        super().__init__(path, crop_size)

    def get_example(self, i):
        while True:
            try:
                image = cv2.imread(self._path[i], cv2.IMREAD_COLOR)
                assert image is not None
                h, w, _ = image.shape
                assert (w >= self._crop_size and h >= self._crop_size)
                break
            except AssertionError:
                i = (i + np.random.randint(1, self._length)) % self._length
        crop_w = randint(0, w - self._crop_size)
        crop_h = randint(0, h - self._crop_size)
        image = image[crop_h: crop_h + self._crop_size, crop_w: crop_w + self._crop_size]
        smoothed = self.blur(image)
        return self.tochainer(image), self.tochainer(smoothed)

    @staticmethod
    def blur(image):
        # detect edge
        edge = cv2.Canny(image, np.random.randint(75, 125), np.random.randint(175, 225))

        # dilate edge
        d_size = np.random.randint(2, 5)
        dilated_edge = cv2.dilate(edge, np.ones((d_size, d_size), np.uint8), iterations=1)

        # gray to bgr
        dilated_edge = cv2.cvtColor(dilated_edge, cv2.COLOR_GRAY2BGR)

        # blur and merge
        b_size = np.random.randint(1, 3) * 2 + 1
        gaussian_smoothing_image = cv2.GaussianBlur(image, (b_size, b_size), 0)
        merged = np.where(dilated_edge, gaussian_smoothing_image, image)
        return merged






