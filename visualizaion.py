import os
import numpy as np
import cv2
import chainer


def out_generated_image(gen, rows, cols, val_iter, out):
    @chainer.training.make_extension()
    def make_image(trainer):
        photo = val_iter.next()
        _photo = np.asarray(photo)
        # _photo = 255 * np.asarray(photo)
        photo = trainer.updater.converter(photo, trainer.updater.device)
        with chainer.using_config('train', False):
            image_gen = gen(photo)
        # photo = chainer.cuda.to_cpu(photo.data)
        image_gen = chainer.cuda.to_cpu(image_gen.data)

        photo = _photo.astype(np.uint8)
        _, _, H, W = photo.shape
        photo = photo.reshape((rows, cols, 3, H, W))
        photo = photo.transpose(0, 3, 1, 4, 2)
        photo = photo.reshape((rows * H, cols * W, 3))

        image_gen = np.asarray(np.clip(image_gen, 0.0, 255.0), dtype=np.uint8)
        # image_gen =  np.asarray(np.clip(image_gen * 255, 0.0, 255.0), dtype=np.uint8)
        _, _, H, W = image_gen.shape
        image_gen = image_gen.reshape((rows, cols, 3, H, W))
        image_gen = image_gen.transpose(0, 3, 1, 4, 2)
        image_gen = image_gen.reshape((rows * H, cols * W, 3))

        preview_dir = f'{out}/preview'
        preview_photo_path = preview_dir + f'/photo_{trainer.updater.iteration:0>8}.png'
        preview_gen_path = preview_dir + f'/gen_{trainer.updater.iteration:0>8}.png'
        if not os.path.exists(preview_dir):
            os.makedirs(preview_dir)
        cv2.imwrite(preview_photo_path, photo)
        cv2.imwrite(preview_gen_path, image_gen)
    return make_image
