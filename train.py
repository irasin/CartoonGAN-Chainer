import argparse
import chainer
from chainer import training, serializers
from chainer.training import extensions
from model import Generator, Discriminator, VGG
from dataset import PhotoDataset, ImageDataset
from updater import InitializationUpdater, CartoonGANUpdater
from visualizaion import out_generated_image


def main():
    parser = argparse.ArgumentParser(description='Chainer: CartoonGAN')
    parser.add_argument('--batchsize', '-b', type=int, default=16,
                        help='Number of images in each mini-batch')
    parser.add_argument('--epoch', '-e', type=int, default=1000,
                        help='Number of sweeps over the dataset to train')
    parser.add_argument('--gpu', '-g', type=int, default=-1,
                        help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--out', '-o', default='result',
                        help='Directory to output the result')
    parser.add_argument('--snapshot_interval', type=int, default=100,
                        help='Interval of snapshot')
    parser.add_argument('--display_interval', type=int, default=10,
                        help='Interval of displaying log to console')
    parser.add_argument('--w', type=int, default=10,
                        help='weight for content_loss')
    parser.add_argument('--photo_dir', type=str, default='/data/chen/DIV2K_train_HR/*.png',
                        help='train photo dir')
    parser.add_argument('--image_dir', type=str, default='/data/chen/anime/*.PNG',
                        help='train image dir')
    parser.add_argument('--val_dir', type=str, default='/data/chen/DIV2K_train_HR/*.png',
                        help='val photo dir')
    parser.add_argument('--crop_size', type=int, default=224,
                        help='crop_size for dataset')
    args = parser.parse_args()

    print('GPU: {}'.format(args.gpu))
    print('# Minibatch-size: {}'.format(args.batchsize))
    print('# epoch: {}'.format(args.epoch))
    print('')

    # setup model
    gen = Generator()
    dis = Discriminator()
    vgg = VGG()

    # if necessary, start from gen_iter_XX.npz
    # serializers.load_npz('gen_iter_100.npz', gen)

    if args.gpu >= 0:
        chainer.cuda.get_device_from_id(args.gpu).use()
        gen.to_gpu()
        dis.to_gpu()
        vgg.to_gpu()

    # Setup an optimizer
    def make_optimizer(model, alpha=0.0002, beta1=0.5, beta2=0.999):
        optimizer = chainer.optimizers.Adam(alpha=alpha, beta1=beta1, beta2=beta2)
        optimizer.setup(model)
        return optimizer

    opt_gen = make_optimizer(gen)
    opt_dis = make_optimizer(dis)

    # setup dataset
    photo = PhotoDataset(args.photo_dir, crop_size=args.crop_size)
    photo_iter = chainer.iterators.SerialIterator(photo, args.batchsize)
    image = ImageDataset(args.image_dir, crop_size=args.crop_size)
    image_iter = chainer.iterators.SerialIterator(image, args.batchsize)
    val = PhotoDataset(args.val_dir, crop_size=args.crop_size)
    val_iter = chainer.iterators.SerialIterator(val, args.batchsize)

    # setup updater
    # updater = InitializationUpdater(
    #     model=[gen, vgg],
    #     w=args.w,
    #     iterator={'main': photo_iter},
    #     optimizer={'gen': opt_gen},
    #     device=args.gpu,
    # )

    updater = CartoonGANUpdater(
        model=[gen, dis, vgg],
        w=args.w,
        iterator={'main': photo_iter, 'image': image_iter},
        optimizer={'gen': opt_gen, 'dis': opt_dis},
        device=args.gpu,
    )

    # setup trainer
    trainer = training.Trainer(updater, (args.epoch, 'epoch'), out=args.out)
    snapshot_interval = (args.snapshot_interval, 'iteration')
    display_interval = (args.display_interval, 'iteration')
    # trainer.extend(extensions.snapshot(filename='snapshot_iter_{.updater.epoch}.npz'), trigger=snapshot_interval)
    trainer.extend(extensions.snapshot_object(gen, 'gen_iter_{.updater.epoch}.npz'), trigger=snapshot_interval)
    trainer.extend(extensions.LogReport(trigger=display_interval))
    trainer.extend(extensions.PrintReport(['epoch', 'iteration', 'dis_loss', 'vgg_loss', 'gen_loss']),
                   trigger=display_interval)
    trainer.extend(extensions.ProgressBar(update_interval=args.display_interval / 10))
    trainer.extend(out_generated_image(gen, 4, 4, val_iter, args.out), trigger=snapshot_interval)

    # Run the trainer
    trainer.run()


if __name__ == '__main__':
    main()
