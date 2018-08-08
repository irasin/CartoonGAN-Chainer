import chainer
import chainer.functions as F


class InitializationUpdater(chainer.training.StandardUpdater):
    def __init__(self, *args, **kwargs):
        self.gen, self.vgg = kwargs.pop('model')
        self.w = kwargs.pop('w')
        super().__init__(*args, **kwargs)

    def update_core(self):
        opt_gen = self.get_optimizer('gen')

        def _update(optimizer, loss):
            optimizer.target.cleargrads()
            loss.backward()
            optimizer.update()

        photo = self.get_iterator('main').next()
        photo = self.converter(photo, self.device)
        image_gen = self.gen(photo)

        # vgg loss
        vgg_loss = F.mean_absolute_error(self.vgg(photo), self.vgg(image_gen))

        # gen will be updated only by vgg loss
        gen_loss = self.w * vgg_loss
        _update(opt_gen, gen_loss)

        # report loss
        chainer.report({'vgg_loss': vgg_loss, 'gen_loss': gen_loss})


class CartoonGANUpdater(chainer.training.StandardUpdater):
    def __init__(self, *args, **kwargs):
        self.gen, self.dis, self.vgg = kwargs.pop('model')
        self.w = kwargs.pop('w')
        super().__init__(*args, **kwargs)
        self.xp = self.gen.xp
        self.label_true = None
        self.label_false = None

    def make_labels(self, feature_map):
        self.label_true = self.xp.ones(feature_map.shape, dtype=self.xp.float32)
        self.label_false = self.xp.zeros(feature_map.shape, dtype=self.xp.float32)

    def update_core(self):
        opt_gen = self.get_optimizer('gen')
        opt_dis = self.get_optimizer('dis')

        def _update(optimizer, loss):
            optimizer.target.cleargrads()
            loss.backward()
            optimizer.update()

        if self.iteration < 1200:  # 2000:
            photo = self.get_iterator('main').next()
            photo = self.converter(photo, self.device)
            image_gen = self.gen(photo)

            # vgg loss
            vgg_loss = F.mean_absolute_error(self.vgg(photo), self.vgg(image_gen))

            # gen will be updated only by vgg loss
            gen_loss = self.w * vgg_loss
            _update(opt_gen, gen_loss)

            # report loss
            chainer.report({'vgg_loss': vgg_loss, 'gen_loss': gen_loss})
        else:
            photo = self.get_iterator('main').next()
            image_batch = self.get_iterator('image').next()
            image = [image for image, _ in image_batch]
            edge_smoothed = [edge_smoothed for _, edge_smoothed in image_batch]
            photo = self.converter(photo, self.device)
            image = self.converter(image, self.device)
            edge_smoothed = self.converter(edge_smoothed, self.device)
            image_gen = self.gen(photo)

            # update dis
            y_image_gen = self.dis(image_gen)
            y_image = self.dis(image)
            y_edge_smoothed = self.dis(edge_smoothed)

            # generate label for dis(only 1 time)
            if self.label_true is None:
                self.make_labels(y_image_gen)

            # dis cartoon image loss
            # loss1 = F.mean_squared_error(y_image, self.label_true)
            loss1 = F.mean_absolute_error(y_image, self.label_true)

            # dis edge_smoothed image loss
            loss2 = F.mean_absolute_error(y_edge_smoothed, self.label_false)

            # dis image_gen loss
            loss3 = F.mean_absolute_error(y_image_gen, self.label_false)

            # dis loss
            dis_loss = loss1 + loss2 + loss3

            # update dis
            _update(opt_dis, dis_loss)

            # update gen
            # gen image_gen loss
            loss4 = F.mean_absolute_error(y_image_gen, self.label_true)

            # gen content loss
            loss5 = F.mean_absolute_error(self.vgg(photo), self.vgg(image_gen))

            # gen loss
            gen_loss = self.w * loss5 + loss4 
            _update(opt_gen, gen_loss)

            # report loss
            chainer.report({'dis_loss': dis_loss, 'gen_loss': gen_loss, 'vgg_loss': loss5})
