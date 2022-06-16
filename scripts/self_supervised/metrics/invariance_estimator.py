import torch


class InvarianceEstimator:
    r"""Given the class of transformations :obj:`transform` and a set of values :obj:`transform_values`,
    will generate augmented views from the sample, then compute their standard deviation in the representation space.
    Both the histogram and the mean will be logged to tensorboard.

    ..Example::
        ```
        >>> imgs, labels = next(iter(DataLoader(dataset, batch_size=1024, shuffle=True,
            worker_init_fn=lambda: np.random.seed(5)))) ##used to make sure it's the same samples everytime.
        >>> transform_class = lambda theta: augs.RandomAffine((theta, theta))
        >>> transform_values = np.deg2rad(np.array([45, 90, 135, 180]))
        >>> inv_est = InvarianceEstimator(transform_class, transform_values, writer)
        >>> inv_est(online_encoder, imgs, step)
        ```

    """
    def __init__(self, transform, transform_values, writer):
        self.transform = transform
        self.transform_values = transform_values
        self.writer = writer

    def __call__(self, net, imgs, step):
        # compute representation of imgs and their augmented views
        y_list = [self._forward(net, imgs)]
        for transform_value in self.transform_values:
            t = self.transform(transform_value)
            augmented = t(imgs)
            y_list.append(self._forward(net, augmented))
        y = torch.stack(y_list)  # num_augmentations x num_imgs x num_features
        y_std = y.std((0, 2))
        self.writer.add_histogram('invariance_estimations/y', y, step)
        self.writer.add_scalar('invariance_estimations/y_std', y_std.mean(), step)

    def _forward(self, net, imgs):
        y = net(imgs).detach()
        y = y.view(y.shape[0], -1)

        return y
