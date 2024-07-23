#######################################################################
#  TARDIS - Transformer And Rapid Dimensionless Instance Segmentation #
#                                                                     #
#  New York Structural Biology Center                                 #
#  Simons Machine Learning Center                                     #
#                                                                     #
#  Robert Kiewisz, Tristan Bepler                                     #
#  MIT License 2021 - 2024                                            #
#######################################################################
import numpy as np
import scipy
import torch

from skimage import exposure


def adaptive_threshold(img: np.ndarray):
    if img.ndim == 3:
        std_ = np.std(img, axis=(1, 2), keepdims=True)
        mean_ = np.mean(img, axis=(1, 2), keepdims=True)
    else:
        std_ = np.std(img)
        mean_ = np.mean(img)

    # Perform the thresholding in a vectorized manner
    return (img >= (mean_ / std_)).astype(np.uint8)


class SimpleNormalize:
    """
    SIMPLE IMAGE NORMALIZATION

    Take int8-int32 image file with 0 - 255 value. All image value are spread
    between 0 - 1.
    """

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """
        Call for image normalization.

        Args:
            x (np.ndarray): Image array.

        Returns:
            np.ndarray: Normalized image.
        """
        if x.dtype == np.uint8:
            x = x / 255
        elif x.dtype == np.int8:
            x = (x + 128) / 255
        elif x.dtype == np.uint16:
            x = x / 65535
        elif x.dtype == np.int16:
            x = (x + 32768) / 65535
        elif x.dtype == np.uint32:
            x = x / 4294967295
        else:
            x = (x + 2147483648) / 4294967295

        return x.astype(np.float32)


class MinMaxNormalize:
    """
    IMAGE NORMALIZATION BETWEEN MIN AND MAX VALUE
    """

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """
        Call for normalization.

        Args:
            x (np.ndarray): Image data.

        Returns:
            np.ndarray: Normalized array.
        """
        MIN = np.min(x)
        MAX = np.max(x)

        if MAX <= 0:
            x = x + abs(MIN)
        x = (x - MIN) / (MAX - MIN)

        return x.astype(np.float32)


class MeanStdNormalize:
    """
    IMAGE NORMALIZATION BASED ON MEAN AND STD
    """

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """
        Call for standardization.

        Args:
            x (np.ndarray): Image data.

        Returns:
            np.ndarray: Standardized array.
        """
        MEAN = float(x.mean())
        STD = float(x.std())

        x = (x - MEAN) / STD
        return x.astype(np.float32)


class RescaleNormalize:
    """
    NORMALIZE IMAGE VALUE USING Skimage

    Rescale intensity with top% and bottom% percentiles as default

    Args:
        clip_range: Histogram percentiles range crop.
    """

    def __init__(self, clip_range=(2, 98)):
        self.range = clip_range

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """
        Call for normalization.

        Args:
            x (np.ndarray): Image or label mask.

        Returns:
            np.ndarray: Normalized array.
        """
        p2, p98 = np.percentile(x, self.range)
        if x.dtype == np.uint8:
            if p98 >= 250:
                p98 = 256
            if p2 <= 5:
                p2 = 0

        return exposure.rescale_intensity(x, in_range=(p2, p98))


class FFTNormalize:
    def __init__(
        self,
        method="affine",
        alpha=900,
        beta_=1,
        num_iters=100,
        sample=1,
        use_cuda=False,
    ):
        self.method = method
        self.alpha = alpha
        self.beta = beta_
        self.num_iters = num_iters
        self.sample = sample
        self.cuda = use_cuda

        self.mean_std = MeanStdNormalize()

    @staticmethod
    def gmm_fit(
        x,
        pi=0.5,
        split=None,
        alpha=0.5,
        beta_=0.5,
        scale=1.0,
        tol=1e-3,
        num_iters=100,
        share_var=True,
        verbose=False,
    ):
        # fit 2-component GMM
        # put a beta distribution prior on pi

        mu = torch.mean(x)
        pi = torch.as_tensor(pi)

        # split into everything > and everything <= pi pixel value
        # for assigning initial parameters
        if split is None:
            split = np.quantile(x.cpu().numpy(), 1 - pi)
        mask = x <= split

        p0 = mask.float()
        p1 = 1 - p0

        mu0 = mu
        s = torch.sum(p0)
        if s > 0:
            mu0 = torch.sum(x * p0) / s

        mu1 = mu
        s = torch.sum(p1)
        if s > 0:
            mu1 = torch.sum(x * p1) / s

        if share_var:
            var = torch.mean(p0 * (x - mu0) ** 2 + p1 * (x - mu1) ** 2)
            var0 = var
            var1 = var
        else:
            var0 = torch.sum(p0 * (x - mu0) ** 2) / torch.sum(p0)
            var1 = torch.sum(p1 * (x - mu1) ** 2) / torch.sum(p1)

        # first, calculate p(k | x, mu, var, pi)
        log_p0 = (
            -((x - mu0) ** 2) / 2 / var0
            - 0.5 * torch.log(2 * np.pi * var0)
            + torch.log1p(-pi)
        )
        log_p1 = (
            -((x - mu1) ** 2) / 2 / var1
            - 0.5 * torch.log(2 * np.pi * var1)
            + torch.log(pi)
        )

        ma = torch.max(log_p0, log_p1)
        Z = ma + torch.log(torch.exp(log_p0 - ma) + torch.exp(log_p1 - ma))

        # the probability of the data is
        logp = scale * torch.sum(Z) + scipy.stats.beta.logpdf(
            pi.cpu().numpy(), alpha, beta_
        )
        logp_cur = logp

        for it in range(1, num_iters + 1):
            # calculate the assignments
            p0 = torch.exp(log_p0 - Z)
            p1 = torch.exp(log_p1 - Z)

            # now, update distribution parameters
            s = torch.sum(p1)
            a = alpha + s
            b = beta_ + p1.numel() - s
            pi = (a - 1) / (a + b - 2)  # MAP estimate of pi

            mu0 = mu
            s = torch.sum(p0)
            if s > 0:
                mu0 = torch.sum(x * p0) / s

            mu1 = mu
            s = torch.sum(p1)
            if s > 0:
                mu1 = torch.sum(x * p1) / s

            if share_var:
                var = torch.mean(p0 * (x - mu0) ** 2 + p1 * (x - mu1) ** 2)
                var0 = var
                var1 = var
            else:
                var0 = torch.sum(p0 * (x - mu0) ** 2) / torch.sum(p0)
                var1 = torch.sum(p1 * (x - mu1) ** 2) / torch.sum(p1)

            # recalculate likelihood p(k | x, mu, var, pi)
            log_p0 = (
                -((x - mu0) ** 2) / 2 / var0
                - 0.5 * torch.log(2 * np.pi * var0)
                + torch.log1p(-pi)
            )
            log_p1 = (
                -((x - mu1) ** 2) / 2 / var1
                - 0.5 * torch.log(2 * np.pi * var1)
                + torch.log(pi)
            )

            ma = torch.max(log_p0, log_p1)
            Z = ma + torch.log(torch.exp(log_p0 - ma) + torch.exp(log_p1 - ma))

            logp = scale * torch.sum(Z) + scipy.stats.beta.logpdf(
                pi.cpu().numpy(), alpha, beta_
            )

            if verbose:
                print(it, logp)

            # check for termination
            if logp - logp_cur <= tol:
                break  # done
            logp_cur = logp

        return logp, mu0, mu1, var1

    def norm_fit(self, x, alpha=900, beta_=1, scale=1.0, num_iters=100, use_cuda=False):
        # try multiple initializations of pi
        pis = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.98, 1])
        splits = np.quantile(x, 1 - pis)

        logps = np.zeros(len(pis))
        mus = np.zeros(len(pis))
        stds = np.zeros(len(pis))

        x = torch.from_numpy(x)
        if use_cuda:
            x = x.cuda()

        for i in range(len(pis)):
            pi = float(pis[i])
            split = splits[i]
            if pi == 1:  # single component model
                mu = x.mean()
                var = x.var()
                logp = scale * torch.sum(
                    -((x - mu) ** 2) / 2 / var - 0.5 * torch.log(2 * np.pi * var)
                ) + scipy.stats.beta.pdf(1, alpha, beta_)
            else:
                logp, mu0, mu, var = self.gmm_fit(
                    x,
                    pi=pi,
                    split=split,
                    alpha=alpha,
                    beta_=beta_,
                    scale=scale,
                    num_iters=num_iters,
                )

            logps[i] = logp.item()
            mus[i] = mu.item()
            stds[i] = np.sqrt(var.item())

        # select normalization parameters with maximum logp
        i = np.argmax(logps)
        return mus[i], stds[i]

    def __call__(self, x: np.ndarray):
        if self.method == "affine":
            return self.mean_std(x)

        if self.sample > 1:
            n = int(np.round(x.size / self.sample))
            scale = x.size / n
            x_sample = np.random.choice(x.flatten(), size=n, replace=False)

            mu, std = self.norm_fit(
                x_sample,
                alpha=self.alpha,
                beta_=self.beta,
                scale=scale,
                num_iters=self.num_iters,
                use_cuda=self.cuda,
            )
        else:
            mu, std = self.norm_fit(
                x,
                alpha=self.alpha,
                beta_=self.beta,
                scale=1,
                num_iters=self.num_iters,
                use_cuda=self.cuda,
            )

        x = (x - mu) / std
        x = x.astype(np.float32)

        return x
