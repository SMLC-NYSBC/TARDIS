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
    """
    Perform adaptive thresholding on an image array using mean and standard deviation.

    This function calculates the standard deviation and mean of the input image.
    If the image is multichannel (e.g., RGB), it processes each channel independently.
    The threshold is determined by dividing the mean by the standard deviation.
    The resulting image is binarized by comparing pixel values to the thresholded
    value.

    :param img: Input image as a NumPy array. For multichannel images, each channel
        is processed independently.
    :type img: np.ndarray
    :return: A binarized image as a NumPy array with values set to 1 for pixels
        meeting the threshold criteria and 0 otherwise.
    :rtype: np.ndarray
    """
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
    SimpleNormalize class provides functionality for normalizing image data arrays to
    floating-point representations within the range of 0 to 1. The class aims to handle
    multiple types of integer-based image data, adjusting ranges based on the specific
    data type to ensure proper normalization.
    """

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """
        Applies normalization to the input NumPy array based on its data type. This is
        particularly useful for converting array values from various integer formats
        to a normalized floating-point range (0 to 1). The normalization process
        ensures consistency across different input data types by applying appropriate
        scaling and offset adjustments to the input values.

        :param x: Input NumPy array to be normalized. The normalization depends on
                  the data type of the array. Supported data types include
                  np.uint8, np.int8, np.uint16, np.int16, np.uint32, and np.int32.
                  Other data types are not handled explicitly.
        :return: A normalized NumPy array of type np.float32, where all values are scaled
                 to the range [0, 1].
        :rtype: np.ndarray
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
    Performs Min-Max normalization on input data.

    This class normalizes input data to the range [0, 1] by scaling the values
    proportionally within this range using the minimum and maximum values of
    the input array. If the maximum value is less than or equal to zero, the
    normalization adjusts the input data before scaling by adding the absolute
    value of the minimum.
    """

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """
        Normalizes the input numpy array to a range of [0, 1]. If the maximum value
        of the array is less than or equal to zero, the array is first shifted by
        adding the absolute value of its minimum. The normalization ensures that
        the minimum value becomes 0 and the maximum value becomes 1. The result
        is then cast to a float32 numpy array.

        :param x: Input numpy array to be normalized.
        :type x: np.ndarray
        :return: A numpy array normalized to the range [0, 1] with dtype np.float32.
        :rtype: np.ndarray
        """
        MIN = np.min(x)
        MAX = np.max(x)

        if MAX <= 0:
            x = x + abs(MIN)
        x = (x - MIN) / (MAX - MIN)

        return x.astype(np.float32)


class MeanStdNormalize:
    """
    A class for normalizing input data using mean and standard deviation.

    This class standardizes input data by centering it to have a mean of zero
    and scaling it to have a standard deviation of one. It is designed for use
    in preprocessing image data or other numerical datasets to improve
    performance in machine learning models.
    """

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """
        Normalize the input NumPy array to have zero mean and unit variance.

        This callable object normalizes the input NumPy array by subtracting its mean
        and dividing by its standard deviation to ensure that the transformed data
        has a mean of 0 and a standard deviation of 1. The result is returned as a
        NumPy array of type float32.

        :param x: Input NumPy array to be normalized.
        :type x: np.ndarray
        :return: A normalized NumPy array with zero mean and unit variance, cast to
            float32.
        :rtype: np.ndarray
        """
        MEAN = float(x.mean())
        STD = float(x.std())

        x = (x - MEAN) / STD
        return x.astype(np.float32)


class RescaleNormalize:
    """
    Performs rescale normalization on a given array.

    This class is designed to normalize image or label mask input based on the
    specified clipping range using a percentile-based approach. It is particularly
    useful for preprocessing image data in various computer vision tasks, ensuring
    that output values fall within a defined intensity range.
    """

    def __init__(self, clip_range=(2, 98)):
        self.range = clip_range

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """
        Normalize the pixel intensity values of an input numpy array by rescaling them
        to a specified range defined by percentiles. The operation ensures that the
        output values are adjusted to enhance the contrast within the given range.

        :param x: A numpy array representing the input image data to be normalized. Values
           should fall within the range that aligns with the data type of the array. For
           example, if `x` is of `np.uint8`, values are in the range [0, 255].
        :return: A numpy array with pixel intensity values normalized to the specified
           percentile range. The datatype and shape of the returned array will match the
           input array.
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
        beta_i=1,
        num_iters=100,
        sample=1,
        use_cuda=False,
    ):
        """
        Initializes an object with transformations and normalization options.

        The class constructor sets the parameters for defining the transformation
        method, coefficients for transformation weight and regulation, iteration
        count, sampling options, CUDA acceleration, and standard mean-variance
        normalization. These values influence the behavior and computational
        efficiency of the intended transformation or process.

        :param method: Transformation method. Default is "affine".
        :param alpha: Weighting factor for transformations.
        :param beta_i: Regularization factor.
        :param num_iters: Number of iterations to be performed.
        :param sample: Sampling rate or number of samples.
        :param use_cuda: Indicator to use CUDA (Boolean).
        """
        self.method = method
        self.alpha = alpha
        self.beta = beta_i
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
        beta_f=0.5,
        scale=1.0,
        tol=1e-3,
        num_iters=100,
        share_var=True,
        verbose=False,
    ):
        """
        Performs Expectation-Maximization (EM) fitting of a two-component Gaussian Mixture
        Model (GMM) with a Beta distribution prior on the mixing coefficient. This function
        estimates the parameters of the GMM (means, variances, and mixing coefficients) and
        returns the log-likelihood and other model parameters.

        The method begins with an initial parameter assignment based on a threshold or
        quantile cut-off and iteratively optimizes the model parameters using EM until the
        log-likelihood converges or the maximum number of iterations is reached.

        :param x: Input tensor for the data to be modeled.
        :param pi: Initial mixing coefficient for the Gaussian components. Defaults to 0.5.
        :param split: Threshold value for initializing component assignments. If None, defaults
            to a quantile-based value computed from the data.
        :param alpha: Parameter for the Beta distribution prior on the mixing coefficient.
            Defaults to 0.5.
        :param beta_f: Parameter for the Beta distribution prior on the mixing coefficient.
            Defaults to 0.5.
        :param scale: Scaling factor for the log-likelihood computation. Defaults to 1.0.
        :param tol: Convergence tolerance for the log-likelihood difference between iteration
            steps. Defaults to 1e-3.
        :param num_iters: Maximum number of iterations for the EM algorithm. Defaults to 100.
        :param share_var: Boolean indicating whether the Gaussian components share a single
            variance value. Defaults to True.
        :param verbose: Boolean to enable printing of the log-likelihood at each iteration.
            Defaults to False.

        :return: A tuple containing:
            1. Final log-likelihood of the fitted model.
            2. Estimated mean of the first Gaussian component.
            3. Estimated mean of the second Gaussian component.
            4. Estimated variance of the second Gaussian component.
        """
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
            pi.cpu().numpy(), alpha, beta_f
        )
        logp_cur = logp

        for it in range(1, num_iters + 1):
            # calculate the assignments
            p0 = torch.exp(log_p0 - Z)
            p1 = torch.exp(log_p1 - Z)

            # now, update distribution parameters
            s = torch.sum(p1)
            a = alpha + s
            b = beta_f + p1.numel() - s
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
                pi.cpu().numpy(), alpha, beta_f
            )

            if verbose:
                print(it, logp)

            # check for termination
            if logp - logp_cur <= tol:
                break  # done
            logp_cur = logp

        return logp, mu0, mu1, var1

    def norm_fit(
        self, x, alpha=900, beta_i=1, scale=1.0, num_iters=100, use_cuda=False
    ):
        """
        Fits a normalization model to the input data by iteratively evaluating different
        probability mixtures while maximizing a log-likelihood measure. The function tries
        multiple initializations of probabilities for Gaussian Mixture Models (GMM) and
        evaluates their effectiveness using a combination of beta-prior distribution and
        the data's log-likelihood. Optimization terminates when the initialization with the
        maximum likelihood is found.

        The function allows either single-component fitting (pi=1) or GMM-based multip-component
        fitting (pi<1). It incorporates optional GPU acceleration for computations and utilizes
        a blend of Torch and Scipy statistical tools.

        :param x: Input data as a 1-dimensional array or tensor.
        :type x: numpy.ndarray or torch.Tensor
        :param alpha: Shape parameter for the beta prior distribution.
        :type alpha: int
        :param beta_i: Second shape parameter for the beta prior distribution.
        :type beta_i: int
        :param scale: Scaling factor for the likelihood computation.
        :type scale: float
        :param num_iters: Number of iterations allowed for the GMM fitting process.
        :type num_iters: int
        :param use_cuda: Boolean flag to determine if computations should run on GPU.
        :type use_cuda: bool
        :return: Tuple containing the mean (mu) and standard deviation (std) of the best-fit
            normalization parameters optimized for maximum log-likelihood (-logp).
        :rtype: tuple[float, float]
        """
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
                ) + scipy.stats.beta.pdf(1, alpha, beta_i)
            else:
                logp, mu0, mu, var = self.gmm_fit(
                    x,
                    pi=pi,
                    split=split,
                    alpha=alpha,
                    beta_f=beta_i,
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
        """
        Processes the input numpy array `x` using a normalization technique based on the
        predefined method and parameters. The method performs either "affine" normalization
        or utilizes sampling and an iterative normalization-fitting process to compute
        the mean and standard deviation. Normalized results are returned as single-precision floats.

        :param x: The input numpy array to be normalized.
        :type x: numpy.ndarray
        :return: The normalized numpy array as single-precision floats.
        :rtype: numpy.ndarray
        """
        if self.method == "affine":
            return self.mean_std(x)

        if self.sample > 1:
            n = int(np.round(x.size / self.sample))
            scale = x.size / n
            x_sample = np.random.choice(x.flatten(), size=n, replace=False)

            mu, std = self.norm_fit(
                x_sample,
                alpha=self.alpha,
                beta_i=self.beta,
                scale=scale,
                num_iters=self.num_iters,
                use_cuda=self.cuda,
            )
        else:
            mu, std = self.norm_fit(
                x,
                alpha=self.alpha,
                beta_i=self.beta,
                scale=1,
                num_iters=self.num_iters,
                use_cuda=self.cuda,
            )

        x = (x - mu) / std
        x = x.astype(np.float32)

        return x
