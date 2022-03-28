import matplotlib.pyplot as plt
from matplotlib import gridspec
from scipy.ndimage import gaussian_filter


def plot_images(img, mask):
    """
    Definition module to visualized selected slice of dataset

    Args:

        img: numpy array of ET image
        mask: numpy array of ET image mask

    Returns:
        matplotlib plot

    :author Robert Kiewisz
    """

    assert img.shape == mask.shape
    assert img.ndim == 2

    classes = mask.shape[2] if len(mask.shape) > 2 else 1
    fig, ax = plt.subplots(1, classes + 1)

    ax[0].set_title('Input image')
    ax[0].imshow(img)

    if classes > 1:
        for i in range(classes):
            ax[i + 1].set_title(f'Output mask (class {i + 1})')
            ax[i + 1].imshow(mask[:, :, i])
    else:
        ax[1].set_title(r'Output mask')
        ax[1].imshow(mask)

    plt.xticks([]), plt.yticks([])
    plt.show()


def plot_training(training_losses: list,
                  validation_losses: list,
                  learning_rate: list,
                  accuracy: list,
                  gaussian=True,
                  sigma=2,
                  fig_size=(8, 6)):
    """
    Definition to build plot containing training and validation losses

    Args:
        training_losses: list of all training losses after epoch
        validation_losses: list of all validation losses after epoch
        learning_rate: learning rate of the network
        accuracy: list of the accuracy from predictions
        gaussian: use gaussian filter to plot regression
        sigma: sigma value for gaussian filter
        fig_size: output size of the figure

    Returns:
        matplotlib plot
    """

    list_len = len(training_losses)
    x_range = list(range(1, list_len + 1))  # number of x values
    accuracy_range = list(range(0, len(accuracy)))

    fig = plt.figure(figsize=fig_size)
    grid = gridspec.GridSpec(ncols=2,
                             nrows=2,
                             figure=fig)

    sub_fig1 = fig.add_subplot(grid[0, 0])
    sub_fig2 = fig.add_subplot(grid[0, 1])
    sub_fig3 = fig.add_subplot(grid[1, 0])

    sub_figures = fig.get_axes()

    for i, sub_fig in enumerate(sub_figures, start=1):
        sub_fig.spines['top'].set_visible(False)
        sub_fig.spines['right'].set_visible(False)

    if gaussian:
        training_losses_gauss = gaussian_filter(training_losses,
                                                sigma=sigma)
        validation_losses_gauss = gaussian_filter(validation_losses,
                                                  sigma=sigma)
        accuracy_gauss = gaussian_filter(accuracy,
                                         sigma=sigma)
        alpha = 0.25
    else:
        alpha = 1.0

    linestyle_original = "-"
    color_original_train = 'red'
    color_original_valid = 'green'
    color_original_accuracy = 'blue'

    # Subfig 1
    if gaussian:
        sub_fig1.plot(x_range, training_losses_gauss,
                      linestyle_original,
                      color=color_original_train,
                      label='Training',
                      alpha=0.75)
        sub_fig1.plot(x_range, validation_losses_gauss,
                      linestyle_original,
                      color=color_original_valid,
                      label='Validation',
                      alpha=0.75)
    else:
        sub_fig1.plot(x_range, training_losses,
                      linestyle_original,
                      color=color_original_train,
                      label='Training',
                      alpha=alpha)
        sub_fig1.plot(x_range, validation_losses,
                      linestyle_original,
                      color=color_original_valid,
                      label='Validation',
                      alpha=alpha)

    sub_fig1.title.set_text('Training & validation loss')
    sub_fig1.set_xlabel('Epoch')
    sub_fig1.set_ylabel('Loss')

    sub_fig1.legend(loc='upper right')

    # Subfig 2
    sub_fig2.plot(x_range, learning_rate,
                  color='black')
    sub_fig2.title.set_text('Learning rate')
    sub_fig2.set_xlabel('Epoch')
    sub_fig2.set_ylabel('LR')

    # Subfig 3
    if gaussian:
        sub_fig3.plot(accuracy_range, accuracy_gauss,
                      linestyle_original,
                      color=color_original_accuracy,
                      alpha=alpha)
    else:
        sub_fig3.plot(accuracy_range, accuracy,
                      linestyle_original,
                      color=color_original_accuracy,
                      alpha=alpha)
    sub_fig3.title.set_text('Prediction accuracy')
    sub_fig3.set_xlabel('Epoch')
    sub_fig3.set_ylabel('Accuracy')

    return fig
