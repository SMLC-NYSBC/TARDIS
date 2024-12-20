#######################################################################
#  TARDIS - Transformer And Rapid Dimensionless Instance Segmentation #
#                                                                     #
#  New York Structural Biology Center                                 #
#  Simons Machine Learning Center                                     #
#                                                                     #
#  Robert Kiewisz, Tristan Bepler                                     #
#  MIT License 2021 - 2024                                            #
#######################################################################

from tardis_em.utils.losses import *

logits = torch.rand((1, 64, 64, 64))
targets = torch.softmax(logits, 0)


class GeneralLoss:
    """
    General class for testing loss functions
    """

    def __init__(self, logits_t: torch.Tensor, targets_t: torch.Tensor, criterion):
        self.logits = logits_t
        self.targets = targets_t
        self.criterion = criterion

    def test_loss(self) -> torch.Tensor:
        return self.criterion(self.logits, self.targets)


def test_adaptive_dice_loss():
    loss = GeneralLoss(logits, targets, AdaptiveDiceLoss())
    loss = loss.test_loss()

    assert isinstance(loss.data, torch.Tensor)
    assert loss > 0

    loss = GeneralLoss(logits, targets, AdaptiveDiceLoss(diagonal=True))
    loss = loss.test_loss()

    assert isinstance(loss.data, torch.Tensor)
    assert loss > 0


def test_bce_loss():
    loss = GeneralLoss(logits, targets, BCELoss())
    loss = loss.test_loss()

    assert isinstance(loss.data, torch.Tensor)
    assert loss > 0

    loss = GeneralLoss(logits, targets, BCELoss(diagonal=True))
    loss = loss.test_loss()

    assert isinstance(loss.data, torch.Tensor)
    assert loss > 0


def test_bce_dice():
    loss = GeneralLoss(logits, targets, BCEDiceLoss())
    loss = loss.test_loss()

    assert isinstance(loss.data, torch.Tensor)
    assert loss > 0

    loss = GeneralLoss(logits, targets, BCEDiceLoss(diagonal=True))
    loss = loss.test_loss()

    assert isinstance(loss.data, torch.Tensor)
    assert loss > 0


def test_ce():
    logits_ = torch.rand((1, 2, 64, 64, 64))
    targets_ = torch.softmax(logits_, 1)

    loss = GeneralLoss(logits_, targets_, CELoss())
    loss = loss.test_loss()

    assert isinstance(loss.data, torch.Tensor)
    assert loss > 0


def test_cl_bce():
    loss = GeneralLoss(logits, targets, ClBCELoss())
    loss = loss.test_loss()

    assert isinstance(loss.data, torch.Tensor)
    assert loss > 0

    loss = GeneralLoss(logits, targets, ClDiceLoss(diagonal=True))
    loss = loss.test_loss()

    assert isinstance(loss.data, torch.Tensor)
    assert loss > 0


def test_cl_dice():
    loss = GeneralLoss(logits, targets, ClDiceLoss())
    loss = loss.test_loss()

    assert isinstance(loss.data, torch.Tensor)
    assert loss > 0

    loss = GeneralLoss(logits, targets, ClDiceLoss(diagonal=True))
    loss = loss.test_loss()

    assert isinstance(loss.data, torch.Tensor)
    assert loss > 0


def test_dice():
    loss = GeneralLoss(logits, targets, DiceLoss())
    loss = loss.test_loss()

    assert isinstance(loss.data, torch.Tensor)
    assert loss > 0

    loss = GeneralLoss(logits, targets, DiceLoss(diagonal=True))
    loss = loss.test_loss()

    assert isinstance(loss.data, torch.Tensor)
    assert loss > 0


def test_sfl():
    loss = GeneralLoss(logits, targets, SigmoidFocalLoss())
    loss = loss.test_loss()

    assert isinstance(loss.data, torch.Tensor)
    assert loss > 0

    loss = GeneralLoss(logits, targets, SigmoidFocalLoss(diagonal=True))
    loss = loss.test_loss()

    assert isinstance(loss.data, torch.Tensor)
    assert loss > 0
