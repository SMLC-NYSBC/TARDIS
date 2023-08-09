#######################################################################
#  TARDIS - Transformer And Rapid Dimensionless Instance Segmentation #
#                                                                     #
#  New York Structural Biology Center                                 #
#  Simons Machine Learning Center                                     #
#                                                                     #
#  Robert Kiewisz, Tristan Bepler                                     #
#  MIT License 2021 - 2023                                            #
#######################################################################
from warnings import filterwarnings

from click.testing import CliRunner


def test_cnn_trainer():
    filterwarnings(action="ignore", category=DeprecationWarning)
    from tardis_pytorch.train_spindletorch import main

    runner = CliRunner()
    result = runner.invoke(main, ["--version"])
    assert str(result) == "<Result okay>"


def test_predictor_mt():
    from tardis_pytorch.predict_mt import main

    filterwarnings(action="ignore", category=DeprecationWarning)

    runner = CliRunner()
    result = runner.invoke(main, ["--version"])
    assert str(result) == "<Result okay>"


def test_compare_mt():
    from tardis_pytorch.compare_spatial_graphs import main

    filterwarnings(action="ignore", category=DeprecationWarning)

    runner = CliRunner()
    result = runner.invoke(main, ["--version"])
    assert str(result) == "<Result okay>"


def test_predictor_cryo_mt():
    from tardis_pytorch.predict_cro_mt import main

    filterwarnings(action="ignore", category=DeprecationWarning)

    runner = CliRunner()
    result = runner.invoke(main, ["--version"])
    assert str(result) == "<Result okay>"


def test_predictor_mem():
    from tardis_pytorch.predict_mem import main

    runner = CliRunner()
    result = runner.invoke(main, ["--version"])
    assert str(result) == "<Result okay>"


def test_gf_trainer():
    from tardis_pytorch.train_DIST import main

    filterwarnings(action="ignore", category=DeprecationWarning)

    runner = CliRunner()
    result = runner.invoke(main, ["--version"])
    assert str(result) == "<Result okay>"
