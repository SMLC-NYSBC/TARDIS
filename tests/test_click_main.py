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


def test_compare_spatial_graphs():
    from tardis_em.compare_spatial_graphs import main

    filterwarnings(action="ignore", category=DeprecationWarning)

    runner = CliRunner()
    result = runner.invoke(main, ["--version"])
    assert str(result) == "<Result okay>"


def test_predict_cro_mt():
    from tardis_em.predict_cro_mt import main

    filterwarnings(action="ignore", category=DeprecationWarning)

    runner = CliRunner()
    result = runner.invoke(main, ["--version"])
    assert str(result) == "<Result okay>"


def test_predict_filament():
    from tardis_em.predict_filament import main

    filterwarnings(action="ignore", category=DeprecationWarning)

    runner = CliRunner()
    result = runner.invoke(main, ["--version"])
    assert str(result) == "<Result okay>"


def test_predict_instances():
    from tardis_em.predict_instances import main

    filterwarnings(action="ignore", category=DeprecationWarning)

    runner = CliRunner()
    result = runner.invoke(main, ["--version"])
    assert str(result) == "<Result okay>"


def test_predict_mem():
    from tardis_em.predict_mem import main

    filterwarnings(action="ignore", category=DeprecationWarning)

    runner = CliRunner()
    result = runner.invoke(main, ["--version"])
    assert str(result) == "<Result okay>"


def test_predict_mem_2d():
    from tardis_em.predict_mem_2d import main

    filterwarnings(action="ignore", category=DeprecationWarning)

    runner = CliRunner()
    result = runner.invoke(main, ["--version"])
    assert str(result) == "<Result okay>"


def test_predict_mt():
    from tardis_em.predict_mt import main

    filterwarnings(action="ignore", category=DeprecationWarning)

    runner = CliRunner()
    result = runner.invoke(main, ["--version"])
    assert str(result) == "<Result okay>"


def test_tardist():
    from tardis_em.tardis import main

    filterwarnings(action="ignore", category=DeprecationWarning)

    runner = CliRunner()
    result = runner.invoke(main, ["--version"])
    assert str(result) == "<Result okay>"


def test_train_DIST():
    from tardis_em.train_DIST import main

    filterwarnings(action="ignore", category=DeprecationWarning)

    runner = CliRunner()
    result = runner.invoke(main, ["--version"])
    assert str(result) == "<Result okay>"


def test_train_spindletorch():
    from tardis_em.train_spindletorch import main

    filterwarnings(action="ignore", category=DeprecationWarning)

    runner = CliRunner()
    result = runner.invoke(main, ["--version"])
    assert str(result) == "<Result okay>"
