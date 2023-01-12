#######################################################################
#  TARDIS - Transformer And Rapid Dimensionless Instance Segmentation #
#                                                                     #
#  New York Structural Biology Center                                 #
#  Simons Machine Learning Center                                     #
#                                                                     #
#  Robert Kiewisz, Tristan Bepler                                     #
#  MIT License 2021 - 2023                                            #
#######################################################################

from click.testing import CliRunner


def test_cnn_trainer():
    from tardis.train_spindletorch import main

    runner = CliRunner()
    result = runner.invoke(main, ['--version'])
    assert str(result) == '<Result okay>'


def test_cnn_predictor():
    from tardis.predict_spindletorch import main

    runner = CliRunner()
    result = runner.invoke(main, ['--version'])
    assert str(result) == '<Result okay>'


def test_predictor_mt():
    from tardis.predict_mt import main

    runner = CliRunner()
    result = runner.invoke(main, ['--version'])
    assert str(result) == '<Result okay>'


def test_predictor_cryo_mt():
    from tardis.predict_cro_mt import main

    runner = CliRunner()
    result = runner.invoke(main, ['--version'])
    assert str(result) == '<Result okay>'

def test_predictor_mem():
    from tardis.predict_mem import main

    runner = CliRunner()
    result = runner.invoke(main, ['--version'])
    assert str(result) == '<Result okay>'


def test_gf_trainer():
    from tardis.train_DIST import main

    runner = CliRunner()
    result = runner.invoke(main, ['--version'])
    assert str(result) == '<Result okay>'
