"""
TARDIS - Transformer And Rapid Dimensionless Instance Segmentation

<module> PyTest main

New York Structural Biology Center
Simons Machine Learning Center

Robert Kiewisz, Tristan Bepler
MIT License 2021 - 2022
"""
from click.testing import CliRunner


def test_cnn_trainer():
    from tardis.train_spindletorch import main

    runner = CliRunner()
    result = runner.invoke(main, ['--version'])
    assert str(result) == '<Result okay>'


def test_gf_trainer():
    from tardis.train_DIST import main

    runner = CliRunner()
    result = runner.invoke(main, ['--version'])
    assert str(result) == '<Result okay>'
