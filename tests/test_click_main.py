from click.testing import CliRunner


# class TestClick:
#     runner = CliRunner()

def test_cnn_trainer():
    from tardis_dev.train_spindletorch import main

    runner = CliRunner()
    result = runner.invoke(main, ['--version'])
    assert str(result) == '<Result okay>'


def test_gf_trainer():
    from tardis_dev.train_DIST import main

    runner = CliRunner()
    result = runner.invoke(main, ['--version'])
    assert str(result) == '<Result okay>'
