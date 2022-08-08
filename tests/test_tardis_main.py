from click.testing import CliRunner
from tardis.spindletorch_postprocess import main as cnn_postprocess
from tardis.predict_spindletorch import main as cnn_predictor
from tardis.predict_MTs import main as t_mt
from tardis.train_spindletorch import main as cnn_trainer
from tardis.train_DIST import main as gf_trainer


class TestClick:
    runner = CliRunner()

    def test_cnn_trainer(self):
        result = self.runner.invoke(cnn_trainer, ['--version'])
        assert str(result) == '<Result okay>'

    def test_cnn_predict(self):
        result = self.runner.invoke(cnn_predictor, ['--version'])
        assert str(result) == '<Result okay>'

    def test_cnn_postprocess(self):
        result = self.runner.invoke(cnn_postprocess, ['--version'])
        assert str(result) == '<Result okay>'

    def test_gf_trainer(self):
        result = self.runner.invoke(gf_trainer, ['--version'])
        assert str(result) == '<Result okay>'

    def test_t_mt(self):
        result = self.runner.invoke(t_mt, ['--version'])
        assert str(result) == '<Result okay>'
