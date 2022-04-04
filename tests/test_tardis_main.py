from click.testing import CliRunner
from tardis.train_image_segmentation import main as cnn_trainer
from tardis.predict_image_segmentation import main as cnn_predictor


class TestClick:
    runner = CliRunner()

    def test_cnn_trainer(self):
        result = self.runner.invoke(cnn_trainer, ['--version'])
        assert str(result) == '<Result okay>'

    def test_cnn_predict(self):
        result = self.runner.invoke(cnn_predictor, ['--version'])
        assert str(result) == '<Result okay>'
