from click.testing import CliRunner
from tardis.train_image_segmentation import main as main_trainer


class TestClick:
    runner = CliRunner()

    def test_trainer(self):
        result = self.runner.invoke(main_trainer, ['--version'])
        assert str(result) == '<Result okay>'
