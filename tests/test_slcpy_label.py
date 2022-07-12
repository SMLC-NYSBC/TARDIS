import numpy as np
from tardis.slcpy.utils.build_semantic_mask import draw_semantic


class TestLabelDrawing:
    pixel_size = 2.32
    r = round(12.5 * pixel_size)

    def test_drawing_2D_single(self):
        df_coord = np.zeros((100, 4))
        start = 0
        stop = 10

        for i in range(10):
            start += 10
            stop += 10
            df_coord[start:stop, 0] = i
            df_coord[start:stop, 1] = i + 4
            df_coord[start:stop, 2] = i + 2

        labels_2D = draw_semantic(mask_size=(64, 64),
                                  coordinate=df_coord,
                                  pixel_size=self.pixel_size,
                                  circle_size=self.r,
                                  multi_layer=False,
                                  tqdm=True)
        assert np.any(labels_2D == 1), 'Label was not created!'
        assert not np.any(
            labels_2D > 1), 'Multi label was created in single mode!'

    def test_drawing_2D_multi(self):
        df_coord = np.zeros((100, 4))
        start = 0
        stop = 10

        for i in range(10):
            start += 10
            stop += 10
            df_coord[start:stop, 0] = i
            df_coord[start:stop, 1] = i + 4
            df_coord[start:stop, 2] = i + 2
            df_coord[start:stop, 3] = start / 2

        labels_2D = draw_semantic(mask_size=(64, 64),
                                  coordinate=df_coord,
                                  pixel_size=self.pixel_size,
                                  circle_size=self.r,
                                  multi_layer=True,
                                  tqdm=True)
        assert np.any(labels_2D > 0), 'Multi label was not created!'

    def test_drawing_3D_single(self):
        df_coord = np.zeros((100, 4))
        start = 0
        stop = 10

        for i in range(10):
            start += 10
            stop += 10
            df_coord[start:stop, 0] = i
            df_coord[start:stop, 1] = i + 4
            df_coord[start:stop, 2] = i + 2

        labels_3D = draw_semantic(mask_size=(64, 64, 64),
                                  coordinate=df_coord,
                                  pixel_size=self.pixel_size,
                                  circle_size=self.r,
                                  multi_layer=False,
                                  tqdm=True)
        assert np.any(labels_3D == 1), 'Label was not created!'
        assert not np.any(
            labels_3D > 1), 'Multi label was created in single mode!'

    def test_drawing_3D_multi(self):
        df_coord = np.zeros((100, 4))
        start = 0
        stop = 10

        for i in range(10):
            start += 10
            stop += 10
            df_coord[start:stop, 0] = i
            df_coord[start:stop, 1] = i + 4
            df_coord[start:stop, 2] = i + 2
            df_coord[start:stop, 3] = start / 2

        labels_3D = draw_semantic(mask_size=(64, 64, 64),
                                  coordinate=df_coord,
                                  pixel_size=self.pixel_size,
                                  circle_size=self.r,
                                  multi_layer=True,
                                  tqdm=True)
        assert np.any(labels_3D > 0), 'Multi label was not created!'
