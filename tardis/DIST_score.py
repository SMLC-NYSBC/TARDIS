from os import getcwd, listdir, mkdir
from os.path import isdir, join
from shutil import rmtree

import click
import numpy as np
import torch

from tardis.dist_pytorch.transformer.network import DIST
from tardis.dist_pytorch.utils.augmentation import preprocess_data
from tardis.dist_pytorch.utils.voxal import VoxalizeDataSetV2
from tardis.slcpy.utils.segment_point_cloud import GraphInstanceV2
from tardis.spindletorch.unet.predictor import Predictor
from tardis.utils.device import get_device
from tardis.utils.logo import tardis_logo
from tardis.utils.metrics import F1_metric, IoU, distAUC, mCov
from tardis.utils.utils import pc_median_dist
from tardis.version import version

torch.backends.cudnn.enabled = False
torch.backends.cudnn.benchmark = True
torch.autograd.set_detect_anomaly(mode=False)
torch.autograd.profiler.profile(enabled=False)
torch.autograd.profiler.emit_nvtx(enabled=False)


@click.command()
@click.option('-dir', '--gf_dir',
              default=getcwd(),
              type=str,
              help='Directory with images for prediction with GF model.',
              show_default=True)
@click.option('-gst', '--gf_structure',
              default='full',
              type=click.Choice(['full', 'full_af', 'self_attn', 'triang', 'dualtriang', 'quad']),
              help='Structure of the graphformer',
              show_default=True)
@click.option('-gni', '--gf_ninput',
              default=None,
              type=int,
              help='Node input feature.',
              show_default=True)
@click.option('-gn', '--gf_ndim',
              default=256,
              type=int,
              help='Number embedding channels for nodes.',
              show_default=True)
@click.option('-ge', '--gf_edim',
              default=128,
              type=int,
              help='Number embedding channels for edges.',
              show_default=True)
@click.option('-gl', '--gf_layer',
              default=6,
              type=int,
              help='Number of GF layers',
              show_default=True)
@click.option('-gh', '--gf_heads',
              default=8,
              type=int,
              help='Number of GF heads in MHA',
              show_default=True)
@click.option('-gd', '--gf_dropout',
              default=0,
              type=float,
              help='If 0, dropout is turn-off. Else indicate dropout rate',
              show_default=True)
@click.option('-gs', '--gf_sigma',
              default=0.6,
              type=float,
              help='Sigma value for distance embedding',
              show_default=True)
@click.option('-wi', '--with_img',
              default=False,
              type=bool,
              help='Directory with train, test folder or folder with dataset ',
              show_default=True)
@click.option('-gth', '--gf_threshold',
              default=0.5,
              type=float,
              help='Threshold value for the prediction',
              show_default=True)
@click.option('-piv', '--point_in_voxal',
              default=500,
              type=int,
              help='Max number of point per voxal',
              show_default=True)
@click.option('-gch', '--checkpoint',
              default=None,
              type=str,
              help='If not None, directory to checkpoint',
              show_default=True)
@click.option('-gm', '--model',
              default='cryo_membrane',
              type=click.Choice(['cryo_membrane', 'microtubules']),
              help='Model checkpoint',
              show_default=True)
@click.option('-ex', '--export',
              default=False,
              type=bool,
              help='If True, export metric',
              show_default=True)
@click.option('-d', '--device',
              default=0,
              type=str,
              help='Define which device use for training: '
              'gpu: Use ID 0 gpus '
              'cpu: Usa CPU '
              '0-9 - specified gpu device id to use',
              show_default=True)
@click.option('-tq', '--tqdm',
              default=True,
              type=bool,
              help='If True, build with progressbar.',
              show_default=True)
@click.version_option(version=version)
def main(gf_dir: str,
         gf_structure: str,
         gf_ninput: int,
         gf_ndim: int,
         gf_edim: int,
         gf_layer: int,
         gf_heads: int,
         gf_dropout,
         gf_sigma: float,
         with_img: bool,
         point_in_voxal: int,
         gf_threshold,
         checkpoint,
         model,
         export: bool,
         device: str,
         tqdm: bool):
    """
    MAIN MODULE FOR GF for metric evaluation
    """
    """Initial setup"""
    tardis_logo(title='Metric evaluation for DIST')
    
    available_format = ('.csv', '.CorrelationLines.am', '.npy')
    GF_list = [f for f in listdir(gf_dir) if f.endswith(available_format)]
    assert len(GF_list) > 0, 'No file found in given directory!'
    macc, mprec, mrecall, mf1, miou = [], [], [], [], []
    seg_prec, seg_rec, mcov_score = [], [], []
    dice_AUC, AP50 = [], []
    n_crop = []

    # Build handlers
    GraphToSegment = GraphInstanceV2(threshold=gf_threshold,
                                     connection=2,
                                     prune=2)

    GF = Predictor(model=DIST(n_out=1,
                              node_input=gf_ninput,
                              node_dim=gf_ndim,
                              edge_dim=gf_edim,
                              num_layers=gf_layer,
                              num_heads=gf_heads,
                              dropout_rate=gf_dropout,
                              coord_embed_sigma=gf_sigma,
                              structure=gf_structure,
                              predict=True),
                   checkpoint=checkpoint,
                   network='graphformer',
                   subtype='without_img',
                   model_type=model,
                   device=get_device(device),
                   tqdm=tqdm)

    if tqdm:
        from tqdm import tqdm as tq

        batch_iter = tq(GF_list)
    else:
        batch_iter = GF_list

    """Process each image with CNN and GF"""
    for i in batch_iter:
        if tqdm:
            batch_iter.set_description(f'Predicting {i}')

        # coord <- array of all point with labels
        # coord_dist <- array of all points with labels after normalization
        # coords <- voxal of coord
        # coord_vx <- voxals for prediction
        coord, img = preprocess_data(coord=join(gf_dir, i),
                                     image=None,
                                     include_label=True,
                                     size=None)

        dist = pc_median_dist(pc=coord[:, 1:], avg_over=True)
        coord_dist = coord
        coord_dist[:, 1:] = coord[:, 1:] / dist

        VD = VoxalizeDataSetV2(coord=coord_dist,
                               init_voxal_size=0,
                               drop_rate=1,
                               downsampling_threshold=point_in_voxal,
                               downsampling_rate=None,
                               graph=True)
        coords, img, graph_target, output_idx = VD.voxalize_dataset(prune=5)
        coord_vx = [c / pc_median_dist(c, avg_over=True) for c in coords]

        dl_iter = tq(zip(coord_vx, img),
                     'Voxals',
                     leave=False)

        graphs = []
        coords = []
        for c, img in dl_iter:
            if with_img:
                graph = GF._predict(x=c[None, :],
                                    y=img[None, :])
            else:
                graph = GF._predict(x=c[None, :],
                                    y=None)
            graphs.append(graph)

        n_crop.append(len(graphs))
        print(f'No. of crops {len(graphs)}')

        graph_target = GraphToSegment._stitch_graph(graph_target, output_idx)
        graph_target = np.where(graph_target > 0, 1, 0)
        graph_logits = GraphToSegment._stitch_graph(graphs, output_idx)

        segments = GraphToSegment.voxal_to_segment(graph=graphs,
                                                   coord=coord[:, 1:],
                                                   idx=output_idx,
                                                   sort=True,
                                                   visualize=None)

        """Prediction evaluation"""
        acc, prec, rec, f1 = F1_metric(graph_target.flatten(),
                                       np.where(graph_logits >= gf_threshold, 1, 0).flatten())
        macc.append(acc)
        mprec.append(prec)
        mrecall.append(rec)
        mf1.append(f1)

        print('GraphFormer scored with:')
        print(f'Accuracy of {acc}')
        print(f'Precision of {prec}')
        print(f'Recall of {rec}')
        print(f'F1 of {f1}')

        # map = AP(graph_target,
        #          graph_logits)
        # print(f'AP50 of {map}')
        # AP50.append(map)

        iou = IoU(graph_target,
                  graph_logits)

        miou.append(iou)
        print(f'mAP of {iou}')

        """Dist AUC"""
        dist_auc = distAUC(coord=coord[:, 1:],
                           target=graph_target)
        dice_AUC.append(dist_auc)
        print(f'Dist AUC: {dist_auc}')

        """Segmentation evaluation"""
        if segments.shape[1] > coord.shape[1]:
            segments = segments[:, :3]  # Hotfix for 2D segments

        mcov, prec, rec, map = mCov(coord,
                                    segments)
        mcov_score.append(mcov)
        seg_prec.append(prec)
        seg_rec.append(rec)
        print(f'AP50 of {map}')
        AP50.append(map)

        print(f'mCov of {round(np.mean(mcov), 3)}')
        print(f'mPrec of {round(np.mean(prec), 3)}')
        print(f'mRec of {round(np.mean(rec), 3)}')

    if export:
        if isdir(join(gf_dir, 'scores')):
            rmtree(join(gf_dir, 'scores'))

        mkdir(join(gf_dir, 'scores'))

        np.savetxt(join(gf_dir, 'scores', 'mAccuracy.csv'),
                   macc,
                   delimiter=',')
        np.savetxt(join(gf_dir, 'scores', 'mPrecision.csv'),
                   mprec,
                   delimiter=',')
        np.savetxt(join(gf_dir, 'scores', 'mRecall.csv'),
                   mrecall,
                   delimiter=',')
        np.savetxt(join(gf_dir, 'scores', 'mF1.csv'),
                   mf1,
                   delimiter=',')
        np.savetxt(join(gf_dir, 'scores', 'mioU.csv'),
                   miou,
                   delimiter=',')
        np.savetxt(join(gf_dir, 'scores', 'mCov.csv'),
                   mcov_score,
                   delimiter=',')

    print('GraphFormer scored with:')
    print(f'Mean accuracy of {round(np.mean(macc), 3)}')
    print(f'Mean precision of {round(np.mean(mprec), 3)}')
    print(f'Mean recall of {round(np.mean(mrecall), 3)}')
    print(f'Mean F1 of {round(np.mean(mf1), 3)}')
    print(f'Mean AP50 of {round(np.mean(AP50), 3)}')
    print(f'Mean mAP of {round(np.mean(miou), 3)}')
    print(f'Mean dist mAUC of {round(np.mean(dice_AUC), 3)}')
    print('')
    print(f'Mean n_crop {np.mean(n_crop)}')
    print(f'Mean mCov of {round(np.mean(mcov_score), 3)}')
    print(f'Mean Seg_mPrec of {round(np.mean(seg_prec), 3)}')
    print(f'Mean Seg_mRec of {round(np.mean(seg_rec), 3)}')


if __name__ == '__main__':
    main()
