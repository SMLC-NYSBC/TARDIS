import time
from os import getcwd, listdir, mkdir
from os.path import isdir, join
from shutil import rmtree

import click
import numpy as np
import torch

from tardis.dist_pytorch.transformer.network import DIST
from tardis.dist_pytorch.utils.augmentation import preprocess_data
from tardis.dist_pytorch.utils.voxal import VoxalizeDataSetV2
from tardis.slcpy.utils.load_data import load_ply
from tardis.slcpy.utils.segment_point_cloud import GraphInstanceV2
from tardis.utils.device import get_device
from tardis.utils.logo import Tardis_Logo, printProgressBar
from tardis.utils.metrics import AP50_ScanNet, F1_metric, IoU, distAUC, mCov
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
@click.option('-sn', '--scannet',
              default=False,
              type=bool,
              help='If True, validate for ScanNet data',
              show_default=True)
@click.option('-d', '--device',
              default=0,
              type=str,
              help='Define which device use for training: '
              'gpu: Use ID 0 gpus '
              'cpu: Usa CPU '
              '0-9 - specified gpu device id to use',
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
         scannet=False):
    """
    MAIN MODULE FOR GF for metric evaluation
    """
    """Initial setup"""
    device = get_device(device)

    tardis_logo = Tardis_Logo()
    tardis_logo(title='Metric evaluation for DIST')

    available_format = ('.csv', '.CorrelationLines.am', '.npy', '.ply')
    GF_list = [f for f in listdir(gf_dir) if f.endswith(available_format)]
    assert len(GF_list) > 0, 'No file found in given directory!'

    if scannet:
        eval_list = ['scene0568_00', 'scene0568_01', 'scene0568_02', 'scene0304_00',
                     'scene0488_00', 'scene0488_01', 'scene0412_00', 'scene0412_01',
                     'scene0217_00', 'scene0019_00', 'scene0019_01', 'scene0414_00',
                     'scene0575_00', 'scene0575_01', 'scene0575_02', 'scene0426_00',
                     'scene0426_01', 'scene0426_02', 'scene0426_03', 'scene0549_00',
                     'scene0549_01', 'scene0578_00', 'scene0578_01', 'scene0578_02',
                     'scene0665_00', 'scene0665_01', 'scene0050_00', 'scene0050_01',
                     'scene0050_02', 'scene0257_00', 'scene0025_00', 'scene0025_01',
                     'scene0025_02', 'scene0583_00', 'scene0583_01', 'scene0583_02',
                     'scene0701_00', 'scene0701_01', 'scene0701_02', 'scene0580_00',
                     'scene0580_01', 'scene0565_00', 'scene0169_00', 'scene0169_01',
                     'scene0655_00', 'scene0655_01', 'scene0655_02', 'scene0063_00',
                     'scene0221_00', 'scene0221_01', 'scene0591_00', 'scene0591_01',
                     'scene0591_02', 'scene0678_00', 'scene0678_01', 'scene0678_02',
                     'scene0462_00', 'scene0427_00', 'scene0595_00', 'scene0193_00',
                     'scene0193_01', 'scene0164_00', 'scene0164_01', 'scene0164_02',
                     'scene0164_03', 'scene0598_00', 'scene0598_01', 'scene0598_02',
                     'scene0599_00', 'scene0599_01', 'scene0599_02', 'scene0328_00',
                     'scene0300_00', 'scene0300_01', 'scene0354_00', 'scene0458_00',
                     'scene0458_01', 'scene0423_00', 'scene0423_01', 'scene0423_02',
                     'scene0307_00', 'scene0307_01', 'scene0307_02', 'scene0606_00',
                     'scene0606_01', 'scene0606_02', 'scene0432_00', 'scene0432_01',
                     'scene0608_00', 'scene0608_01', 'scene0608_02', 'scene0651_00',
                     'scene0651_01', 'scene0651_02', 'scene0430_00', 'scene0430_01',
                     'scene0689_00', 'scene0357_00', 'scene0357_01', 'scene0574_00',
                     'scene0574_01', 'scene0574_02', 'scene0329_00', 'scene0329_01',
                     'scene0329_02', 'scene0153_00', 'scene0153_01', 'scene0616_00',
                     'scene0616_01', 'scene0671_00', 'scene0671_01', 'scene0618_00',
                     'scene0382_00', 'scene0382_01', 'scene0490_00', 'scene0621_00',
                     'scene0607_00', 'scene0607_01', 'scene0149_00', 'scene0695_00',
                     'scene0695_01', 'scene0695_02', 'scene0695_03', 'scene0389_00',
                     'scene0377_00', 'scene0377_01', 'scene0377_02', 'scene0342_00',
                     'scene0139_00', 'scene0629_00', 'scene0629_01', 'scene0629_02',
                     'scene0496_00', 'scene0633_00', 'scene0633_01', 'scene0518_00',
                     'scene0652_00', 'scene0406_00', 'scene0406_01', 'scene0406_02',
                     'scene0144_00', 'scene0144_01', 'scene0494_00', 'scene0278_00',
                     'scene0278_01', 'scene0316_00', 'scene0609_00', 'scene0609_01',
                     'scene0609_02', 'scene0609_03', 'scene0084_00', 'scene0084_01',
                     'scene0084_02', 'scene0696_00', 'scene0696_01', 'scene0696_02',
                     'scene0351_00', 'scene0351_01', 'scene0643_00', 'scene0644_00',
                     'scene0645_00', 'scene0645_01', 'scene0645_02', 'scene0081_00',
                     'scene0081_01', 'scene0081_02', 'scene0647_00', 'scene0647_01',
                     'scene0535_00', 'scene0353_00', 'scene0353_01', 'scene0353_02',
                     'scene0559_00', 'scene0559_01', 'scene0559_02', 'scene0593_00',
                     'scene0593_01', 'scene0246_00', 'scene0653_00', 'scene0653_01',
                     'scene0064_00', 'scene0064_01', 'scene0356_00', 'scene0356_01',
                     'scene0356_02', 'scene0030_00', 'scene0030_01', 'scene0030_02',
                     'scene0222_00', 'scene0222_01', 'scene0338_00', 'scene0338_01',
                     'scene0338_02', 'scene0378_00', 'scene0378_01', 'scene0378_02',
                     'scene0660_00', 'scene0553_00', 'scene0553_01', 'scene0553_02',
                     'scene0527_00', 'scene0663_00', 'scene0663_01', 'scene0663_02',
                     'scene0664_00', 'scene0664_01', 'scene0664_02', 'scene0334_00',
                     'scene0334_01', 'scene0334_02', 'scene0046_00', 'scene0046_01',
                     'scene0046_02', 'scene0203_00', 'scene0203_01', 'scene0203_02',
                     'scene0088_00', 'scene0088_01', 'scene0088_02', 'scene0088_03',
                     'scene0086_00', 'scene0086_01', 'scene0086_02', 'scene0670_00',
                     'scene0670_01', 'scene0256_00', 'scene0256_01', 'scene0256_02',
                     'scene0249_00', 'scene0441_00', 'scene0658_00', 'scene0704_00',
                     'scene0704_01', 'scene0187_00', 'scene0187_01', 'scene0131_00',
                     'scene0131_01', 'scene0131_02', 'scene0207_00', 'scene0207_01',
                     'scene0207_02', 'scene0461_00', 'scene0011_00', 'scene0011_01',
                     'scene0343_00', 'scene0251_00', 'scene0077_00', 'scene0077_01',
                     'scene0684_00', 'scene0684_01', 'scene0550_00', 'scene0686_00',
                     'scene0686_01', 'scene0686_02', 'scene0208_00', 'scene0500_00',
                     'scene0500_01', 'scene0552_00', 'scene0552_01', 'scene0648_00',
                     'scene0648_01', 'scene0435_00', 'scene0435_01', 'scene0435_02',
                     'scene0435_03', 'scene0690_00', 'scene0690_01', 'scene0693_00',
                     'scene0693_01', 'scene0693_02', 'scene0700_00', 'scene0700_01',
                     'scene0700_02', 'scene0699_00', 'scene0231_00', 'scene0231_01',
                     'scene0231_02', 'scene0697_00', 'scene0697_01', 'scene0697_02',
                     'scene0697_03', 'scene0474_00', 'scene0474_01', 'scene0474_02',
                     'scene0474_03', 'scene0474_04', 'scene0474_05', 'scene0355_00',
                     'scene0355_01', 'scene0146_00', 'scene0146_01', 'scene0146_02',
                     'scene0196_00', 'scene0702_00', 'scene0702_01', 'scene0702_02',
                     'scene0314_00', 'scene0277_00', 'scene0277_01', 'scene0277_02',
                     'scene0095_00', 'scene0095_01', 'scene0015_00', 'scene0100_00',
                     'scene0100_01', 'scene0100_02', 'scene0558_00', 'scene0558_01',
                     'scene0558_02', 'scene0685_00', 'scene0685_01', 'scene0685_02']
        GF_list = [f for f in GF_list if f[:-22] in eval_list]
        assert len(GF_list) > 0, 'No file found in given directory!'

        all_prec, all_rec, all_ap50, all_label = [], [], [], []
    else:
        macc, mprec, mrecall, mf1, miou = [], [], [], [], []
        seg_prec, seg_rec, mcov_score = [], [], []
        dice_AUC, AP50 = [], []
    n_crop = []

    # Build handlers
    GraphToSegment = GraphInstanceV2(threshold=gf_threshold,
                                     connection=2,
                                     prune=2)

    model = DIST(n_out=1,
                 node_input=gf_ninput,
                 node_dim=gf_ndim,
                 edge_dim=gf_edim,
                 num_layers=gf_layer,
                 num_heads=gf_heads,
                 dropout_rate=gf_dropout,
                 coord_embed_sigma=gf_sigma,
                 structure=gf_structure,
                 predict=True)
    save_train = torch.load(checkpoint, map_location='cpu')
    model.load_state_dict(save_train['model_state_dict'])

    """Process each image with CNN and GF"""
    start = 0
    end = 0
    for id, i in enumerate(GF_list):
        elapse = round((end - start) * (len(GF_list) - id), 0)
        tardis_logo(title='Metric evaluation for DIST',
                    text_2=f'Predicting {i} Est. elapse time: {elapse}',
                    text_3=printProgressBar(id, len(GF_list)),
                    text_4='Current task: Preprocessing image...')
        start = time.time()
        # coord <- array of all point with labels
        # coord_dist <- array of all points with labels after normalization
        # coords <- voxal of coord
        # coord_vx <- voxals for prediction
        if scannet:
            coord = load_ply(join(gf_dir, i), downsample=0.035)
            img = [[0] for _ in range(len(coord))]
        else:
            coord, img = preprocess_data(coord=join(gf_dir, i),
                                         image=None,
                                         include_label=True,
                                         size=None)

        dist = pc_median_dist(pc=coord[:, 1:], avg_over=True, box_size=0.15)
        coord_dist = coord
        coord_dist[:, 1:] = coord[:, 1:] / dist

        VD = VoxalizeDataSetV2(coord=coord_dist,
                               init_voxal_size=0,
                               drop_rate=0.1,
                               voxal_3d=False,
                               downsampling_threshold=point_in_voxal,
                               downsampling_rate=None,
                               graph=True)
        if scannet:
            coords, _, graph_target, output_idx = VD.voxalize_dataset(mesh=True,
                                                                      out_idx=True,
                                                                      prune=5)
        else:
            coords, _, graph_target, output_idx = VD.voxalize_dataset(prune=5)
        coord_vx = [c / pc_median_dist(c) for c in coords]

        # Generate GT .txt
        # save GT .txt
        GraphToSegment = GraphInstanceV2(threshold=0.91,
                                         connection=5,
                                         prune=100)

        graphs = []
        coords = []
        model.to(device)
        model.eval()
        for idx, (c, img) in enumerate(zip(coord_vx, img)):
            tardis_logo(title='Metric evaluation for DIST',
                        text_2=f'Predicting {i} Est. elapse time: {elapse}',
                        text_3=printProgressBar(id, len(GF_list)),
                        text_4='Current task: Voxals prediction...',
                        text_5=printProgressBar(idx, len(coord_vx)))
            if with_img:
                x, y = c.to(device), img.to(device)
                graphs.append(model(x[None, :], y[None, :])[0, 0, :].cpu().detach().numpy())
            else:
                with torch.no_grad():
                    x = c.to(device)
                    graphs.append(model(x[None, :])[0, 0, :].cpu().detach().numpy())

        n_crop.append(len(graphs))

        tardis_logo(title='Metric evaluation for DIST',
                    text_2=f'Predicting {i} Est. elapse time: {elapse}',
                    text_3=printProgressBar(id, len(GF_list)),
                    text_4='Current task: Graph prediction...')

        # graph_target = GraphToSegment._stitch_graph(graph_target, output_idx)
        # graph_target = np.where(graph_target > 0, 1, 0)
        # graph_logits = GraphToSegment._stitch_graph(graphs, output_idx)
        try:
            if scannet:
                segments = GraphToSegment.voxal_to_segment(graph=graphs,
                                                           coord=coord_dist[:, 1:],
                                                           idx=output_idx,
                                                           sort=False,
                                                           visualize=None)
            else:
                segments = GraphToSegment.voxal_to_segment(graph=graphs,
                                                           coord=coord_dist[:, 1:],
                                                           idx=output_idx,
                                                           sort=True,
                                                           visualize=None)

            """Prediction evaluation"""
            tardis_logo(title='Metric evaluation for DIST',
                        text_2=f'Predicting {i} Est. elapse time: {elapse}',
                        text_3=printProgressBar(id, len(GF_list)),
                        text_4='Current task: Calculating metrics...')
            if scannet:
                ap50, prec, rec, label = AP50_ScanNet(segments, coord_dist)

                all_prec.append(prec)
                all_rec.append(rec)
                all_ap50.append(ap50)
                all_label.append(label)
            else:
                graph_logits = GraphToSegment._stitch_graph(graphs, output_idx)
                acc, prec, rec, f1 = F1_metric(graph_target.flatten(),
                                               np.where(graph_logits >= gf_threshold, 1, 0).flatten())
                macc.append(acc)
                mprec.append(prec)
                mrecall.append(rec)
                mf1.append(f1)

                iou = IoU(graph_target,
                          graph_logits)

                miou.append(iou)

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

                AP50.append(map)
        except:
            continue
        end = time.time()

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
    if scannet:
        tardis_logo(title='Metric evaluation for DIST',
                    text_2='DIST model scored with:')
        classes_eval = []
        for i in np.unique(np.concatenate(all_label)):
            id = list(np.where(np.array(np.concatenate(all_label)) == i)[0])
            prec = np.mean([c for idx, c in enumerate(np.concatenate(all_prec)) if idx in id])
            rec = np.mean([c for idx, c in enumerate(np.concatenate(all_rec)) if idx in id])
            ap = np.mean([c for idx, c in enumerate(np.concatenate(all_ap50)) if idx in id])
            classes_eval.append(f'{i}: AP50: {ap}, mPrec: {prec}, mRec: {rec}\n')
        print(classes_eval)
    else:
        tardis_logo(title='Metric evaluation for DIST',
                    text_2='DIST model scored with:',
                    text_3=f'Mean precision of {round(np.mean(mprec), 3)}',
                    text_4=f'Mean recall of {round(np.mean(mrecall), 3)}',
                    text_5=f'Mean f1 of {round(np.mean(mf1), 3)}',
                    text_6=f'Mean AP50 of {round(np.mean(AP50), 3)}',
                    text_7=f'Mean n_crop {np.mean(n_crop)}',
                    text_8=f'Mean mCov of {round(np.mean(mcov_score), 3)}',
                    text_9=f'Mean Seg_mPrec of {round(np.mean(seg_prec), 3)}',
                    text_10=f'Mean Seg_mPrec of {round(np.mean(seg_prec), 3)}')


if __name__ == '__main__':
    main()
