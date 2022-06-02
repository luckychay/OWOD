import cv2
import os
import argparse
import torch
from torch.distributions.weibull import Weibull
from torch.distributions.transforms import AffineTransform
from torch.distributions.transformed_distribution import TransformedDistribution
from torch.nn.functional import threshold
from detectron2.utils.logger import setup_logger
setup_logger()

from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor, default_argument_parser
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog


def create_distribution(scale, shape, shift):
    wd = Weibull(scale=scale, concentration=shape)
    transforms = AffineTransform(loc=shift, scale=1.)
    weibull = TransformedDistribution(wd, transforms)
    return weibull


def compute_prob(x, distribution):
    eps_radius = 0.5
    num_eval_points = 100
    start_x = x - eps_radius
    end_x = x + eps_radius
    step = (end_x - start_x) / num_eval_points
    dx = torch.linspace(x - eps_radius, x + eps_radius, num_eval_points)
    pdf = distribution.log_prob(dx).exp()
    prob = torch.sum(pdf * step)
    return prob


def update_label_based_on_energy(logits, classes, unk_dist, known_dist):
    unknown_class_index = 80
    cls = classes
    lse = torch.logsumexp(logits[:, :5], dim=1)
    for i, energy in enumerate(lse):
        p_unk = compute_prob(energy, unk_dist)
        p_known = compute_prob(energy, known_dist)
        if torch.isnan(p_unk) or torch.isnan(p_known):
            continue
        if p_unk > p_known:
            cls[i] = unknown_class_index
    return cls

def get_parser():
    parser = argparse.ArgumentParser(description="Detectron2 demo for builtin configs")
    parser.add_argument(
        "--config-file",
        default="configs/quick_schedules/mask_rcnn_R_50_FPN_inference_acc_test.yaml",
        metavar="FILE",
        help="path to config file",
    )
    parser.add_argument("--webcam", action="store_true", help="Take inputs from webcam.")
    parser.add_argument("--video-input", help="Path to video file.")
    parser.add_argument(
        "--input",
        nargs="+",
        help="file name",
    )
    parser.add_argument(
        "--output",
        help="A file or directory to save output visualizations. "
        "If not given, will show output in an OpenCV window.",
    )

    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Minimum score for instance predictions to be shown",
    )

    parser.add_argument(
        "--nms",
        type=float,
        default=0.5,
        help="Minimum score for nms",
    )

    parser.add_argument(
        "--task",
        help="task symbol",
        default=[],
    )
    return parser

if __name__=="__main__":

    args = get_parser().parse_args()

    if args.input:
        # Get image
        file_name = args.input
    else:
        file_name = [f for f in os.listdir("/home/appuser/OWOD/datasets/VOC2007/JPEGImages/")]

    task_num = args.task
    thres = args.threshold
    nms_thres = args.nms
    
    cfg_file = '/home/appuser/OWOD/output/{}/config.yaml'.format(task_num)

    # Get the configuration ready
    cfg = get_cfg()
    cfg.merge_from_file(cfg_file)
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = float(thres)
    # cfg.MODEL.ROI_HEADS.POSITIVE_FRACTION = 0.8
    cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST = float(nms_thres)

    if task_num[-1] != '4':
        model = '/home/appuser/OWOD/output/{}/model_final.pth'.format(task_num)
    else:
        model = '/home/appuser/OWOD/output/{}_ft/model_final.pth'.format(task_num)
    
    print("using model {}".format(model))
    print("using config {}".format(cfg_file))
    cfg.MODEL.WEIGHTS = model

    predictor = DefaultPredictor(cfg)

    if task_num[-1] !='4':
        param_save_location = os.path.join('/home/appuser/OWOD/output/{}_final/energy_dist_{}.pkl'.format(task_num,str(20*int(task_num[-1]))))
        params = torch.load(param_save_location)
        unknown = params[0]
        known = params[1]
        unk_dist = create_distribution(unknown['scale_unk'], unknown['shape_unk'], unknown['shift_unk'])
        known_dist = create_distribution(known['scale_known'], known['shape_known'], known['shift_known'])

    for i in range(len(file_name)):

        im = cv2.imread("/home/appuser/OWOD/datasets/VOC2007/JPEGImages/" + file_name[i])

        print("image shape:",im.shape)

        try:
            outputs = predictor(im)
        except Exception as e:
            print(e)
            continue

        if task_num[-1] != '4':
            print('Before' + str(outputs["instances"].pred_classes))

            instances = outputs["instances"].to(torch.device("cpu"))
            dev =instances.pred_classes.get_device()
            classes = instances.pred_classes.tolist()
            logits = instances.logits
            classes = update_label_based_on_energy(logits, classes, unk_dist, known_dist)
            classes = torch.IntTensor(classes).to(torch.device("cuda"))
            outputs["instances"].pred_classes = classes
            print(classes)

            print('After' + str(outputs["instances"].pred_classes))
            

        v = Visualizer(im[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2)
        v = v.draw_instance_predictions(outputs['instances'].to('cpu'))
        img = v.get_image()[:, :, ::-1]

        del outputs

        cv2.namedWindow(file_name[i], cv2.WINDOW_NORMAL)
        cv2.imshow(file_name[i], img)
        cv2.waitKey(0)
        cv2.destroyWindow(file_name[i])
        print('testing image {}'.format(i))
        cv2.imwrite('output_' + file_name[i], img)
        
        # print("please input arguments!")

