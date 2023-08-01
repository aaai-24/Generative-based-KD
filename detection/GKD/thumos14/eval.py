import argparse
from GKD.evaluation.eval_detection import ANETdetection

parser = argparse.ArgumentParser()
parser.add_argument('output_json', type=str)
parser.add_argument('gt_json', type=str,
                    default='./thumos_annotations/thumos_gt.json', nargs='?')
args = parser.parse_args()

tious = [0.1, 0.2, 0.3, 0.4, 0.5]
anet_detection = ANETdetection(
    ground_truth_filename=args.gt_json,
    prediction_filename=args.output_json,
    subset='test', tiou_thresholds=tious)
mAPs, average_mAP, ap = anet_detection.evaluate()
avg = 0.0
for (tiou, mAP) in zip(tious, mAPs):
    print("mAP at tIoU {} is {:.5f}".format(tiou, mAP))
    avg += mAP
avg /= len(tious)
print('AVG = {:.5f}'.format(avg))
