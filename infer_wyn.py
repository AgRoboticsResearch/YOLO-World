import numpy as np
import torch
from mmengine.config import Config
from mmengine.dataset import Compose
from mmengine.runner import Runner
from mmengine.runner.amp import autocast
from mmyolo.registry import RUNNERS
from torchvision.ops import nms


if __name__ == "__main__":
    # load config
    cfg = Config.fromfile("F:\doctor\YOLO-World\configs\pretrain\yolo_world_l_t2i_bn_2e-4_100e_4x8gpus_obj365v1_goldg_train_lvis_minival.py")  #原来的
    #cfg = Config.fromfile("F:\doctor\YOLO-World\configs\segmentation\yolo_world_seg_m_dual_vlpan_2e-4_80e_8gpus_seghead_finetune_lvis.py")
    cfg.work_dir = "."
    cfg.load_from = "F:\doctor\YOLO-World\pretrained_weights/yolow-v8_l_clipv2_frozen_t2iv2_bn_o365_goldg_pretrain.pth"  #原来的
    #cfg.load_from = "F:\doctor\YOLO-World\checkpoint\yolo_world_seg_m_dual_vlpan_2e-4_80e_8gpus_seghead_finetune_lvis-7bca59a7.pth"
    runner = Runner.from_cfg(cfg)
    runner.call_hook("before_run")
    runner.load_or_resume()
    pipeline = cfg.test_dataloader.dataset.pipeline
    runner.pipeline = Compose(pipeline)

    # run model evaluation
    runner.model.eval()


def colorstr(*input):
    """
        Helper function for style logging
    """
    *args, string = input if len(input) > 1 else ("bold", input[0])
    colors = {"bold": "\033[1m"}

    return "".join(colors[x] for x in args) + f"{string}"


import PIL.Image
import cv2
import logging
import supervision as sv
import os

bounding_box_annotator = sv.BoxAnnotator()
label_annotator = sv.LabelAnnotator(text_position=sv.Position.CENTER)


#class_names = ("strawberry, stem, flowers, leaf, others")
class_names = ("red strawberry, green strawberry, flowers, leaf")
#class_names = ("person,dog,cat")

def run_image(
        runner,
        input_image,
        outname,
        max_num_boxes=100,
        score_thr=0.05,
        nms_thr=0.5,
):
    #output_image = "runs/detect/"+output_image
    out_dir = "F:\doctor\YOLO-World\output_wyn\image_demo/human_head/"
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    output_image = out_dir+outname 
    texts = [[t.strip()] for t in class_names.split(",")] + [[" "]]
    #print('line 62 texts', texts)
    data_info = runner.pipeline(dict(img_id=0, img_path=input_image,
                                     texts=texts))

    data_batch = dict(
        inputs=data_info["inputs"].unsqueeze(0),
        data_samples=[data_info["data_samples"]],
    )

    with autocast(enabled=False), torch.no_grad():
        output = runner.model.test_step(data_batch)[0]
        runner.model.class_names = texts
        pred_instances = output.pred_instances

    # nms
    keep_idxs = nms(pred_instances.bboxes, pred_instances.scores, iou_threshold=nms_thr)
    pred_instances = pred_instances[keep_idxs]
    pred_instances = pred_instances[pred_instances.scores.float() > score_thr]

    if len(pred_instances.scores) > max_num_boxes:
        indices = pred_instances.scores.float().topk(max_num_boxes)[1]
        pred_instances = pred_instances[indices]
    output.pred_instances = pred_instances

    # predictions
    pred_instances = pred_instances.cpu().numpy()


    detections = sv.Detections(
        xyxy=pred_instances['bboxes'],
        class_id=pred_instances['labels'],
        confidence=pred_instances['scores']
    )

    # label ids with confidence scores
    '''labels = [
        f"{class_id} {confidence:0.2f}"
        for class_id, confidence
        in zip(detections.class_id, detections.confidence)
    ]'''
    #print(texts[detections.class_id])
    labels = [
        f"{texts[class_id][0]} {confidence:0.2f}"
        for class_id, confidence
        in zip(detections.class_id, detections.confidence)
    ]

    # draw bounding box with label
    image = PIL.Image.open(input_image)
    svimage = np.array(image)
    svimage = bounding_box_annotator.annotate(svimage, detections)
    svimage = label_annotator.annotate(svimage, detections, labels)

    # save output image
    cv2.imwrite(output_image, svimage[:, :, ::-1])
    print(f"Results saved to {colorstr('bold', output_image)}")

    return svimage[:, :, ::-1]

img_dir = 'F:\doctor\YOLO-World\data\images'
#img_dir = 'F:\doctor\YOLO-World\data_wyn\im'
for file in os.listdir(img_dir):
    file_path = os.path.join(img_dir, file)
    img = run_image(runner, file_path, file)
    
