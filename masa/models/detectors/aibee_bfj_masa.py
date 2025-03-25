"""
Author: Yingqin She
Licensed: Apache-2.0 License
"""

from mmdet.registry import MODELS
from mmdet.models.detectors.retinanet import RetinaNet
from mmcv.ops.nms import batched_nms
from mmengine.structures import InstanceData


@MODELS.register_module()
class AibeeBfjMasa(RetinaNet):
    """Implementation of `Grounding DINO: Marrying DINO with Grounded Pre-
    Training for Open-Set Object Detection.

    <https://arxiv.org/abs/2303.05499>`_

    Code is modified from the `official github repo
    <https://github.com/IDEA-Research/GroundingDINO>`_.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)



    def predict(
        self, batch_inputs, detection_features, batch_data_samples, rescale: bool = True
    ):
        results_list = self.bbox_head.predict(
            detection_features, batch_data_samples, rescale=rescale)
        batch_data_samples = self.add_pred_to_datasample(
            batch_data_samples, results_list)
        result = batch_data_samples[0]
        det_bboxes, keep_idx = batched_nms(boxes=result.pred_instances.bboxes,
                                           scores=result.pred_instances.scores,
                                           idxs=result.pred_instances.labels,
                                           class_agnostic=True,
                                           nms_cfg=dict(type='nms',
                                                        iou_threshold=0.8,
                                                        class_agnostic=True,
                                                        split_thr=100000))
        det_results = InstanceData()
        det_results.bboxes = det_bboxes[:, :4]
        det_results.labels = result.pred_instances.labels[keep_idx]
        det_results.scores = result.pred_instances.scores[keep_idx]
        result.pred_instances = det_results
        return batch_data_samples