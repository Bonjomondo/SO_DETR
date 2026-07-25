"""Tests for SO-DETR V1.1 scale-adaptive Expanded-IoU."""

import unittest

import torch

from ultralytics.utils.metrics import bbox_inner_iou, scale_adaptive_expanded_iou_ratio


class ScaleAdaptiveExpandedIoUTest(unittest.TestCase):
    def test_ratio_decreases_with_target_area(self):
        boxes = torch.tensor([
            [0.5, 0.5, 0.01, 0.01],
            [0.5, 0.5, 0.10, 0.10],
            [0.5, 0.5, 0.50, 0.50],
        ])
        ratios = scale_adaptive_expanded_iou_ratio(boxes).squeeze(-1)

        self.assertGreater(ratios[0].item(), ratios[1].item())
        self.assertGreater(ratios[1].item(), ratios[2].item())
        self.assertTrue(torch.all(ratios >= 1.0))
        self.assertTrue(torch.all(ratios <= 1.5))

    def test_xyxy_and_xywh_areas_are_equivalent(self):
        xywh = torch.tensor([[0.5, 0.5, 0.2, 0.1]])
        xyxy = torch.tensor([[0.4, 0.45, 0.6, 0.55]])
        ratio_xywh = scale_adaptive_expanded_iou_ratio(xywh, xywh=True)
        ratio_xyxy = scale_adaptive_expanded_iou_ratio(xyxy, xywh=False)

        torch.testing.assert_close(ratio_xywh, ratio_xyxy)

    def test_bbox_inner_iou_accepts_per_target_ratios(self):
        boxes = torch.tensor([
            [0.50, 0.50, 0.05, 0.05],
            [0.50, 0.50, 0.40, 0.40],
        ])
        ratios = scale_adaptive_expanded_iou_ratio(boxes)
        iou = bbox_inner_iou(boxes, boxes, xywh=True, ratio=ratios)

        self.assertEqual(iou.shape, (2, 1))
        torch.testing.assert_close(iou, torch.ones_like(iou), atol=1e-5, rtol=1e-5)

    def test_invalid_hyperparameters_fail_early(self):
        boxes = torch.zeros(1, 4)
        with self.assertRaises(ValueError):
            scale_adaptive_expanded_iou_ratio(boxes, tau=0)
        with self.assertRaises(ValueError):
            scale_adaptive_expanded_iou_ratio(boxes, alpha=-0.1)
        with self.assertRaises(ValueError):
            scale_adaptive_expanded_iou_ratio(boxes, min_ratio=1.5, max_ratio=1.0)


if __name__ == "__main__":
    unittest.main()
