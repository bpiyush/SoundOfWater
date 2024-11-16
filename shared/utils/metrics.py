"""Helpers for metric functions"""
import numpy as np
import pandas as pd


def calculate_iou(box1, box2):
    """
    Calculate Intersection over Union (IoU) between two bounding boxes.

    Args:
        box1 (tuple): Coordinates of the first bounding box in the format (x1, y1, x2, y2).
        box2 (tuple): Coordinates of the second bounding box in the format (x1, y1, x2, y2).

    Returns:
        float: Intersection over Union (IoU) score.
    """
    # Extract coordinates
    x1, y1, x2, y2 = box1
    x1_, y1_, x2_, y2_ = box2

    # Calculate the intersection area
    intersection_area = max(0, min(x2, x2_) - max(x1, x1_)) * max(0, min(y2, y2_) - max(y1, y1_))

    # Calculate the areas of each bounding box
    box1_area = (x2 - x1) * (y2 - y1)
    box2_area = (x2_ - x1_) * (y2_ - y1_)

    # Calculate IoU
    iou = intersection_area / float(box1_area + box2_area - intersection_area)

    return iou


def compute_intersection_1d(x, y):
    # sort the boxes
    x1, x2 = sorted(x)
    y1, y2 = sorted(y)
    
    # compute the intersection
    intersection = max(0, min(x2, y2) - max(x1, y1))
    
    return intersection

def compute_union_1d(x, y):
    # sort the boxes
    x1, x2 = sorted(x)
    y1, y2 = sorted(y)
    
    # compute the union
    union = max(x2, y2) - min(x1, y1)
    
    return union


def compute_iou_1d(pred_box, true_box):
    """
    Compute IoU for 1D boxes.
    
    Args:
        pred_box (float): Predicted box, [x1, x2]
        true_box (float): Ground truth box, [x1, x2]
    
    Returns:
        float: IoU
    """
    intersection = compute_intersection_1d(pred_box, true_box)
    union = compute_union_1d(pred_box, true_box)
    iou = intersection / union
    return iou


def compute_iou_1d_single_candidate_multiple_targets(pred_box, true_boxes):
    """
    Compute IoU for 1D boxes.
    
    Args:
        pred_box (float): Predicted box, [x1, x2]
        true_boxes (np.ndarray): Ground truth boxes, shape: (N, 2)
    
    Returns:
        float: IoU
    """
    ious = []
    for i, true_box in enumerate(true_boxes):
        ious.append(compute_iou_1d(pred_box, true_box))
    return np.array(ious)


def compute_iou_1d_multiple_candidates_multiple_targets(pred_boxes, true_boxes):
    """
    Compute IoU for 1D boxes.
    
    Args:
        pred_boxes (np.ndarray): Predicted boxes, shape: (N, 2)
        true_boxes (np.ndarray): Ground truth boxes, shape: (N, 2)
    
    Returns:
        float: IoU
    """
    iou_matrix = np.zeros((len(pred_boxes), len(true_boxes)))
    for i, pred_box in enumerate(pred_boxes):
        for j, true_box in enumerate(true_boxes):
            iou_matrix[i, j] = compute_iou_1d(pred_box, true_box)
    return iou_matrix


def compute_mean_iou_1d(pred_boxes, gt_boxes, threshold=0.5):
    """
    Computes mean IOU for 1D bounding boxes.
    
    Args:
        pred_boxes (np.ndarray): Predicted boxes, shape: (N, 2)
        gt_boxes (np.ndarray): Ground truth boxes, shape: (N, 2)
        threshold (float): Threshold to consider a prediction correct
    
    Returns:
        float: Mean IOU
    """
    # Compute IoU for each pair of boxes
    iou_matrix = np.zeros((len(pred_boxes), len(gt_boxes)))
    for i, pred_box in enumerate(pred_boxes):
        for j, gt_box in enumerate(gt_boxes):
            iou_matrix[i, j] = compute_iou_1d(pred_box, gt_box)

    # Compute the max IoU for each predicted box
    max_iou_indices = np.argmax(iou_matrix, axis=1)
    max_iou = iou_matrix[np.arange(len(pred_boxes)), max_iou_indices]
    
    # For each predicted box, compute TP and FP ground truth boxes
    tp = np.zeros(len(pred_boxes))
    fp = np.zeros(len(pred_boxes))
    iou = np.zeros(len(pred_boxes))
    
    tp = np.where(iou_matrix >= threshold, 1, 0)
    tp = max_iou >= threshold
    fp = max_iou < threshold
    iou = max_iou
    mean_iou = np.mean(iou)
    import ipdb; ipdb.set_trace()







def calculate_mAP_1d(pred_boxes, pred_scores, true_boxes, iou_thresh=0.5):
    """Calculate mean average precision for 1D boxes.

    Args:
        pred_boxes (numpy array): Predicted boxes, shape (num_boxes,)
        pred_scores (numpy array): Predicted scores, shape (num_boxes,)
        true_boxes (numpy array): Ground truth boxes, shape (num_boxes,)
        iou_thresh (float): IoU threshold to consider a prediction correct

    Returns:
        float: Mean average precision (mAP)
    """
    # Sort predicted boxes by score (in descending order)
    sort_inds = np.argsort(pred_scores)[::-1]
    pred_boxes = pred_boxes[sort_inds]
    pred_scores = pred_scores[sort_inds]

    # Compute true positives and false positives at each threshold
    tp = np.zeros(len(pred_boxes))
    fp = np.zeros(len(pred_boxes))
    for i, box in enumerate(pred_boxes):
        ious = np.abs(box - true_boxes) / np.maximum(1e-9, np.abs(box) + np.abs(true_boxes))
        if len(ious) > 0:
            max_iou_idx = np.argmax(ious)
            if ious[max_iou_idx] >= iou_thresh:
                if tp[max_iou_idx] == 0:
                    tp[i] = 1
                    fp[i] = 0
                else:
                    fp[i] = 1
            else:
                fp[i] = 1

    # Compute precision and recall at each threshold
    tp_cumsum = np.cumsum(tp)
    fp_cumsum = np.cumsum(fp)
    recall = tp_cumsum / len(true_boxes)
    precision = tp_cumsum / (tp_cumsum + fp_cumsum)

    # Compute AP as area under precision-recall curve
    ap = 0
    for t in np.arange(0, 1.1, 0.1):
        if np.sum(recall >= t) == 0:
            p = 0
        else:
            p = np.max(precision[recall >= t])
        ap += p / 11

    return ap


def segment_iou(target_segment, candidate_segments):
    """Compute the temporal intersection over union between a
    target segment and all the test segments.
    Parameters
    ----------
    target_segment : 1d array
        Temporal target segment containing [starting, ending] times.
    candidate_segments : 2d array
        Temporal candidate segments containing N x [starting, ending] times.
    Outputs
    -------
    tiou : 1d array
        Temporal intersection over union score of the N's candidate segments.
    """
    tt1 = np.maximum(target_segment[0], candidate_segments[:, 0])
    tt2 = np.minimum(target_segment[1], candidate_segments[:, 1])
    # Intersection including Non-negative overlap score.
    segments_intersection = (tt2 - tt1).clip(0)
    # Segment union.
    segments_union = (candidate_segments[:, 1] - candidate_segments[:, 0]) \
      + (target_segment[1] - target_segment[0]) - segments_intersection
    # Compute overlap as the ratio of the intersection
    # over union of two segments.
    tIoU = segments_intersection.astype(float) / segments_union
    return tIoU


def interpolated_prec_rec(prec, rec):
    """Interpolated AP - VOCdevkit from VOC 2011.
    """
    mprec = np.hstack([[0], prec, [0]])
    mrec = np.hstack([[0], rec, [1]])
    for i in range(len(mprec) - 1)[::-1]:
        mprec[i] = max(mprec[i], mprec[i + 1])
    idx = np.where(mrec[1::] != mrec[0:-1])[0] + 1
    ap = np.sum((mrec[idx] - mrec[idx - 1]) * mprec[idx])
    return ap


from tqdm import tqdm
def compute_average_precision_detection(
        ground_truth,
        prediction,
        tiou_thresholds=np.linspace(0.5, 0.95, 10),
    ):
    """Compute average precision (detection task) between ground truth and
    predictions data frames. If multiple predictions occurs for the same
    predicted segment, only the one with highest score is matches as
    true positive. This code is greatly inspired by Pascal VOC devkit.

    Ref: https://github.com/zhang-can/CoLA/blob/\
        d21f1b5a4c6c13f9715cfd4ac1ebcd065d179157/eval/eval_detection.py#L200

    Parameters
    ----------
    ground_truth : df
        Data frame containing the ground truth instances.
        Required fields: ['video-id', 't-start', 't-end']
    prediction : df
        Data frame containing the prediction instances.
        Required fields: ['video-id, 't-start', 't-end', 'score']
    tiou_thresholds : 1darray, optional
        Temporal intersection over union threshold.
    Outputs
    -------
    ap : float
        Average precision score.
    """
    ap = np.zeros(len(tiou_thresholds))
    if prediction.empty:
        return ap

    npos = float(len(ground_truth))
    lock_gt = np.ones((len(tiou_thresholds),len(ground_truth))) * -1
    # Sort predictions by decreasing score order.
    sort_idx = prediction['score'].values.argsort()[::-1]
    prediction = prediction.loc[sort_idx].reset_index(drop=True)

    # Initialize true positive and false positive vectors.
    tp = np.zeros((len(tiou_thresholds), len(prediction)))
    fp = np.zeros((len(tiou_thresholds), len(prediction)))

    # Adaptation to query faster
    ground_truth_gbvn = ground_truth.groupby('video-id')

    # Assigning true positive to truly grount truth instances.
    for idx, this_pred in prediction.iterrows():

        try:
            # Check if there is at least one ground truth in the video associated.
            ground_truth_videoid = ground_truth_gbvn.get_group(this_pred['video-id'])
        except Exception as e:
            fp[:, idx] = 1
            continue

        this_gt = ground_truth_videoid.reset_index()
        tiou_arr = segment_iou(this_pred[['t-start', 't-end']].values,
                               this_gt[['t-start', 't-end']].values)
        # We would like to retrieve the predictions with highest tiou score.
        tiou_sorted_idx = tiou_arr.argsort()[::-1]
        for tidx, tiou_thr in enumerate(tiou_thresholds):
            for jdx in tiou_sorted_idx:
                if tiou_arr[jdx] < tiou_thr:
                    fp[tidx, idx] = 1
                    break
                if lock_gt[tidx, this_gt.loc[jdx]['index']] >= 0:
                    continue
                # Assign as true positive after the filters above.
                tp[tidx, idx] = 1
                lock_gt[tidx, this_gt.loc[jdx]['index']] = idx
                break

            if fp[tidx, idx] == 0 and tp[tidx, idx] == 0:
                fp[tidx, idx] = 1

    tp_cumsum = np.cumsum(tp, axis=1).astype(float)
    fp_cumsum = np.cumsum(fp, axis=1).astype(float)
    recall_cumsum = tp_cumsum / npos

    precision_cumsum = tp_cumsum / (tp_cumsum + fp_cumsum)

    for tidx in range(len(tiou_thresholds)):
        ap[tidx] = interpolated_prec_rec(precision_cumsum[tidx,:], recall_cumsum[tidx,:])


    return ap


def ap_wrapper(
        true_clips,
        pred_clips,
        pred_scores,
        tiou_thresholds=np.linspace(0.5, 0.95, 10),
    ):
    assert isinstance(true_clips, np.ndarray)
    assert len(true_clips.shape) == 2 and true_clips.shape[1] == 2
    assert isinstance(pred_clips, np.ndarray)
    assert len(pred_clips.shape) == 2 and pred_clips.shape[1] == 2
    assert isinstance(pred_scores, np.ndarray)
    assert len(pred_scores.shape) == 1 and len(pred_scores) == pred_clips.shape[0]
    
    true_df = pd.DataFrame(
        {
            "video-id": ["video1"] * len(true_clips),
            "t-start": true_clips[:, 0],
            "t-end": true_clips[:, 1],
        }
    )
    pred_df = pd.DataFrame(
        {
            "video-id": ["video1"] * len(pred_clips),
            "t-start": pred_clips[:, 0],
            "t-end": pred_clips[:, 1],
            "score": pred_scores,
        }
    )
    return compute_average_precision_detection(
        true_df,
        pred_df,
        tiou_thresholds=tiou_thresholds,
    )


def nms_1d(df: pd.DataFrame, score_col="score", iou_thresh=0.5):
    """Applies NMS on 1D (start, end) box predictions."""
    columns = set(df.columns)
    # assert columns == set(["video_id", "start", "end", "score"])
    assert set(["start", "end", "video_id", score_col]).issubset(columns)
    video_ids = df["video_id"].unique()
    
    # Group by video_id
    groups = df.groupby("video_id")
    
    # Loop over videos
    keep_indices = []
    net_success_fraction = []
    tqdm._instances.clear()
    iterator = tqdm(
        video_ids,
        desc="Applying NMS to each video",
        bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}',
    )
    for video_id in iterator:

        # Get rows for this video
        rows = groups.get_group(video_id)

        # Sort by score
        rows = rows.sort_values(score_col, ascending=False)
        
        # Loop over rows until empty
        n_clips = len(rows)
        n_clips_selected_in_video = 0
        while len(rows):
            
            # Add top row to keep_indices
            top_row = rows.iloc[0]
            keep_indices.append(rows.index[0])
            n_clips_selected_in_video += 1
            top_row = top_row.to_dict()
            
            top_segment = np.array([top_row["start"], top_row["end"]])
            rows = rows.iloc[1:]
            other_segments = rows[["start", "end"]].values
            iou_values = segment_iou(top_segment, other_segments)
            
            # Remove rows IoU > iou_thresh
            rows = rows[iou_values < iou_thresh]

        net_success_fraction.append(n_clips_selected_in_video / n_clips)
    net_success_fraction = np.array(net_success_fraction).mean()
    print("> Net success fraction: {:.2f}".format(net_success_fraction))
    
    return keep_indices


if __name__ == "__main__":
    true_clips = np.array(
        [
            [0.1, 0.7],
            [3.4, 7.8],
            [3.9, 5.4],
        ]
    )
    pred_clips = np.array(
        [
            [0.2, 0.8],
            [3.5, 7.9],
            [3.9, 5.4],
            [5.6, 6.7],
            [6.0, 6.5],
        ],
    )
    pred_scores = np.array([0.9, 0.8, 0.7, 0.6, 0.5])
    
    # 1. Check IoU for a single pair of boxes
    iou = compute_iou_1d(pred_clips[0], true_clips[0])
    # Manually check that the result is correct
    # Clips are [0.1, 0.7] and [0.2, 0.8]
    # Intersection: [0.2, 0.7] - length = 0.5
    # Union: [0.1, 0.8] - length = 0.7
    # Ratio: 0.5 / 0.7 = 0.714
    assert np.isclose(iou, 0.714, 3), "Incorrect IoU"
    
    # 2. Check IoU for a single predicted box and multiple ground truth boxes
    ious = compute_iou_1d_single_candidate_multiple_targets(pred_clips[0], true_clips)
    assert np.allclose(ious, [0.714, 0.0, 0.0], 3), "Incorrect IoU"
    
    # 3. Check mean IoU for multiple predicted boxes and multiple ground truth boxes
    ious = compute_iou_1d_multiple_candidates_multiple_targets(pred_clips, true_clips)
    assert ious.shape == (5, 3), "Incorrect shape"
    
    ap = ap_wrapper(
        true_clips,
        pred_clips,
        pred_scores,
        tiou_thresholds=np.linspace(0.5, 0.95, 3),
    )
    # Take the mean of the APs across IoU thresholds
    final_ap = np.mean(ap)
    import ipdb; ipdb.set_trace()
