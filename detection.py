import torch
import pandas as pd
from pathlib import Path
import numpy as np
from fastprogress import progress_bar
import warnings
from contextlib import contextmanager
import time

from src.models import models, AttBlock
from src.preproc import clip_to_image

PERIOD = 30
# Arbitrary
ratio = {
    "ref2_th03": 0.25/0.77,
    "ref2_th04": 0.14/0.77,
    "eff_th04": 0.13/0.77,
    "ext": 0.25/0.77
}
all_time_duration = 0
# We may determine tresholds for each class but it's not done here
thresholds = {}
inv_bird_call = np.load('inv_bird_code.npy', allow_pickle=True)


@contextmanager
def timer(name: str):
    t0 = time.time()
    msg = f"[{name}] start"
    print(msg)
    yield
    global all_time_duration
    all_time_duration += time.time() - t0
    msg = f"[{name}] done in {time.time() - t0:.2f} s"
    print(msg)


def prediction_for_clip(test_df: pd.DataFrame,
                        clip: Path, models):
    """Given a clip, the function predict the bird singing"""
    images = clip_to_image(clip)
    array = np.asarray(images)
    tensors = torch.from_numpy(array)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    estimated_event_list = []
    global_time = 0.0
    audio_id = test_df["audio_id"].values[0]
    for image in progress_bar(tensors):
        image = image[None, :]/255.0
        image = image.to(device)
        outputs = {}
        with torch.no_grad():
            for key in models:
                prediction = models[key](image)
                framewise_outputs = prediction["framewise_output"].detach(
                ).cpu().numpy()[0]
                outputs[key] = framewise_outputs

        key = list(outputs.keys())[0]
        framewise_outputs = np.zeros_like(outputs[key], dtype=np.float32)
        for key in outputs:
            framewise_outputs += ratio[key] * outputs[key]
        thresholded = np.zeros_like(framewise_outputs)
        for i in range(len(inv_bird_call)):
            thresholded[:, i] = framewise_outputs[:, i] >= 0.01
            # thresholded[:, i] = framewise_outputs[:, i] >= thresholds[INV_BIRD_CODE[i]] # uncoment if there is personalized tresholds
        sec_per_frame = PERIOD / thresholded.shape[0]
        for target_idx in range(thresholded.shape[1]):
            if thresholded[:, target_idx].mean() == 0:
                pass
            else:
                detected = np.argwhere(thresholded[:, target_idx]).reshape(-1)
                head_idx = 0
                tail_idx = 0
                while True:
                    if (tail_idx + 1 == len(detected)) or (
                            detected[tail_idx + 1] -
                            detected[tail_idx] != 1):
                        onset = sec_per_frame * detected[
                            head_idx] + global_time
                        offset = sec_per_frame * detected[
                            tail_idx] + global_time
                        onset_idx = detected[head_idx]
                        offset_idx = detected[tail_idx]
                        max_confidence = framewise_outputs[
                            onset_idx:offset_idx, target_idx].max()
                        mean_confidence = framewise_outputs[
                            onset_idx:offset_idx, target_idx].mean()
                        estimated_event = {
                            "audio_id": audio_id,
                            "ebird_code": inv_bird_call[target_idx],
                            "onset": onset,
                            "offset": offset,
                            "max_confidence": max_confidence,
                            "mean_confidence": mean_confidence
                        }
                        estimated_event_list.append(estimated_event)
                        head_idx = tail_idx + 1
                        tail_idx = tail_idx + 1
                        if head_idx >= len(detected):
                            break
                    else:
                        tail_idx += 1
        global_time += PERIOD
    prediction_df = pd.DataFrame(estimated_event_list)
    return prediction_df


def prediction(test_df: pd.DataFrame,
               models):
    """"given the pass of a folder containing audios and a csv corresponding, it returns a prediction for each audio which need a postprocess"""
    unique_audio_id = test_df.audio_id.unique()
    warnings.filterwarnings("ignore")
    prediction_dfs = []
    for audio_id in unique_audio_id:
        clip_path = audio_id
        test_df_for_audio_id = test_df.query(
            f"audio_id == '{audio_id}'").reset_index(drop=True)
        with timer(f"Prediction & load on {audio_id}"):
            prediction_df = prediction_for_clip(test_df_for_audio_id,
                                                clip=clip_path,
                                                models=models)

        prediction_dfs.append(prediction_df)

    prediction_df = pd.concat(prediction_dfs, axis=0,
                              sort=False).reset_index(drop=True)
    return prediction_df


def postproc(prediction_df, test):
    """Make the postprocessing"""
    labels = {}

    for audio_id, sub_df in prediction_df.groupby("audio_id"):
        events = sub_df[["ebird_code", "mean_confidence"]].values
        n_events = len(events)
        bird_max_conf = np.max(events[:, 1])
        for i in range(n_events):
            if events[i][1] == bird_max_conf:
                row_id = f"{audio_id}"
                bird = events[i][0]
                labels[row_id] = {bird, ""}
    for key in labels:
        labels[key] = " ".join(sorted(list(labels[key])))

    row_ids = list(labels.keys())
    birds = list(labels.values())
    post_processed = pd.DataFrame({
        "audio_id": row_ids,
        "birds": birds})
    all_row_id = test[["audio_id"]]
    submission = all_row_id.merge(post_processed, on="audio_id", how="left")
    submission = submission.fillna("nocall")
    return submission


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for model in models:  # load the model
        num_ftrs = models[model].fc1.in_features
        models[model].att_block = AttBlock(
            num_ftrs, len(inv_bird_call), activation="sigmoid")
        models[model].load_state_dict(torch.load(
            'weights_trained/'+model+'.pth'))
        models[model].to(device)

    test = pd.read_csv("input/test.csv",encoding="utf-8")
    test["audio_id"] = test["audio_id"].map(str)

    prediction_df = prediction(
        test_df=test, models=models)
    post_processed = prediction_df

    if not prediction_df.empty:
        post_processed = postproc(prediction_df, test)
    all_row_id = test[["audio_id"]]
    submission = all_row_id.merge(post_processed, on="audio_id", how="left")
    submission.to_csv("output/submission.csv", index=False)
    print(f"all done in {all_time_duration:.2f} s")
