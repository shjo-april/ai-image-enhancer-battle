# Copyright 2026 Sanghyun Jo. Licensed under Apache 2.0.
# DanbooruMOAT: ONNX-based image tagger for content rating and general tags.

import onnxruntime
import numpy as np
import pandas as pd
from PIL import Image


class DanbooruMOAT:
    """MOAT-based image tagger that predicts content rating, general tags, and characters."""

    def __init__(self, onnx_path="model.onnx", general_th=0.35, general_mcut=False,
                 character_th=0.85, character_mcut=False):
        self.tags, self.rating_indices, self.general_indices, self.character_indices = \
            self._load_labels(onnx_path.replace(".onnx", ".csv"))

        self.model = onnxruntime.InferenceSession(onnx_path)
        _, self.height, self.width, _ = self.model.get_inputs()[0].shape

        self.general_th = general_th
        self.general_mcut = general_mcut
        self.character_th = character_th
        self.character_mcut = character_mcut

    @staticmethod
    def _load_labels(csv_path):
        kaomojis = [
            "0_0", "(o)_(o)", "+_+", "+_-", "._.", "<o>_<o>", "<|>_<|>",
            "=_=", ">_<", "3_3", "6_9", ">_o", "@_@", "^_^", "o_o",
            "u_u", "x_x", "|_|", "||_||",
        ]
        df = pd.read_csv(csv_path)
        tags = df["name"].map(lambda x: x.replace("_", " ") if x not in kaomojis else x).tolist()
        rating_indices = list(np.where(df["category"] == 9)[0])
        general_indices = list(np.where(df["category"] == 0)[0])
        character_indices = list(np.where(df["category"] == 4)[0])
        return tags, rating_indices, general_indices, character_indices

    @staticmethod
    def _mcut_threshold(probs: np.ndarray):
        sorted_probs = probs[probs.argsort()[::-1]]
        difs = sorted_probs[:-1] - sorted_probs[1:]
        t = difs.argmax()
        return (sorted_probs[t] + sorted_probs[t + 1]) / 2

    def _preprocess(self, image: Image.Image):
        if image.mode == "RGBA":
            canvas = Image.new("RGBA", image.size, (255, 255, 255))
            canvas.alpha_composite(image)
            image = canvas
        image = image.convert("RGB")

        # Pad to square then resize
        max_dim = max(image.size)
        pad_left = (max_dim - image.size[0]) // 2
        pad_top = (max_dim - image.size[1]) // 2
        padded = Image.new("RGB", (max_dim, max_dim), (255, 255, 255))
        padded.paste(image, (pad_left, pad_top))
        if max_dim not in (self.width, self.height):
            padded = padded.resize((self.width, self.height), Image.BICUBIC)

        arr = np.asarray(padded, dtype=np.float32)[:, :, ::-1]  # RGB to BGR
        return arr[np.newaxis]

    def predict(self, image: Image.Image):
        """Return (rating_dict, general_tags_dict, character_tags_dict)."""
        input_name = self.model.get_inputs()[0].name
        label_name = self.model.get_outputs()[0].name
        preds = self.model.run([label_name], {input_name: self._preprocess(image)})[0]
        labels = list(zip(self.tags, preds[0].astype(np.float32)))

        rating = dict([labels[i] for i in self.rating_indices])

        # General tags
        general_tags = [labels[i] for i in self.general_indices]
        th = self.general_th
        if self.general_mcut:
            th = self._mcut_threshold(np.array([x[1] for x in general_tags]))
        general_tags = dict(sorted(
            [x for x in general_tags if x[1] > th],
            key=lambda x: x[1], reverse=True,
        ))

        # Character tags
        character_tags = [labels[i] for i in self.character_indices]
        th = self.character_th
        if self.character_mcut:
            th = max(0.15, self._mcut_threshold(np.array([x[1] for x in character_tags])))
        character_tags = dict(sorted(
            [x for x in character_tags if x[1] > th],
            key=lambda x: x[1], reverse=True,
        ))

        return rating, general_tags, character_tags
