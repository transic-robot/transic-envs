import os
import time
import cv2
import numpy as np


class Cv2Display:
    def __init__(
        self,
        window_name="display",
        image_size=None,
        channel_order="auto",
        bgr2rgb=True,
        step_sleep=0,
        enabled=True,
    ):
        """
        Use cv2.imshow() to pop a window, requires virtual desktop GUI

        Args:
            channel_order: auto, hwc, or chw
            image_size: None to use the original image size, otherwise resize
            step_sleep: sleep for a few seconds
        """
        self._window_name = window_name
        if isinstance(image_size, int):
            image_size = (image_size, image_size)
        else:
            assert image_size is None or len(image_size) == 2
        self._image_size = image_size
        assert channel_order in ["auto", "chw", "hwc"]
        self._channel_order = channel_order
        self._bgr2rgb = bgr2rgb
        self._step_sleep = step_sleep
        self._enabled = enabled

    def _resize(self, img):
        if self._image_size is None:
            return img
        H, W = img.shape[:2]
        Ht, Wt = self._image_size  # target
        return cv2.resize(
            img,
            self._image_size,
            interpolation=cv2.INTER_AREA if Ht < H else cv2.INTER_LINEAR,
        )

    def _reorder(self, img):
        if self._channel_order == "chw":
            return np.transpose(img, (1, 2, 0))
        elif self._channel_order == "hwc":
            return img
        else:
            if img.shape[0] in [1, 3]:  # chw
                return np.transpose(img, (1, 2, 0))
            else:
                return img

    def __call__(self, img):
        if not self._enabled:
            return
        import torch

        # prevent segfault in IsaacGym
        display_var = os.environ.get("DISPLAY", None)
        if not display_var:
            os.environ["DISPLAY"] = ":0.0"

        if torch.is_tensor(img):
            img = img.detach().cpu().numpy()

        img = self._resize(self._reorder(img))
        if self._bgr2rgb:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        time.sleep(self._step_sleep)
        cv2.imshow(self._window_name, img)
        cv2.waitKey(1)

        if display_var is not None:
            os.environ["DISPLAY"] = display_var

    def close(self):
        if not self._enabled:
            return
        cv2.destroyWindow(self._window_name)
