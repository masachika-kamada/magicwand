import cv2
import numpy as np


SHIFT_KEY = cv2.EVENT_FLAG_SHIFTKEY
ALT_KEY = cv2.EVENT_FLAG_ALTKEY


def _find_exterior_contours(img):
    ret = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(ret) == 2:
        return ret[0]
    elif len(ret) == 3:
        return ret[1]
    raise Exception("Check the signature for `cv2.findContours()`.")


class SelectionWindow:
    def __init__(self, img, name="Magic Wand Selector", connectivity=4, tolerance=32):
        self.name = name
        h, w = img.shape[:2]
        self.img = img
        self.mask = np.zeros((h, w), dtype=np.uint8)
        self._flood_mask = np.zeros((h + 2, w + 2), dtype=np.uint8)
        self._flood_fill_flags = (
            connectivity | cv2.FLOODFILL_FIXED_RANGE | cv2.FLOODFILL_MASK_ONLY | 255 << 8
        )  # 255 << 8 tells to fill with the value 255
        cv2.namedWindow(self.name)
        self.tolerance = (tolerance,) * 3
        cv2.createTrackbar(
            "Tolerance", self.name, tolerance, 255, self._trackbar_callback
        )
        cv2.setMouseCallback(self.name, self._mouse_callback)

    def _trackbar_callback(self, pos):
        self.tolerance = (pos,) * 3

    def _mouse_callback(self, event, x, y, flags, *userdata):

        if event != cv2.EVENT_LBUTTONDOWN:
            return

        modifier = flags & (ALT_KEY + SHIFT_KEY)

        self._flood_mask[:] = 0
        cv2.floodFill(
            self.img,
            self._flood_mask,
            (x, y),
            0,
            self.tolerance,
            self.tolerance,
            self._flood_fill_flags,
        )
        flood_mask = self._flood_mask[1:-1, 1:-1].copy()

        if modifier == (ALT_KEY + SHIFT_KEY):
            self.mask = cv2.bitwise_and(self.mask, flood_mask)
        elif modifier == SHIFT_KEY:
            self.mask = cv2.bitwise_or(self.mask, flood_mask)
        elif modifier == ALT_KEY:
            self.mask = cv2.bitwise_and(self.mask, cv2.bitwise_not(flood_mask))
        else:
            self.mask = flood_mask

        self._update()

    def _update(self):
        """Updates an image in the already drawn window."""
        viz = self.img.copy()
        contours = _find_exterior_contours(self.mask)
        viz = cv2.drawContours(viz, contours, -1, color=(255,) * 3, thickness=-1)
        viz = cv2.addWeighted(self.img, 0.75, viz, 0.25, 0)
        viz = cv2.drawContours(viz, contours, -1, color=(255,) * 3, thickness=1)

        self.mean, self.stddev = cv2.meanStdDev(self.img, mask=self.mask)
        meanstr = "mean=({:.2f}, {:.2f}, {:.2f})".format(*self.mean[:, 0])
        stdstr = "std=({:.2f}, {:.2f}, {:.2f})".format(*self.stddev[:, 0])
        cv2.imshow(self.name, viz)
        # cv2.displayStatusBar(self.name, ", ".join((meanstr, stdstr)))
        print(", ".join((meanstr, stdstr)))

    def show(self):
        """Draws a window with the supplied image."""
        self._update()
        print("Press [q] or [esc] to close the window.")
        while True:
            k = cv2.waitKey() & 0xFF
            if k in (ord("q"), ord("\x1b")):
                cv2.destroyWindow(self.name)
                break
