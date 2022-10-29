#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (c) 2021 kumattau
#
# Use of this source code is governed by a MIT License
#

"""
video capture example using python-v4l2
"""

import fcntl
import mmap
import selectors
import sys

import cv2
import numba.cuda
import numpy as np

import v4l2


def main():
    DEVICE = "/dev/video0"
    BUF_TYPE = v4l2.V4L2_BUF_TYPE_VIDEO_CAPTURE
    FMT_TYPE = v4l2.V4L2_PIX_FMT_MJPEG
    WIDTH = 1280
    HEIGHT = 720
    REQ_COUNT = 4

    if 1 < len(sys.argv) and sys.argv[1] == "--userptr":
        MEM_TYPE = v4l2.V4L2_MEMORY_USERPTR
        mode = "USERPTR"
    else:
        MEM_TYPE = v4l2.V4L2_MEMORY_MMAP
        mode = "MMAP"
    print(f"{sys.argv[0]}: Start capture with {mode} mode. Press ESC to exit")

    fd = open(DEVICE, "rb+", buffering=0)

    fmt = v4l2.v4l2_format()
    fmt.type = BUF_TYPE
    fmt.fmt.pix.width = WIDTH
    fmt.fmt.pix.height = HEIGHT
    fmt.fmt.pix.pixelformat = FMT_TYPE
    fcntl.ioctl(fd, v4l2.VIDIOC_S_FMT, fmt)

    def decode_fourcc(fourcc):
        return "".join(fourcc.to_bytes(4, "little").decode("utf-8"))

    print("Camera spec: "
          f"width={fmt.fmt.pix.width},"
          f"height={fmt.fmt.pix.height},"
          f"pixelformat={decode_fourcc(fmt.fmt.pix.pixelformat)}")

    req = v4l2.v4l2_requestbuffers()
    req.type = BUF_TYPE
    req.count = REQ_COUNT
    req.memory = MEM_TYPE
    fcntl.ioctl(fd, v4l2.VIDIOC_REQBUFS, req)

    mems = []
    for i in range(req.count):
        buf = v4l2.v4l2_buffer()
        buf.type = BUF_TYPE
        buf.memory = MEM_TYPE
        buf.index = i
        fcntl.ioctl(fd, v4l2.VIDIOC_QUERYBUF, buf)

        if MEM_TYPE == v4l2.V4L2_MEMORY_USERPTR:
            arr = numba.cuda.pinned_array((buf.length,), dtype="uint8")
            buf.m.userptr = arr.__array_interface__["data"][0]
        else:
            mem = mmap.mmap(
                fd.fileno(), buf.length,
                flags=mmap.MAP_SHARED, prot=mmap.PROT_READ, offset=buf.m.offset)
            arr = np.frombuffer(mem, dtype="uint8")

        mems.append(arr)
        fcntl.ioctl(fd, v4l2.VIDIOC_QBUF, buf)

    selector = selectors.DefaultSelector()
    selector.register(fd, selectors.EVENT_READ)
    # convert bytes to pass int *, not int
    fcntl.ioctl(fd, v4l2.VIDIOC_STREAMON, BUF_TYPE.to_bytes(8, "little"))

    buf = v4l2.v4l2_buffer()

    while True:
        _ = selector.select()

        buf.type = BUF_TYPE
        buf.memory = MEM_TYPE
        fcntl.ioctl(fd, v4l2.VIDIOC_DQBUF, buf)

        arr = mems[buf.index]
        img = cv2.imdecode(arr, cv2.IMREAD_UNCHANGED)
        fcntl.ioctl(fd, v4l2.VIDIOC_QBUF, buf)

        cv2.imshow(DEVICE, img)
        if cv2.waitKey(1) == 27:
            break


if __name__ == "__main__":
    main()
