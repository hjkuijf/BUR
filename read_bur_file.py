# -*- coding: utf-8 -*-
"""
Created on Mon Feb 10 10:20:26 2025

@author: hjkuijf
"""

from dataclasses import dataclass
import numpy as np
from pathlib import Path


@dataclass
class BUR:
    # Note: these are the fields that I could discover in the files, there are more.
    patient_initials: str
    patient_lastname: str
    patient_id: str
    patient_dateofbirth: str
    unknown_name: str
    date_of_scan: str
    time_of_scan: str
    gender: str
    scan_filename: str
    series_uid: str

    # Note: both images should be scaled *4 in the up-down direction.
    flux_array: np.array
    photo_array: np.array


def read_bur_file(bur_filename: Path) -> BUR:
    f = bur_filename.read_bytes()

    # This is only tested with V3.0 files
    assert (
        f[:64].decode("ascii").rstrip("\x00") == "moorLDI Image Data File V3.0"
    )

    # Image dimensions
    image_size_x = int.from_bytes(f[4104:4108], "little")
    image_size_y = int.from_bytes(f[4108:4112], "little")

    # String fields appear to be 128 bytes
    def read_bur_field(start):
        return f[start : start + 128].decode("ascii").rstrip("\x00")

    # The flux and photo image are at the end of the file, after the header.
    def read_bur_image(start):
        a = np.frombuffer(
            f, dtype=np.int16, count=image_size_y * image_size_x, offset=start
        )
        return np.flipud(np.reshape(a, (image_size_y, image_size_x)))

    return BUR(
        patient_initials=read_bur_field(1024),
        patient_lastname=read_bur_field(1152),
        patient_id=read_bur_field(1280),
        patient_dateofbirth=read_bur_field(1408),
        unknown_name=read_bur_field(2048),
        date_of_scan=read_bur_field(2304),
        time_of_scan=read_bur_field(2432),
        gender=read_bur_field(3584),
        scan_filename=read_bur_field(61440),
        series_uid=read_bur_field(61568),
        flux_array=read_bur_image(65544),  # start byte of flux image
        photo_array=read_bur_image(65544 + image_size_y * image_size_x * 2),
    )
