"""Lossless metric depth and a separate, fixed-scale display image."""
import math
import cv2
import numpy as np

PREVIEW_RANGE_M = (.1, 1.5)


def preview(depth, scale):
    if depth.dtype != np.uint16 or depth.ndim != 2:
        raise ValueError('깊이 영상은 16비트 단일 채널이어야 합니다.')
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError('깊이 거리 단위를 확인하세요.')
    near, far = PREVIEW_RANGE_M
    intensity = 255*(1-np.clip((depth.astype(np.float32)*scale-near)/(far-near),0,1))
    image = cv2.applyColorMap(intensity.astype(np.uint8),cv2.COLORMAP_TURBO)
    image[depth == 0] = 0
    ok, encoded = cv2.imencode('.jpg',image,[cv2.IMWRITE_JPEG_QUALITY,85])
    if not ok:raise ValueError('깊이 미리보기 변환 실패')
    return encoded.tobytes()


def encode(depth, scale):
    display = preview(depth, scale)
    ok, encoded = cv2.imencode('.png',depth,[cv2.IMWRITE_PNG_COMPRESSION,1])
    if not ok:raise ValueError('깊이 원본 저장 변환 실패')
    return dict(png=encoded.tobytes(),preview_jpeg=display,shape=list(depth.shape),
                dtype='uint16',scale_m_per_unit=scale,invalid_value=0,aligned_to='color',
                valid_fraction=float(np.count_nonzero(depth)/depth.size))


def preview_file(path, scale):
    raw=cv2.imread(str(path),cv2.IMREAD_UNCHANGED)
    if raw is None:raise ValueError('저장된 깊이 영상을 읽을 수 없습니다.')
    return preview(raw,scale)
