"""The registered arm2 top camera, including saved OpenCV intrinsics."""
import numpy as np
from scipy.spatial.transform import Rotation
from core import atomic_json


def top_camera(env, *, select=False):
    from isaacsim.core.utils.extensions import enable_extension
    from pxr import UsdGeom, Usd
    enable_extension('isaacsim.sensors.camera')
    from isaacsim.sensors.camera import Camera
    intrinsics=env.ref.profile['intrinsics'];extrinsics=env.ref.profile['extrinsics']
    size=tuple(intrinsics['size']);matrix=np.asarray(intrinsics['K'])
    tf=np.asarray(extrinsics['base_from_camera'],float)
    # OpenCV: +X right, +Y down, +Z forward. USD: +Y up, -Z forward.
    rotation=tf[:3,:3]@np.diag([1.,-1.,-1.])
    camera=getattr(env,'top_camera',None)
    if camera is None:
        camera=Camera('/World/TopCamera',name='registered_top_camera',resolution=size)
        env.top_camera=camera
    camera.set_world_pose(tf[:3,3]/1000,Rotation.from_matrix(rotation).as_quat()[[3,0,1,2]],camera_axes='usd')
    optics=UsdGeom.Camera(camera.prim)
    optics.CreateFocalLengthAttr(24.)
    optics.CreateHorizontalApertureAttr(float(24*size[0]/matrix[0,0]))
    optics.CreateVerticalApertureAttr(float(24*size[1]/matrix[1,1]))
    camera.set_opencv_pinhole_properties(cx=float(matrix[0,2]),cy=float(matrix[1,2]),
        fx=float(matrix[0,0]),fy=float(matrix[1,1]),pinhole=intrinsics['D'])
    # The CAD optical origin lies inside its opaque 20 mm display lens.
    # Clip that near geometry, without moving the registered optical center.
    camera.set_clipping_range(.03,10.)
    if select:
        from omni.kit.viewport.utility import get_active_viewport
        viewport=get_active_viewport()
        if viewport is None:raise RuntimeError('상단 카메라를 표시할 뷰포트가 없습니다.')
        viewport.set_active_camera('/World/TopCamera')
        viewport.fill_frame=False;viewport.set_texture_resolution(size)
    actual=np.asarray(UsdGeom.Xformable(camera.prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())).T
    if not np.allclose(actual[:3,:3],rotation,atol=1e-6) or not np.allclose(actual[:3,3],tf[:3,3]/1000,atol=1e-6):
        raise RuntimeError('등록된 카메라 좌표와 표시 카메라가 일치하지 않습니다.')
    atomic_json(env.output/'top_camera.json',dict(source='arm2 profile intrinsics/extrinsics',
        base_from_camera_mm=tf.tolist(),resolution=list(size),K=intrinsics['K'],D=intrinsics['D'],
        lens_model='OpenCV pinhole distortion',pose_source=extrinsics.get('source'),
        near_clip_m=.03,
        pose_measured=bool(extrinsics.get('verified',False)),scene_pose_matches_profile=True,
        camera_prim='/World/TopCamera'))
    return camera
