import io,unittest
import numpy as np
from PIL import Image,JpegImagePlugin
from arm_sim.camera_stream import CameraStream

class CameraResolutionTests(unittest.TestCase):
    def test_overview_reduces_compression_without_changing_resolution(self):
        stream=CameraStream(['top','wrist'],port=0,frame_options={'top':{'quality':92,'subsampling':0}})
        try:
            frame=np.random.default_rng(0).integers(0,256,(800,1200,4),dtype=np.uint8);frame[:,:,3]=255
            stream.update('top',frame)
            overview=Image.open(io.BytesIO(stream.frames['top'][0]))
            self.assertEqual(overview.size,(960,640))
            self.assertEqual(JpegImagePlugin.get_sampling(overview),0)
            stream.update('wrist',np.zeros((360,480,4),dtype=np.uint8))
            self.assertEqual(Image.open(io.BytesIO(stream.frames['wrist'][0])).size,(480,360))
            stream.update('wrist',frame)
            original=Image.open(io.BytesIO(stream.frames['wrist'][0]))
            self.assertEqual(original.size,(960,640))
            self.assertLess(sum(overview.quantization[0]),sum(original.quantization[0]))
            reference=Image.fromarray(frame).convert('RGB');reference.thumbnail((960,960))
            pixels=np.asarray(reference,dtype=float)
            self.assertLess(np.mean((np.asarray(overview,dtype=float)-pixels)**2),np.mean((np.asarray(original,dtype=float)-pixels)**2))
        finally:stream.close()
