"""Reuse preview colors/textures with neutral indoor fluorescent area lighting."""
import numpy as np
from pxr import UsdGeom, UsdShade, UsdLux, Sdf, Gf

def linear_rgb(color):
    color=np.array(color,dtype=float)
    return tuple(np.where(color<=.04045,color/12.92,((color+.055)/1.055)**2.4).tolist())

def finish_material(cell, material, color, surface=None, visual=None):
    surface=surface or {}
    path=str(material.GetPath())
    shader=UsdShade.Shader(cell.stage.GetPrimAtPath(path+'/Shader'))
    shader.CreateInput('diffuseColor',Sdf.ValueTypeNames.Color3f).Set(linear_rgb(color))
    shader.CreateInput('roughness',Sdf.ValueTypeNames.Float).Set(float(surface.get('roughness',.9)))
    shader.CreateInput('useSpecularWorkflow',Sdf.ValueTypeNames.Int).Set(1)
    spec=float(surface.get('specular',.05))
    shader.CreateInput('specularColor',Sdf.ValueTypeNames.Color3f).Set((spec,spec,spec))
    emission=float(surface.get('emission',0))
    shader.CreateInput('emissiveColor',Sdf.ValueTypeNames.Color3f).Set(tuple(v*emission for v in linear_rgb(color)))
    if surface.get('texture'):
        tex=UsdShade.Shader.Define(cell.stage,path+'/Texture');tex.CreateIdAttr('UsdUVTexture')
        tex.CreateInput('file',Sdf.ValueTypeNames.Asset).Set(surface['texture'])
        tex.CreateInput('sourceColorSpace',Sdf.ValueTypeNames.Token).Set('sRGB')
        tex.CreateInput('wrapS',Sdf.ValueTypeNames.Token).Set('repeat')
        tex.CreateInput('wrapT',Sdf.ValueTypeNames.Token).Set('repeat')
        reader=UsdShade.Shader.Define(cell.stage,path+'/UV');reader.CreateIdAttr('UsdPrimvarReader_float2')
        reader.CreateInput('varname',Sdf.ValueTypeNames.Token).Set('st')
        tex.CreateInput('st',Sdf.ValueTypeNames.Float2).ConnectToSource(reader.ConnectableAPI(),'result')
        shader.CreateInput('diffuseColor',Sdf.ValueTypeNames.Color3f).ConnectToSource(tex.ConnectableAPI(),'rgb')
        geom=UsdGeom.Mesh(visual)
        points=np.asarray(geom.GetPointsAttr().Get(),float)
        tf=np.array(surface['world_from_texture'])
        local=(points-tf[:3,3])@tf[:3,:3]
        uv=local[:,:2]*np.array(surface['texrepeat'])
        primvar=UsdGeom.PrimvarsAPI(visual).CreatePrimvar('st',Sdf.ValueTypeNames.TexCoord2fArray,UsdGeom.Tokens.vertex)
        primvar.Set(uv.tolist())
    return material

def indoor_lights(stage):
    # Geometry/intensity are a reproducible room-light approximation, not a lux
    # measurement. Long overhead emitters create the broad shadows of fixtures.
    ambient=UsdLux.DomeLight.Define(stage,'/World/Lighting/RoomBounce')
    ambient.CreateIntensityAttr(600.)
    ambient.CreateColorAttr(Gf.Vec3f(.88,.91,.95))
    for i,x in enumerate((-.15,.6)):
        light=UsdLux.RectLight.Define(stage,f'/World/Lighting/Fluorescent{i}')
        light.CreateWidthAttr(1.2);light.CreateHeightAttr(.10)
        light.CreateIntensityAttr(18000.)
        light.CreateEnableColorTemperatureAttr(True);light.CreateColorTemperatureAttr(5000.)
        light.AddTranslateOp().Set(Gf.Vec3d(x,.05,1.7))
    return dict(kind='indoor fluorescent approximation',color_temperature_k=5000,
                ceiling_emitters=2,emitter_size_m=[1.2,.1],emitter_z_m=1.7,
                ambient_intensity=600.,ceiling_intensity=18000.,measured_lux=False)
