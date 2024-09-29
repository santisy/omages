# Import necessary modules
import argparse
import numpy as np
from xgutils import bpyutil, geoutil, fresnelvis, visutil
import matplotlib as mpl
import matplotlib.pyplot as plt
import trimesh
from PIL import Image

parser = argparse.ArgumentParser()
parser.add_argument("--obj1", type=str, required=True)
parser.add_argument("--obj2", type=str, required=True)
parser.add_argument("--out_name", type=str, required=True)
parser.add_argument("--cam_pose", type=lambda x: [float(y) for y in x.split(",")], required=True)

args = parser.parse_args()

def compare_mesh(v1, f1, v2, f2, visual=True):
    import igl
    #hdist = igl.hausdorff(v2, f2, v1, f1)
    hd2to1, hd1to2 = geoutil.hausdorff(v2, f2, v1, f1)
    hdist = max( hd2to1.max(), hd1to2.max() )
    sqrtChamfer = hd2to1.mean() + hd1to2.mean()
    #chamfer     = (hd2to1**2).mean() + (hd1to2**2).mean()
    chamfer, sqrD1, sqrD2 = geoutil.chamfer_dist_mesh(v1, f1, v2, f2)
    # chamfer =
    ret = dict(hdist=hdist, sqrtChamfer=sqrtChamfer, chamfer=chamfer, hd2to1=hd2to1, hd1to2=hd1to2)
    ret['v1'], ret['f1'], ret['v2'], ret['f2'] = v1, f1, v2, f2
    ret['sqrD1'], ret['sqrD2'] = sqrD1, sqrD2
    return ret

def visual_error_mesh(vert, face, metric, resolution=(512,512), camPos=(1.0, 1.0, 1.0), camera_type = "perspective", shape_id=""):
    # use log scale color
    color = np.log10(1e-8 + metric)
    norm = mpl.colors.Normalize(vmin=-6, vmax=-1)
    color = mpl.cm.jet(norm(color))
    rdvert = vert[...,[0,2,1]]
    rdvert[...,2] *= -1
    errormap_img = fresnelvis.render_mesh(rdvert, face, vert_color=color[face].reshape(-1,4), specular=0., roughness=0.99, solid=1.0, render_kwargs=dict(preview=True), camera_kwargs=dict(resolution=(512,512), camPos=camPos, camera_type = camera_type))
    if shape_id == 'B082Q8NLZ3': # the sofa demo, then cut it to show its interior
        filterV = rdvert[..., 0] <= .83
        rdvert, face = geoutil.filterMesh(rdvert, face, filterV)
        color = color[filterV]
        errormap_img = fresnelvis.render_mesh(rdvert, face, vert_color=color[face].reshape(-1,4), specular=0., roughness=0.99, solid=1.0, render_kwargs=dict(preview=True), camera_kwargs=dict(resolution=(512,512), camPos=(2.8,.8,1.5), camera_type = "perspective"))

    # errormap_img = fresnelvis.render_cloud(rdv1, color=color, radius=0.008, specular=0., roughness=0.99, solid=1.0, render_kwargs=dict(preview=True), camera_kwargs=dict(resolution=(512,512), camPos=(2.18,1.5,3.2), camera_type = "perspective"))
    # using fresnel package to render the shadeless errormap. Could also achieve using blender via compositing, but it needs a bit more setup.
    ret = dict()
    ret['errormap_img'] = errormap_img
    ret['vert_color'] = color
    return ret



# Load the meshes
mesh_path_1 = args.obj1
mesh_path_2 = args.obj2
cam_pose = args.cam_pose
cam_pose_str = f"{cam_pose[0]:.1f}_{cam_pose[1]:.1f}_{cam_pose[2]:.1f}"

mesh1 = trimesh.load(mesh_path_1)
v1, f1 = mesh1.vertices, mesh1.faces

mesh2 = trimesh.load(mesh_path_2)
v2, f2 = mesh2.vertices, mesh2.faces

# Compare the meshes
ret = compare_mesh(v1, f1, v2, f2)

# Visualize the error map on mesh1
visualization_result = visual_error_mesh(
    vert=v1,
    face=f1,
    metric=ret['sqrD1'],  # Use squared distances from mesh1 to mesh2
    resolution=(512, 512),
    camPos=cam_pose,
    camera_type="perspective",
    shape_id=""
)

# Save the error map image
errormap_img = visualization_result['errormap_img']
errormap_img_uint8 = (errormap_img * 255).clip(0, 255).astype(np.uint8)
Image.fromarray(errormap_img_uint8, mode="RGBA").save(f"error_maps/error_map_{args.out_name}_camPose-{cam_pose_str}_mesh.png")

# Optionally, visualize the error map on mesh2
visualization_result2 = visual_error_mesh(
    vert=v2,
    face=f2,
    metric=ret['sqrD2'],  # Use squared distances from mesh2 to mesh1
    resolution=(512, 512),
    camPos=cam_pose,
    camera_type="perspective",
    shape_id=""
)

# Save the error map image for mesh2
errormap_img2 = visualization_result2['errormap_img']
errormap_img2_uint8 = (errormap_img2 * 255).clip(0, 255).astype(np.uint8)
Image.fromarray(errormap_img2_uint8, mode="RGBA").save(f"error_maps/error_map_{args.out_name}_camPose-{cam_pose_str}_mesh2.png")


