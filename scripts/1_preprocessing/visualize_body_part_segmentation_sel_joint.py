'''
Usage:
python3 visualize_body_part_segmentation_sel_joint.py smpl /media/brcao/eData4TB1/Repos/TransPose/models/SMPL_male.pkl
'''

import os
import sys
import json
import trimesh
import subprocess
import numpy as np
from smplx import SMPL, SMPLH, SMPLX
from matplotlib import cm as mpl_cm, colors as mpl_colors


def download_url(url, outdir):
    print(f'Downloading files from {url}')
    cmd = ['wget', '-c', url, '-P', outdir]
    subprocess.call(cmd)
    file_path = os.path.join(outdir, url.split('/')[-1])
    return file_path


def part_segm_to_vertex_colors(part_segm, n_vertices, alpha=1.0):
    vertex_labels = np.zeros(n_vertices)

    for part_idx, (k, v) in enumerate(part_segm.items()):
        vertex_labels[v] = part_idx

    cm = mpl_cm.get_cmap('jet')
    norm_gt = mpl_colors.Normalize()

    vertex_colors = np.ones((n_vertices, 4))
    vertex_colors[:, 3] = alpha
    vertex_colors[:, :3] = cm(norm_gt(vertex_labels))[:, :3]

    return vertex_colors


def main(body_model='smpl', body_model_path='body_models/smpl/'):
    main_url = 'https://raw.githubusercontent.com/Meshcapade/wiki/main/assets/SMPL_body_segmentation/'
    if body_model == 'smpl':
        part_segm_url = os.path.join(main_url, 'smpl/smpl_vert_segmentation.json')
        body_model = SMPL(model_path=body_model_path)
    elif body_model == 'smplx':
        part_segm_url = os.path.join(main_url, 'smplx/smplx_vert_segmentation.json')
        body_model = SMPLX(model_path=body_model_path)
    elif body_model == 'smplh':
        part_segm_url = os.path.join(main_url, 'smpl/smpl_vert_segmentation.json')
        body_model = SMPLH(model_path=body_model_path)
    else:
        raise ValueError(f'{body_model} is not defined, \"smpl\", \"smplh\" or \"smplx\" are valid body models')

    part_segm_filepath = download_url(part_segm_url, '.')
    part_segm = json.load(open(part_segm_filepath))
    print('\npart_segm.keys(): ', part_segm.keys())
    '''
    part_segm.keys():  dict_keys(['rightHand', 'rightUpLeg', 'leftArm', 'leftLeg', 'leftToeBase', 'leftFoot', \
        'spine1', 'spine2', 'leftShoulder', 'rightShoulder', 'rightFoot', 'head', 'rightArm', 'leftHandIndex1', \
        'rightLeg', 'rightHandIndex1', 'leftForeArm', 'rightForeArm', 'neck', 'rightToeBase', 'spine', \
        'leftUpLeg', 'leftHand', 'hips'])
    '''

    vertices = body_model().vertices[0].detach().numpy()
    faces = body_model.faces

    vertex_colors = part_segm_to_vertex_colors(part_segm, vertices.shape[0])

    print('\nlen(vertices): ', len(vertices))
    print('\nlen(vertex_colors): ', len(vertex_colors))
    # len(vertices):  6890
    # len(vertex_colors):  6890

    # print('\nvertex_colors[:10]: ', vertex_colors[:10])
    '''
    vertex_colors[:10]:  [[0.41429475 1.         0.55344719 1.        ]
                            [0.41429475 1.         0.55344719 1.        ]
                            [0.41429475 1.         0.55344719 1.        ]
                            [0.41429475 1.         0.55344719 1.        ]
                            [0.41429475 1.         0.55344719 1.        ]
                            [0.41429475 1.         0.55344719 1.        ]
                            [0.41429475 1.         0.55344719 1.        ]
                            [0.41429475 1.         0.55344719 1.        ]
                            [0.41429475 1.         0.55344719 1.        ]
                            [0.41429475 1.         0.55344719 1.        ]]
    '''
    # for i in range(len(vertex_colors)):
    #     vertex_colors[i] = [1, 1, 1, 1]

    vert_i = 4446 # edit
    vertex_colors[vert_i] = [1, 0, 0, 1]
    for i in range(len(vertex_colors)):
        if i != vert_i: vertex_colors[i] = [1, 1, 1, 0.5] 
    mesh = trimesh.Trimesh(vertices, faces, process=False, vertex_colors=vertex_colors)
    mesh.show(background=(0,0,0,0))

# 1685 - 1627 = 58
    # 1358 + 58 =
    #- 870 = 48
    # 4657 + 48 = 
    # 6432 - 6369 = 63
    # 328 + 63 = 

if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2])