import smplx

# Load the SMPL model to get the face list
model_path = '/media/brcao/eData4TB1/Repos/IMUPoser_bryanbocao/IMUPoser/src/imuposer/smpl/basicmodel_m_lbs_10_207_0_v1.0.0.pkl' # 'path/to/SMPL_NEUTRAL.pkl'
model_path = '/media/brcao/eData4TB1/Repos/TransPose/models/SMPL_male.pkl'

smpl = smplx.create(model_path, model_type='smpl')
faces = smpl.faces  # List of faces
print(dir(smpl))

import smplx
import torch

# Specify the path to the SMPL model file
# model_path = 'path/to/SMPL_NEUTRAL.pkl'  # Update with the actual path

# # Load the SMPL model
# smpl = smplx.create(model_path, model_type='smpl', gender='neutral', use_pca=False)

# # Specify body shape and pose parameters (can be zero for the neutral shape)
# num_betas = 10  # Number of shape coefficients
# betas = torch.zeros([1, num_betas])  # Shape parameters
# body_pose = torch.zeros([1, 69])  # Body pose parameters (69 for SMPL)

# # Forward pass through SMPL to get the output vertices
# output = smpl(betas=betas, body_pose=body_pose, global_orient=torch.zeros([1, 3]))
# vertices = output.vertices[0].detach().cpu().numpy()  # Convert vertices to NumPy array

# # Print the vertex list
# print("SMPL Vertex List:")
# print(vertices)