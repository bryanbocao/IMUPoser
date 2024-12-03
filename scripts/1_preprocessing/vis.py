from aitviewer.renderables.smpl import SMPLSequence
from aitviewer.viewer import Viewer

if __name__ == '__main__':
    v = Viewer()
    v.scene.add(SMPLSequence.t_pose())
    v.run()