"""CIDS without metadata: frequency adapter for SAM (checker) + frequency council of 4 independent experts.

Pipeline
  ① SegResNet-100 (finder)  -> probability map + candidate boxes
  ③ 3D Haar wavelets        -> 8 directional sub-bands of the image
  ⑤ SamFreqAdapter          -> learned band weights -> residual added to SAM-Med3D's image embedding
  ② SAM-S2b + adapter        -> refined probability per candidate (the checker)
  ⑥ FreqCouncil              -> 4 independent band experts (E1 shape, E2 one-axis edges, E3 two-axis
                               corners, E4 texture), each = its bands + finder map + checker map -> its
                               own mask; a gate combines them -> final mask + uncertainty
  Dual supervision: ground truth + distillation towards finder and checker where they agree.
  The acquisition-geometry tokens (metadata stage) will enter through the reserved `cond` inputs.

Modules
  bands.py      band groups, band energy
  sam_crops.py  rebuild SAM's 128^3 input crop exactly as S0 did (for the adapter)
  adapter.py    SamFreqAdapter
  experts.py    BandExpert (independent small 3D U-Net per band group)
  council.py    CouncilGate
  model.py      FreqCouncil (4 experts + gate)
  losses.py     dual-supervision loss
  data.py       patches / volumes = image + finder map + checker map
"""
from .bands import BAND_GROUPS, GROUP_NAMES, band_energy, split_groups  # noqa: F401
