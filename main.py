#!/usr/bin/env python

import getopt
import PIL
import PIL.Image
import sys
import numpy as np
import torch
import os

from utils import interactive_warnings
from utils import read_exr, read_tiff

from network import estimate
##########################################################

args_strModel = "lf"
args_strOne = "./images/one.png"
args_strTwo = "./images/two.png"
args_strVideo = "./videos/car-turn.mp4"
args_strOut = "./out.png"

ARGS_LONGOPTS = [
  "model=",
  "one=",
  "two=",
  "floOne=",
  "floTwo=",
  "depOne=",
  "depTwo=",
  "video=",
  "out=",
]

for strOption, strArg in getopt.getopt(sys.argv[1:], "", ARGS_LONGOPTS)[0]:
  if strArg != "":
    match strOption:
      case "--model":
        args_strModel = strArg  # which model to use
      case "--one":
        args_strOne = strArg  # path to the first frame
      case "--two":
        args_strTwo = strArg  # path to the second frame
      case "--floOne":
        args_strFloOne = strArg  # path to the velocity frame
      case "--floTwo":
        args_strFloTwo = strArg  # path to the velocity frame
      case "--depOne":
        args_strDepOne = strArg  # path to the depth frame
      case "--depTwo":
        args_strDepTwo = strArg  # path to the depth frame
      case "--video":
        args_strVideo = strArg  # path to a video
      case "--out":
        args_strOut = strArg  # path to where the output should be stored

##########################################################

if __name__ == "__main__":
  if args_strOut.split(".")[-1] in ["bmp", "jpg", "jpeg", "png"]:
    tenOne = torch.FloatTensor(np.ascontiguousarray(np.array(PIL.Image.open(args_strOne).convert("RGB"))[:, :, ::-1].transpose(2, 0, 1).astype(np.float32) * (1.0 / 255.0)))
    tenTwo = torch.FloatTensor(np.ascontiguousarray(np.array(PIL.Image.open(args_strTwo).convert("RGB"))[:, :, ::-1].transpose(2, 0, 1).astype(np.float32) * (1.0 / 255.0)))

    # CUSTOM FLO from GameEngine
    try:
      tenFloOne = read_exr(args_strFloOne)
      tenFloTwo = read_exr(args_strFloTwo)
    except Exception:
      interactive_warnings("Custom Velocity ignored, PWCnet is used.")
      tenFloOne, tenFloTwo = None, None

    # CUSTOM DEPTH from GameEngine
    try:
      tenDepOne = read_tiff(args_strDepOne)
      tenDepTwo = read_tiff(args_strDepTwo)
    except Exception:
      interactive_warnings("Custom Depth ignored, Softmetric is used.")
      tenDepOne, tenDepTwo = None, None

    tenOutput = estimate(tenOne, tenTwo, tenFloOne, tenFloTwo, tenDepOne, tenDepTwo, [0.5])[0]

    directory = "./output"
    if not os.path.exists(directory):
      os.makedirs(directory)

    PIL.Image.fromarray((tenOutput.clip(0.0, 1.0).numpy().transpose(1, 2, 0)[:, :, ::-1] * 255.0).astype(np.uint8)).save(args_strOut)
# end
