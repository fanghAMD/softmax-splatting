import os
import argparse

BASE_CMD = "python main.py --model lf"
SOURCE_DIR = "./Insomniac_Spiderman1_Interaction_Inidcator_02"
OUTPUT_DIR = "./output"
SKIP_INTERVAL = 2


def extract_frame_index(path):
  index = None

  for fileNameSegment in os.path.basename(path).split("_"):
    try:
      index = int(fileNameSegment)
    except Exception:
      continue

  return index


def generate_flowdepth_path(index, input=SOURCE_DIR):
  vel_file = f"{input}/vel/{index:06d}_vel.exr"
  dep_file = f"{input}/dep/{index:06d}_dep.tif"

  return vel_file, dep_file


def loop(input=SOURCE_DIR, output=OUTPUT_DIR, interval=SKIP_INTERVAL, vel=False, dep=False):
  # Create subdirectory based on config
  out_dir = output + "/output_using"
  if vel:
    out_dir += "Flow"
  if dep:
    out_dir += "Depth"
  if not vel and not dep:
    out_dir += "None"
  os.makedirs(out_dir, exist_ok=True)

  # Load list of files
  img_dir = input + "/col"
  src_imgs = os.listdir(img_dir)

  for i in range(0, len(src_imgs) - interval, interval):
    img_1 = os.path.join(img_dir, src_imgs[i])
    img_2 = os.path.join(img_dir, src_imgs[i + interval])

    cmd = f"{BASE_CMD} --one {img_1} --two {img_2}"
    
    idx_1 = extract_frame_index(img_1)
    idx_2 = extract_frame_index(img_2)
    vel_1, dep_1 = generate_flowdepth_path(idx_1, input=input)
    vel_2, dep_2 = generate_flowdepth_path(idx_2, input=input)

    if vel:
      assert os.path.exists(vel_1), f"File {vel_1} could not be found."
      assert os.path.exists(vel_2), f"File {vel_2} could not be found."
      cmd += f" --floOne {vel_1} --floTwo {vel_2}"

    if dep:
      assert os.path.exists(dep_1), f"File {dep_1} could not be found."
      assert os.path.exists(dep_2), f"File {dep_2} could not be found."
      cmd += f" --depOne {dep_1} --depTwo {dep_2}"

    # Generate output file name
    i_interp = (idx_1 + idx_2) / 2
    file_out = f"{i_interp:08.1f}" if i_interp % 1 else f"{int(i_interp):06d}"
    file_out = f"{out_dir}/{file_out}_interp.png"
    cmd += f" --out {file_out}"

    print(cmd)
    os.system(cmd)


if __name__ == "__main__":
  parser = argparse.ArgumentParser(description="Loop through directory option")
  parser.add_argument("-i", "--input", help="Input directory with all source data.", type=str)
  parser.add_argument("-0", "--output", help="Output directory with all source data.", type=str)

  parser.add_argument("-ii", "--interval", help="Image file sampling interval.", type=int)

  parser.add_argument("-v", "--vel", help="Use velocity files.", action="store_true")
  parser.add_argument("-d", "--dep", help="Use depth files.", action="store_true")

  args = parser.parse_args()
  config = vars(args)

  loop(**{key: val for key, val in config.items() if val is not None})
