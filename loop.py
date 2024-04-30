import os
import argparse

BASE_CMD = "python main.py --model lf"
SOURCE_DIR = "./Insomniac_Spiderman1_Interaction_Inidcator_02"
OUTPUT_DIR = "./output"
SKIP_INTERVAL = 2


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
  if vel:
    vel_dir = input + "/vel"
    src_vels = os.listdir(vel_dir)
  if dep:
    dep_dir = input + "/dep"
    src_deps = os.listdir(dep_dir)

  for i in range(0, len(src_imgs) - interval, interval):
    img_1 = os.path.join(img_dir, src_imgs[i])
    img_2 = os.path.join(img_dir, src_imgs[i + interval])

    cmd = f"{BASE_CMD} --one {img_1} --two {img_2}"

    if vel:
      vel_1 = os.path.join(vel_dir, src_vels[int(i / interval)])
      vel_2 = os.path.join(vel_dir, src_vels[int(i / interval) + 1])
      cmd += f" --floOne {vel_1} --floTwo {vel_2}"

    if dep:
      dep_1 = os.path.join(dep_dir, src_deps[int(i / interval)])
      dep_2 = os.path.join(dep_dir, src_deps[int(i / interval) + 1])
      cmd += f" --depOne {dep_1} --depTwo {dep_2}"

    # Generate output file name
    i_interp = (2 * i + interval) / 2
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
