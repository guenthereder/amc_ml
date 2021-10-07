#!/usr/bin/env python3

from Viewer import *
import argparse


if __name__ == '__main__':
  
  parser = argparse.ArgumentParser(description='View ASF 3D Files')
  parser.add_argument('asf_path', type=str, help='path to the asf file to view')
  parser.add_argument('amc_path', type=str, help='path to the amc file to view')
  args = parser.parse_args()

  joints = parse_asf(args.asf_path)
  motions = parse_amc(args.amc_path)
  v = Viewer(joints, motions)
  v.run()
