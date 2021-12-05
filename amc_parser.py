#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
from transforms3d.euler import euler2mat
from mpl_toolkits.mplot3d import Axes3D
import argparse
import os

from Joint import *


def read_line(stream, idx):
  if idx >= len(stream):
    return None, idx
  line = stream[idx].strip().split()
  idx += 1
  return line, idx


def parse_asf(file_path):
  '''read joint data only'''
  with open(file_path) as f:
    content = f.read().splitlines()

  for idx, line in enumerate(content):
    # meta infomation is ignored
    if line == ':bonedata':
      content = content[idx+1:]
      break

  # read joints
  joints = {'root': Joint('root', np.zeros(3), 0, np.zeros(3), [], [])}
  idx = 0
  while True:
    # the order of each section is hard-coded

    line, idx = read_line(content, idx)

    if line[0] == ':hierarchy':
      break

    assert line[0] == 'begin'

    line, idx = read_line(content, idx)
    assert line[0] == 'id'

    line, idx = read_line(content, idx)
    assert line[0] == 'name'
    name = line[1]

    line, idx = read_line(content, idx)
    assert line[0] == 'direction'
    direction = np.array([float(axis) for axis in line[1:]])

    # skip length
    line, idx = read_line(content, idx)
    assert line[0] == 'length'
    length = float(line[1])

    line, idx = read_line(content, idx)
    assert line[0] == 'axis'
    assert line[4] == 'XYZ'

    axis = np.array([float(axis) for axis in line[1:-1]])

    dof = []
    limits = []

    line, idx = read_line(content, idx)
    if line[0] == 'dof':
      dof = line[1:]
      for i in range(len(dof)):
        line, idx = read_line(content, idx)
        if i == 0:
          assert line[0] == 'limits'
          line = line[1:]
        assert len(line) == 2
        mini = float(line[0][1:])
        maxi = float(line[1][:-1])
        limits.append((mini, maxi))

      line, idx = read_line(content, idx)

    assert line[0] == 'end'
    joints[name] = Joint(
      name,
      direction,
      length,
      axis,
      dof,
      limits
    )

  # read hierarchy
  assert line[0] == ':hierarchy'

  line, idx = read_line(content, idx)

  assert line[0] == 'begin'

  while True:
    line, idx = read_line(content, idx)
    if line[0] == 'end':
      break
    assert len(line) >= 2
    for joint_name in line[1:]:
      joints[line[0]].children.append(joints[joint_name])
    for nm in line[1:]:
      joints[nm].parent = joints[line[0]]

  return joints


def parse_amc(file_path):
  with open(file_path) as f:
    content = f.read().splitlines()

  for idx, line in enumerate(content):
    if line == ':DEGREES':
      content = content[idx+1:]
      break

  frames = []
  idx = 0
  line, idx = read_line(content, idx)
  assert line[0].isnumeric(), line
  EOF = False
  while not EOF:
    joint_degree = {}
    while True:
      line, idx = read_line(content, idx)
      if line is None:
        EOF = True
        break
      if line[0].isnumeric():
        break
      joint_degree[line[0]] = [float(deg) for deg in line[1:]]
    frames.append(joint_degree)
  return frames


if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='View ASF 3D Files')
  parser.add_argument('asf_path', type=str, help='path to the asf file to view')
  parser.add_argument('amc_path', type=str, help='path to the amc file to view')
  args = parser.parse_args()

  print('parsing %s' % args.asf_path)
  joints = parse_asf(args.asf_path)
  motions = parse_amc(args.amc_path)
  joints['root'].set_motion(motions[0])
  joints['root'].draw()


  # lv0 = './data/subject'
  # lv1s = os.listdir(lv0)
  # for lv1 in lv1s:
  #   lv2s = os.listdir('/'.join([lv0, lv1]))
  #   asf_path = '%s/%s/%s.asf' % (lv0, lv1, lv1)
  #   print('parsing %s' % asf_path)
  #   joints = parse_asf(asf_path)
  #   motions = parse_amc('./nopose.amc')
  #   joints['root'].set_motion(motions[0])
  #   joints['root'].draw()

    # for lv2 in lv2s:
    #   if lv2.split('.')[-1] != 'amc':
    #     continue
    #   amc_path = '%s/%s/%s' % (lv0, lv1, lv2)
    #   print('parsing amc %s' % amc_path)
    #   motions = parse_amc(amc_path)
    #   for idx, motion in enumerate(motions):
    #     print('setting motion %d' % idx)
    #     joints['root'].set_motion(motion)