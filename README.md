# AMCParser

Forked from [CalciferZh](https://github.com/CalciferZh/AMCParser)

## Whats different?

This repo contains scripts to pre-process the (well knonwn) ASF/AMC MoCap data into pandas dataframes that can be easily used for machine learning. Since I do not have much time a.t.m. I leave this code as is for the near future.

## Usage

`./3Dviewer.py <asf_file> <amc_file>`

`./amc_parser.py <asf_file> <amc_file>`

## Parser

The asf/amc parsers are straightforward and easy to understand. The parsers are fully tested on the CMU MoCap dataset, but I don't expect it can work on other datasets without any modification. However, it won't be hard to extend it for more complicating asf/amc files.

## Visualization

Matplotlib is used to draw joints and bones in 3D statically; PyGame and PyOpenGL are used to draw motion sequence.

In 3DViewer, we support:

* `WASD` to move around.
* `QE` to zoom in/out.
* `↑ ↓ ← →` to rotate.
* `LEFT MOUSE BUTTON` to drag.
* `RETURN` to reset camera view.
* `SPACE` to start/pause.
* `,` and `.` to rewind and forward.
* `ESC` to quit

## Dependencies

* numpy
* transforms3d
* matplotlib
* pygame
* pyopengl

All the dependencies are available via `pip install`.
