# event-level-metric
Python implementation of Event Level Cross-Category Metric (ELC) proposed in article "[Gaze-in-wild: A dataset for studying eye and head coordination in everyday activities](https://www.nature.com/articles/s41598-020-59251-5)".

## Overview

The Event-based error metric is used for reporting and debugging label sequences from human labelers and classification algorithms. It provides a way to compare and evaluate event sequences, particularly useful in the context of eye-tracking and gaze analysis.


## Installation

To use this package, clone the repository and install the required dependencies:

```
git clone https://github.com/your-username/event-level-metric.git
cd event-level-metric
pip install numpy
```

## Usage

The main functionality is provided by the `elc` function in the `elc.py` file. Here's a basic example of how to use it:
```
import numpy as np
from elc import elc

test_r = np.hstack([[0]*30, [2]*15, [0]*45, [1]*35])
# test_t = np.copy(test_r)
test_t = np.hstack([[0] * 27, [2] * 16, [0] * 47, [2] * 35])
# set the window size for the metric
fs_win = 7  # choose winsize=3.5 for 25ms window
fp_win = 7  # chhose winsize=14 for 100 ms window
winsize = [fs_win, fp_win]

l2dis_all, olr_all, conf_mat, percent_detach = elc(test_r, test_t, winsize)
print("l2dis_all: ", l2dis_all)
print("olr_all: ", olr_all)
print("confusion matrix: ", conf_mat)
print("detached percentage: ", percent_detach)
```

Please use the bibtext below for your citation:

```
@article{kothari2020gaze,
  title={Gaze-in-wild: A dataset for studying eye and head coordination in everyday activities},
  author={Kothari, Rakshit and Yang, Zhizhuo and Kanan, Christopher and Bailey, Reynold and Pelz, Jeff B and Diaz, Gabriel J},
  journal={Scientific reports},
  volume={10},
  number={1},
  pages={2539},
  year={2020},
  publisher={Nature Publishing Group UK London}
}
```