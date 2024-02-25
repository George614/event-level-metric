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