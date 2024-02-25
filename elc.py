import numpy as np
import math


def to_event(labels):
    """
    convert sequence of labels to event representation event structure contains event type, start position, end position
    :param labels: sequence of labels per sample
    :return: event structure
    """
    events = []
    stype = labels[0]
    start = 0
    for i in range(1, len(labels)):
        if stype == labels[i]:
            if i == len(labels) - 1:
                end = i
                events.append(np.array([stype, start, end]))
            continue
        else:
            end = i - 1
            events.append(np.array([stype, start, end]))
            start = i
            stype = labels[i]
    events = np.vstack(events)
    return events


def overlap_ratio(l1, l2, u1, u2):
    """
    calculate the overlap ratio from a pair of matched measures
    :param l1: start of event 1
    :param l2: start of event 2
    :param u1: end of event 1
    :param u2: end of event 2
    :return: overlap ratio of two events
    """
    lowerbound = min(l1, l2)
    upperbound = max(u1, u2)
    overlapsrt = max(l1, l2)
    overlapend = min(u1, u2)
    return (overlapend - overlapsrt + 1) / (upperbound - lowerbound + 1)


def l2_dis(m1, m2):
    """
    calculate L2 distance from a pair of adjacent measures that at least one of them is matched
    distance can be derived directly from offsets
    :param m1: offset at the start
    :param m2: offset at the end
    :return: L2 distance
    """
    return math.sqrt(m1**2 + m2**2)


def fill_unlabeled(seq1, seq2):
    """
    fill unlabeled region
    :param seq1: reference sequence
    :param seq2: testing sequence
    """
    for idx, sample in enumerate(seq1):
        if sample == -1 and seq2[idx] != -1:
            seq1[idx] = seq2[idx]
    for idx, sample in enumerate(seq2):
        if sample == -1 and seq1[idx] != -1:
            seq2[idx] = seq1[idx]


def matching(events_r, events_t, winsize):
    """
    match two event sequences using a window-based method
    :param events_r: reference event sequence
    :param events_t: testing event sequence
    :param winsize: window size for saccade / non-saccade related transition points
    :return: a structure containing matching information
    """
    N = len(events_r)
    found = np.zeros((N, 2))
    offsets = np.zeros((N, 2))
    winsize_s = winsize[0]
    winsize_ns = winsize[1]
    tranType_r = np.ones((N, 2), dtype=np.int) * -1
    tranType_t = np.ones((N, 4), dtype=np.int) * -1
    tranType_r[:, 1] = events_r[:, 0]

    for i in range(N):
        tranType_r[i, 0] = events_r[max(i - 1, 0), 0]
        if events_r[i, 0] == 3 or events_r[max(i - 1, 0), 0] == 3:
            winsize = winsize_s
        else:
            winsize = winsize_ns
        # window boundaries for start and end points
        lower_s = max(events_r[i, 1] - winsize // 2, 0)
        upper_s = min(
            events_r[i, 1] + winsize // 2 - (winsize + 1) % 2, events_r[-1, 2]
        )
        lower_e = max(events_r[i, 2] - winsize // 2, 0)
        upper_e = min(
            events_r[i, 2] + winsize // 2 - (winsize + 1) % 2, events_r[-1, 2]
        )
        # find matching point for the start of the current event
        for j in range(len(events_t)):
            if (
                lower_s <= events_t[j, 1] <= upper_s
                and events_r[i, 0] == events_t[j, 0]
            ):
                # return the first matching transition point for the start found in the window
                found[i, 0] = 1
                offsets[i, 0] = events_t[j, 1] - events_r[i, 1]
                tranType_t[i, 0] = events_t[max(j - 1, 0), 0]
                tranType_t[i, 1] = events_t[j, 0]
                break
        # find matching point for the end of the current event
        for k in range(len(events_t)):
            if (
                lower_e <= events_t[k, 2] <= upper_e
                and events_r[i, 0] == events_t[k, 0]
            ):
                found[i, 1] = 1
                offsets[i, 1] = events_t[k, 2] - events_r[i, 2]
                tranType_t[i, 2] = events_t[k, 0]
                tranType_t[i, 3] = events_t[min(k + 1, len(events_t) - 1), 0]
                break

    matched = found[:, 0] * found[:, 1]
    matched = np.expand_dims(matched, axis=-1)
    # measures: 0-1 start and end idx of reference events, 2 matching status of each event in reference,
    # 3-4 found for start and end point, 5-6 offset of start and end, 7-8 transition type of each event
    # start point of reference, 9-12 transition type of each event start and end point of testing.
    measures = np.concatenate(
        [events_r[:, 1:], matched, found, offsets, tranType_r, tranType_t], axis=1
    )
    return np.asarray(measures, dtype=np.int)


def offset_correction(measures, seq_r, seq_t):
    """
    correct time offset where the transition points are matched
    :param measures: measure structure containing matching information
    :param seq_r_done: sequence of labels for reference
    :param seq_t_done: sequence of labels for testing
    :return: the corrected label sequences for reference and testing
    """
    N = len(seq_r)
    seq_r_done = np.copy(seq_r)
    seq_t_done = np.copy(seq_t)
    shifted = np.zeros((N,), dtype=np.bool)
    for i in range(len(measures)):
        # make correction at start point
        start_idx = measures[i, 0]
        end_idx = measures[i, 1]
        if measures[i, 3] and not shifted[max(0, start_idx - 1)]:
            offset = measures[i, 5]
            half = offset // 2
            rmd = offset % 2
            if offset > 0:  # ref leading
                seq_r_done[start_idx : start_idx + half] = measures[i, 7]
                seq_t_done[
                    max(start_idx + offset - half - rmd, 0) : start_idx + offset + 1
                ] = measures[i, 10]
            elif offset < 0:  # ref lagging
                seq_r_done[start_idx + half : start_idx + 1] = measures[i, 8]
                seq_t_done[start_idx + offset : start_idx + offset - half - rmd] = (
                    measures[i, 9]
                )
            shifted[start_idx] = True
        # make correction at end point
        if measures[i, 4]:
            offset = measures[i, 6]
            half = offset // 2
            rmd = offset % 2
            if offset > 0:  # ref leading
                if offset == 1:
                    seq_t_done[end_idx + 1] = measures[i, 12]
                else:
                    seq_r_done[end_idx : end_idx + half] = measures[i, 8]
                    seq_t_done[
                        max(0, end_idx + offset - half - rmd) : end_idx + offset + 1
                    ] = measures[i, 12]
            elif offset < 0:  # ref lagging
                seq_r_done[max(0, end_idx + half) : end_idx + 1] = measures[
                    min(i + 1, len(measures) - 1), 8
                ]
                seq_t_done[max(0, end_idx + offset) : end_idx + offset - half - rmd] = (
                    measures[i, 11]
                )
            shifted[end_idx] = True
    return seq_r_done, seq_t_done


def process_matched(measures, seq_r, seq_t):
    """
    calculate L2 distance and overlap ratio from matched events
    :param measures: measure structure containing matching information
    :param seq_r: sequence of labels for reference
    :param seq_t: sequence of labels for testing
    :return: score structure containing eventIndex, eventType, L2 distance, Overlap Ratio. Number of correctly
    classified events. Percentage of detached events.
    """
    scores = []
    # number of correctly classified events in each class
    num_fix = 0
    num_pur = 0
    num_sac = 0
    # counter for unmatched events which the labels are same during the reference segment
    fix_detach = 0
    pur_detach = 0
    sac_detach = 0
    for i in range(len(measures)):
        if measures[i, 2]:
            evt = measures[i, 8]
            if evt == 0:
                num_fix += 1
            elif evt == 1:
                num_pur += 1
            elif evt == 2:
                num_sac += 1
            else:
                raise Exception(
                    "event type should be within [0,1,2]. The evt was: {}".format(evt)
                )
            # Timing offsets (L2 distance) and Overlap Ratio calculation
            l2dis = l2_dis(measures[i, 5], measures[i, 6])
            lower_ref = measures[i, 0]
            lower_test = lower_ref + measures[i, 5]
            upper_ref = measures[i, 1]
            upper_test = upper_ref + measures[i, 6]
            olr = overlap_ratio(lower_ref, lower_test, upper_ref, upper_test)
            scores.append(np.array([i, evt, l2dis, olr]))
        else:
            s_idx = int(measures[i, 0])
            e_idx = int(measures[i, 1])
            all_equal = seq_r[s_idx:e_idx] == seq_t[s_idx:e_idx]
            if np.all(all_equal):
                if seq_r[s_idx] == 0:
                    fix_detach += 1
                elif seq_r[s_idx] == 1:
                    pur_detach += 1
                elif seq_r[s_idx] == 2:
                    sac_detach += 1
                else:
                    raise Exception(
                        "event type should be within [0,1,2]. The evt was: {}".format(
                            seq_r[s_idx]
                        )
                    )

    scores = np.vstack(scores)
    num_cor = [num_fix, num_pur, num_sac]  # number of correctly classified events
    num_detach = [fix_detach, pur_detach, sac_detach]
    events_type = measures[:, 8]
    numF = sum(events_type == 0)
    numP = sum(events_type == 1)
    numS = sum(events_type == 2)
    percent_detach = [num_detach[0] / numF, num_detach[1] / numP, num_detach[2] / numS]

    return scores, num_cor, percent_detach


def change_label(label_seq):
    """
    change the label to facilitate sample wise comparison
    :param label_seq: original label sequence
    :return: label sequence with changed labels
    """
    changed = np.copy(label_seq)
    changed[label_seq == 0] = 1
    changed[label_seq == 1] = 10
    changed[label_seq == 2] = 100
    return changed


def cal_conf_mat(seq_r, seq_t, num_cor):
    """
    calculate the numbers in the confusion matrix
    :param seq_r: label sequence after offset correction for reference
    :param seq_t: label sequence after offset correction for testing
    :param num_cor: numbers of correctly classified events
    :return: confusion matrix
    """
    seq_ref = change_label(seq_r)
    seq_test = change_label(seq_t)
    diff_seq = seq_ref - seq_test
    diff_events = to_event(diff_seq)
    FP = sum(diff_events[:, 0] == -9)
    FS = sum(diff_events[:, 0] == -99)
    PF = sum(diff_events[:, 0] == 9)
    PS = sum(diff_events[:, 0] == -90)
    SF = sum(diff_events[:, 0] == 99)
    SP = sum(diff_events[:, 0] == 90)
    conf_mat = np.array(
        [[num_cor[0], FP, FS], [PF, num_cor[1], PS], [SF, SP, num_cor[2]]], dtype=np.int
    )
    return conf_mat


def elc(seq_r, seq_t, winsize, ignore_idx=-1):
    """
    ELC metric main function. Calculate L2 distance and overlap ratio for matched events, confusion matrix and
    percentage of detached events.
    :param seq_r: label sequence for reference
    :param seq_t: label sequence for testing
    :param winsize: window size for saccade / non-saccade related transition points
    :param ignore_idx: index value to be ignored
    :return: mean and std of l2 distance and overlap ratio for each individual class, confusion matrix and
             percentage of detached events
    """
    keep_idx = np.logical_and(seq_r != ignore_idx, seq_t != ignore_idx)
    seq_r = seq_r[keep_idx]
    seq_t = seq_t[keep_idx]
    events_r = to_event(seq_r)
    events_t = to_event(seq_t)
    measures = matching(events_r, events_t, winsize)
    scores, num_cor, percent_detach = process_matched(measures, seq_r, seq_t)
    evt = scores[:, 1]
    l2dis = scores[:, 2]
    olr = scores[:, 3]
    l2dis_f_mean = np.mean(l2dis[evt == 0])
    l2dis_f_std = np.std(l2dis[evt == 0])
    l2dis_p_mean = np.mean(l2dis[evt == 1]) if sum(l2dis[evt == 1]) > 0 else None
    l2dis_p_std = np.std(l2dis[evt == 1]) if sum(l2dis[evt == 1]) > 0 else None
    l2dis_s_mean = np.mean(l2dis[evt == 2])
    l2dis_s_std = np.std(l2dis[evt == 2])
    olr_f_mean = np.mean(olr[evt == 0])
    olr_f_std = np.std(olr[evt == 0])
    olr_p_mean = np.mean(olr[evt == 1]) if sum(olr[evt == 1]) > 0 else None
    olr_p_std = np.std(olr[evt == 1]) if sum(olr[evt == 1]) > 0 else None
    olr_s_mean = np.mean(olr[evt == 2])
    olr_s_std = np.std(olr[evt == 2])
    l2dis_all = [
        l2dis_f_mean,
        l2dis_f_std,
        l2dis_p_mean,
        l2dis_p_std,
        l2dis_s_mean,
        l2dis_s_std,
    ]
    olr_all = [olr_f_mean, olr_f_std, olr_p_mean, olr_p_std, olr_s_mean, olr_s_std]
    # fill_unlabeled(seq_r, seq_t)
    seq_r_done, seq_t_done = offset_correction(measures, seq_r, seq_t)
    conf_mat = cal_conf_mat(seq_r_done, seq_t_done, num_cor)
    return l2dis_all, olr_all, conf_mat, percent_detach
