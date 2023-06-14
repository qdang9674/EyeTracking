import numpy as np
# from scipy.signal import butter, filtfilt
from typing import List


def get_eye_feature_names() -> List[str]:
    return [
        'num_blinks',
        'pupil_size_mean',
        'pupil_size_std',
        'sac_count',
        'sac_dist_mean',
        'sac_dist_std',
        'sac_vel_mean',
        'sac_vel_std',
        'fix_count',
        'fix_duration_mean',
        'fix_duration_std',
        'fix_x_mean',
        'fix_x_std',
        'fix_y_mean',
        'fix_y_std',
        'x_pos_lrdiff_mean',
        'x_pos_lrdiff_std',
        'y_pos_lrdiff_mean',
        'y_pos_lrdiff_std',
        'pup_diam_lrdiff_mean',
        'pup_diam_lrdiff_std',
    ]


def get_eye_features(timestamps_in_sec: np.ndarray, tobii_data: np.ndarray, make_plots=False):

    # https://github.com/esdalmaijer/PyGazeAnalyser
    # version 0.1.0 (01-03-2014)
    # IF YOU DECIDE TO PUBLISH RESULTS OBTAINED WITH THIS SOFTWARE, PLEASE INCLUDE IN YOUR REFERENCES:
    # Dalmaijer, E.S., Mathôt, S., & Van der Stigchel, S. (2013). PyGaze: an
    # open-source, cross-platform toolbox for minimal-effort programming of
    # eye tracking experiments. Behaviour Research Methods.
    # doi:10.3758/s13428-013-0422-2

    TOBII_SAMP_RATE = 250

    # Does the buffer hold enough data to be confident of feature calculations?
    # Points may be concentrated in time, or spread out, or like confetti over
    # time. So if you say you need at least one second of data  to do feature
    # calculations, you're really just specifying how many non-NaN points the
    # buffer has to hold before you bother doing calculations.
    MIN_PUPIL_TIME_NEEDED = 0.2  # in seconds
    MIN_POSITION_TIME_NEEDED = 1  # in seconds

    # To distinguish blinks from saccades
    BLINK_MIN_DURATION = 0.3  # in seconds

    def saccade_detection(x, y, t, minlen=0.017, maxlen=BLINK_MIN_DURATION, vel_thresh=6, acc_thresh=340):
        """
        Detects saccades, defined as consecutive samples with an inter-sample
        velocity of over a velocity threshold or an acceleration threshold

        Inputs
            x          - numpy array of x positions
            y          - numpy array of y positions
            t          - numpy array of tracker timestamps in seconds
            minlen     - minimal length of saccades in seconds; all detected
                         saccades with len(sac) < minlen will be ignored
            maxlen     - maximal length of saccades in seconds; all detected
                         saccades with len(sac) >= maxlen will be ignored
            vel_thresh - velocity threshold in pixels/second
            acc_thresh - acceleration threshold in pixels / second**2

        Outputs
            Ssac - list of lists, each containing [starttime]
            Esac - list of lists, each containing [starttime, endtime, duration, startx, starty, endx, endy]
        """

        # CONTAINERS
        Ssac = []
        Esac = []

        # INTER-SAMPLE MEASURES
        # the distance between samples is the square root of the sum of the
        # squared horizontal and vertical interdistances
        # intdist = (np.diff(x)**2 + np.diff(y)**2)**0.5
        intdist = np.hypot(np.diff(x), np.diff(y))
        # get inter-sample times
        inttime = np.diff(t)

        # VELOCITY AND ACCELERATION
        # the velocity between samples is the inter-sample distance divided by
        # the inter-sample time
        vel = intdist / inttime
        # the acceleration is the sample-to-sample difference in velocity
        acc = np.diff(vel) / inttime[:-1]

        # SACCADE START AND END
        t0i = 0
        stop = False
        while not stop:
            # saccade start (t1) is when the velocity or acceleration
            # surpass threshold, saccade end (t2) is when both return
            # under threshold

            # detect saccade starts
            sacstarts = np.where((vel[1+t0i:] > vel_thresh) | (acc[t0i:] > acc_thresh))[0]
            if len(sacstarts) > 0:
                # timestamp for starting position
                t1i = t0i + sacstarts[0] + 1
                if t1i >= len(t)-1:
                    t1i = len(t)-2
                t1 = t[t1i]

                # add to saccade starts
                Ssac.append([t1])

                # detect saccade endings
                sacends = np.where((vel[1+t1i:] < vel_thresh) & (acc[t1i:] < acc_thresh))[0]
                if len(sacends) > 0:
                    # timestamp for ending position
                    t2i = sacends[0] + 1 + t1i + 2
                    if t2i >= len(t):
                        t2i = len(t)-1
                    t2 = t[t2i]
                    dur = t2 - t1

                    # ignore saccades that did not last long enough or lasted
                    # too long (and as such were probably blinks)
                    if minlen <= dur < maxlen:
                        # add to saccade ends
                        Esac.append([t1, t2, dur, x[t1i], y[t1i], x[t2i], y[t2i]])
                    else:
                        # remove last saccade start on too low duration
                        Ssac.pop(-1)

                    # update t0i
                    t0i = 0 + t2i
                else:
                    stop = True
            else:
                stop = True

        return Ssac, Esac

    def fixation_detection(x, y, t, max_dist_sq=.05**2, mindur=0.1):

        """
        Detects fixations, defined as consecutive samples with an inter-sample
        distance of less than a set amount of pixels

        Inputs
            x           - numpy array of x positions
            y           - numpy array of y positions
            t           - numpy array of timestamps in seconds
            max_dist_sq - maximal inter sample distance squared in pixels^2
            mindur      - minimal duration of a fixation in seconds; detected
                          fixation cadidates will be disregarded if they are below
                          this duration

        Outputs
            fixations - list of lists, each containing [starttime, endtime, duration, endx, endy]
        """

        # empty list to contain data
        t_start = None
        fixations = []

        # loop through all coordinates
        i_start = 0
        fixation_started = False
        for i in range(1, len(x)):
            # calculate Euclidean distance from the current fixation coordinate
            # to the next coordinate
            squared_distance = (x[i_start]-x[i])**2 + (y[i_start]-y[i])**2

            # check if the next coordinate is below maximal distance
            if squared_distance <= max_dist_sq and not fixation_started:
                # start a new fixation
                i_start = i
                fixation_started = True
                t_start = t[i]
            elif squared_distance > max_dist_sq and fixation_started:
                # only store the fixation if the duration is ok
                if t[i-1] - t_start >= mindur:
                    fixations.append([t_start, t[i-1], t[i-1]-t_start, x[i_start], y[i_start]])
                    t_start = None
                # end the current fixation
                fixation_started = False
                t_start = None
                i_start = i
            elif not fixation_started:
                i_start += 1
        # add last fixation end (we can lose it if squared_distance > max_dist_sq is false for the last point and the duration)
        if t_start is not None:
            fixations.append([t_start, t[len(x)-1], t[len(x)-1]-t_start, x[i_start], y[i_start]])
        return fixations

    # Tobii data is sampled at 300 Hz. I've seen unreal pupil diameter spikes
    # over 13 ms, and after some trial and error, it looks like a low-pass
    # filter at 10 Hz does a reasonable job of removing them. You may want to
    # implement that, but be careful because filtfilt won't like it when too
    # many data are missing (i.e. nans)
    # b, a = butter(5, 10, 'low', fs=300)
    # lpupd = l_pupil_diameter.copy()
    # rpupd = r_pupil_diameter.copy()
    # try:
    #     lpupd[~np.isnan(lpupd)] = filtfilt(b, a, lpupd[~np.isnan(lpupd)])
    # except:
    #     warnings.warn(f"Couldn't filter Left Pupil Diameter over timespan {timestamps_in_sec[0]:.6f} - {timestamps_in_sec[-1]:.6f}", RuntimeWarning, stacklevel=2)
    # try:
    #     rpupd[~np.isnan(rpupd)] = filtfilt(b, a, rpupd[~np.isnan(rpupd)])
    # except:
    #     warnings.warn(f"Couldn't filter Right Pupil Diameter over timespan {timestamps_in_sec[0]:.6f} - {timestamps_in_sec[-1]:.6f}", RuntimeWarning, stacklevel=2)

    l_pos_x = tobii_data[:, 0]
    l_pos_y = tobii_data[:, 1]
    r_pos_x = tobii_data[:, 2]
    r_pos_y = tobii_data[:, 3]
    l_pupil_diameter = tobii_data[:, 4]
    r_pupil_diameter = tobii_data[:, 5]

    # Average left and right diameters
    ldgood = ~np.isnan(l_pupil_diameter)
    rdgood = ~np.isnan(r_pupil_diameter)
    pupil_diameter = (l_pupil_diameter + r_pupil_diameter)/2
    pupil_diameter[ldgood & ~rdgood] = l_pupil_diameter[ldgood & ~rdgood]
    pupil_diameter[~ldgood & rdgood] = r_pupil_diameter[~ldgood & rdgood]

    if (~np.isnan(pupil_diameter)).sum() > MIN_PUPIL_TIME_NEEDED*TOBII_SAMP_RATE:
        pupil_size_mean = np.nanmean(pupil_diameter)
        pupil_size_std = np.nanstd(pupil_diameter)
    else:
        pupil_size_mean = np.nan
        pupil_size_std = np.nan

    # Average left and right positions
    lxgood = ~np.isnan(l_pos_x)
    rxgood = ~np.isnan(r_pos_x)
    x_pos = (l_pos_x + r_pos_x)/2
    x_pos[lxgood & ~rxgood] = l_pos_x[lxgood & ~rxgood]
    x_pos[~lxgood & rxgood] = r_pos_x[~lxgood & rxgood]

    lygood = ~np.isnan(l_pos_y)
    rygood = ~np.isnan(r_pos_y)
    y_pos = (l_pos_y + r_pos_y)/2
    y_pos[lygood & ~rygood] = l_pos_y[lygood & ~rygood]
    y_pos[~lygood & rygood] = r_pos_y[~lygood & rygood]

    # Capture possible stabismus when person brain hard at work, haha
    xgood = lxgood & rxgood
    if xgood.sum() > MIN_POSITION_TIME_NEEDED * TOBII_SAMP_RATE:
        x_pos_lrdiff = l_pos_x[xgood] - r_pos_x[xgood]
        x_pos_lrdiff_mean = np.nanmean(x_pos_lrdiff)
        x_pos_lrdiff_std = np.nanstd(x_pos_lrdiff)
    else:
        x_pos_lrdiff_mean = np.nan
        x_pos_lrdiff_std = np.nan

    ygood = lygood & rygood
    if ygood.sum() > MIN_POSITION_TIME_NEEDED * TOBII_SAMP_RATE:
        y_pos_lrdiff = l_pos_y[ygood] - r_pos_y[ygood]
        y_pos_lrdiff_mean = np.nanmean(y_pos_lrdiff)
        y_pos_lrdiff_std = np.nanstd(y_pos_lrdiff)
    else:
        y_pos_lrdiff_mean = np.nan
        y_pos_lrdiff_std = np.nan

    dgood = ldgood & rdgood
    if dgood.sum() > MIN_PUPIL_TIME_NEEDED * TOBII_SAMP_RATE:
        pup_diam_lrdiff = l_pupil_diameter[dgood] - r_pupil_diameter[dgood]
        pup_diam_lrdiff_mean = np.nanmean(pup_diam_lrdiff)
        pup_diam_lrdiff_std = np.nanstd(pup_diam_lrdiff)
    else:
        pup_diam_lrdiff_mean = np.nan
        pup_diam_lrdiff_std = np.nan

    # Keep x and y data only where both exist and if you have enough, then do
    # blink, saccade, and fixation detection
    pos_good = ~np.isnan(x_pos) & ~np.isnan(y_pos)
    if pos_good.sum() > MIN_POSITION_TIME_NEEDED*TOBII_SAMP_RATE:
        x_pos = x_pos[pos_good]
        y_pos = y_pos[pos_good]
        timestamps = timestamps_in_sec[pos_good]

        # Calculate number of blinks
        dt = np.diff(timestamps)
        num_blinks = np.where(dt >= BLINK_MIN_DURATION)[0].size

        # Detect saccades
        s_saccade, e_saccade = saccade_detection(x_pos, y_pos, timestamps)
        sac_dist = np.array([np.hypot(sac[4]-sac[3], sac[6]-sac[5]) for sac in e_saccade])
        sac_vel = np.array([sd/(sac[1]-sac[0]) for sd, sac in zip(sac_dist, e_saccade)])

        sac_count = len(sac_dist)
        sac_dist_mean = np.nanmean(sac_dist) if sac_count > 0 else np.nan
        sac_dist_std = np.nanstd(sac_dist) if sac_count > 0 else np.nan
        sac_vel_mean = np.nanmean(sac_vel) if sac_count > 0 else np.nan
        sac_vel_std = np.nanstd(sac_vel) if sac_count > 0 else np.nan

        # Detect fixations
        fixations = fixation_detection(x_pos, y_pos, timestamps)
        fix_dur = np.array([fix[2] for fix in fixations])
        fix_x = np.array([fix[3] for fix in fixations])
        fix_y = np.array([fix[4] for fix in fixations])

        fix_count = len(fix_dur)
        fix_duration_mean = np.nanmean(fix_dur) if fix_count > 0 else np.nan
        fix_duration_std = np.nanstd(fix_dur) if fix_count > 0 else np.nan
        fix_x_mean = np.nanmean(fix_x) if fix_count > 0 else np.nan
        fix_x_std = np.nanstd(fix_x) if fix_count > 0 else np.nan
        fix_y_mean = np.nanmean(fix_y) if fix_count > 0 else np.nan
        fix_y_std = np.nanstd(fix_y) if fix_count > 0 else np.nan
    else:
        num_blinks = np.nan
        sac_count = np.nan
        sac_dist_mean = np.nan
        sac_dist_std = np.nan
        sac_vel_mean = np.nan
        sac_vel_std = np.nan
        fix_count = np.nan
        fix_duration_mean = np.nan
        fix_duration_std = np.nan
        fix_x_mean = np.nan
        fix_x_std = np.nan
        fix_y_mean = np.nan
        fix_y_std = np.nan

    if make_plots:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(14, 5))
        plt.plot(timestamps, x_pos, label='x')
        plt.plot(timestamps, y_pos, label='y')
        for f in fixations:
            _ = plt.fill([f[0], f[1], f[1], f[0]], [min(f[3:])]*2 + [max(f[3:])]*2, 'r', alpha=0.6, edgecolor='k', linewidth=1)
        sac_tstarts = [sac[0] for sac in e_saccade]
        sac_tends = [sac[1] for sac in e_saccade]
        for ststart, stend in zip(sac_tstarts, sac_tends):
            _ = plt.axvspan(ststart, stend, color='g', alpha=0.5)
        plt.legend()
        plt.ylim(-0.3, 1.3)
        plt.tight_layout()
        plt.show()

    return np.array([
        num_blinks,
        pupil_size_mean, # focus on this
        pupil_size_std, # focus on this 
        sac_count,
        sac_dist_mean,
        sac_dist_std,
        sac_vel_mean,
        sac_vel_std,
        fix_count,
        fix_duration_mean,
        fix_duration_std,
        fix_x_mean,
        fix_x_std,
        fix_y_mean,
        fix_y_std,
        x_pos_lrdiff_mean,
        x_pos_lrdiff_std,
        y_pos_lrdiff_mean,
        y_pos_lrdiff_std,
        pup_diam_lrdiff_mean,
        pup_diam_lrdiff_std,
    ])
