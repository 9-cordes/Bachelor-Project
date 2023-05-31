import numpy as np
import matplotlib.pyplot as plt
import util
from matplotlib.patches import Patch
from matplotlib.patches import Rectangle


def get_missing_durations(durations, stimulus_presentations):
    for i in range(len(durations)):
        if np.isnan(durations[i]):
            durations[i] = stimulus_presentations.iloc[i].stop_time - stimulus_presentations.iloc[i].start_time
    return durations


def create_unit_activity_vectors(units, spike_times, stimulus_presentations) -> np.array:
    """
    creates the general population Response Matrix with all unit activity vectors, stimulus dependent
    """

    custom_bins = stimulus_presentations[['start_time', 'stop_time']].to_numpy().flatten()

    durations = stimulus_presentations[['duration']].to_numpy().flatten()

    durations = get_missing_durations(np.copy(durations), stimulus_presentations)

    # construct Matrix:
    M = np.zeros((len(units), len(durations)))

    for i, unit_id in enumerate(units.index):
        unit_spike_times = spike_times[unit_id] # get spike times for current unit
        # count spike_times for each bin and intervals in between
        unit_hist, bin_edges = np.histogram(unit_spike_times, bins=custom_bins)
        unit_bin_counts = unit_hist[::2] # take every 2nd Element so that we only have the desired bins

        mean_unit_bin_counts = unit_bin_counts / durations

        assert M[i, :].shape == mean_unit_bin_counts.shape

        M[i, :] = mean_unit_bin_counts

    return M


def plot_unit_activity_vectors(M):
    plt.figure(figsize=(15, 1), dpi=180)
    plt.xticks(np.arange(200), fontsize=2)
    plt.yticks(np.arange(44, step=5), fontsize=4)
    plt.imshow(M[:, :200], aspect='auto')


def construct_unit_activity_population_response_matrix(units, spike_times, stimulus_presentations):

    durations = stimulus_presentations[['duration']].to_numpy().flatten()
    durations_complete = get_missing_durations(np.copy(durations), stimulus_presentations)

    M = create_unit_activity_vectors(units, spike_times, stimulus_presentations)
    # plot_unit_activity_vectors(M)

    # comment in the following for the overall Correlation Matrix (all responses mixed, not stimulus specific)
    # C = np.corrcoef(np.transpose(M))

    # fig, axs = plt.subplots(1, 3, figsize=(20,8))
    # fig.tight_layout()
#
    # im0 = axs[0].imshow(C)
    # im1 = axs[1].imshow(C[-5000:, :5000])
    # im2 = axs[2].imshow(C[:200, :200])
#
    # fig.suptitle(t='Representational Similarity (Corr) of all Session Stimuli', fontsize=22)

    return M


# stimulus presentations plot:
def plot_stimulus_presentations_with_activities(stimulus_presentations, size=(180, 2), block=0, activities=None, title=None):

    if activities is not None:
        fig, (ax1, ax0) = plt.subplots(2, figsize=size, sharex=True, gridspec_kw={'height_ratios': [5, 1]})
    else:
        fig, ax0 = plt.subplots(figsize=size)

    if title is not None:
        fig.suptitle(title, fontsize=int(0.5 * size[0]))

    timings = []
    for i in range(len(stimulus_presentations)):
        stimulus = stimulus_presentations.iloc[i]

        color = util.image_to_color(stimulus.image_name)
        ax0.add_patch(Rectangle((stimulus.start_time, 0), stimulus.duration, 0.1, facecolor=color, alpha=0.8))
        timings.append((stimulus.start_time, stimulus.stop_time))

        if activities is not None:
            im = ax1.imshow(np.flip(activities[:, [i]]), extent=(stimulus.start_time-0.22,
                                                                 stimulus.stop_time+0.22,
                                                                 0,
                                                                 activities.shape[0]
                                                                 ), aspect='auto')

    ax0.set_yticks([0, 1])
    ax0.set_xticks(np.arange(25.0, 800.0, step=1))
    ax0.spines[["left", "top", "right"]].set_visible(False)
    ax0.grid(True)
    ax0.margins(x=0, y=0)

    plt.tight_layout()
    plt.xticks(fontsize=11)
    # ax.legend()

    if activities is not None:
        ax1.set_xlim(timings[0][0], timings[-1][1])
        # ax1.set_yticks(np.arange(activities.shape[0], step=5))
        ax1.invert_yaxis()
        ### TODO:
        # values on y axis are wrong direction!

        # ax1.set_xticks(np.arange(len(stimulus_presentations)))


def get_unit_activity_vectors_for_image(image_name, M, stimulus_presentations):
    image_stimuli = stimulus_presentations[stimulus_presentations.image_name == image_name]
    image_stimuli_block0 = image_stimuli[image_stimuli.stimulus_block == 0]
    image_stimuli_block5 = image_stimuli[image_stimuli.stimulus_block == 5]

    A_image = M[:, image_stimuli.index]
    A_image_b0 = M[:, image_stimuli_block0.index]
    A_image_b5 = M[:, image_stimuli_block5.index]

    assert A_image.shape[1] == A_image_b0.shape[1] + A_image_b5.shape[1]

    return A_image, A_image_b0, A_image_b5


def plot_blockwise_unit_activities_for_image(image_name, M, stimulus_presentations, session):
    fig, axs = plt.subplots(5, 1, figsize=(60, 36))

    image = session.stimulus_templates.loc[image_name].warped

    complete, b0, b5 = get_unit_activity_vectors_for_image(image_name, M, stimulus_presentations)

    C = np.corrcoef(np.transpose(complete))

    axs[0].set_title(image_name, fontsize=46)
    im0 = axs[0].imshow(image)


    axs[1].set_title('all ocurrences', fontsize=46)
    im1 = axs[1].imshow(complete)

    axs[2].set_title('CORRCOEF', fontsize=46)
    im2 = axs[2].imshow(C)

    axs[3].set_title('in active block 0', fontsize=46)
    im3 = axs[3].imshow(b0)
    axs[4].set_title('in passive block 5', fontsize=46)
    im4 = axs[4].imshow(b5)


def check_reduction_over_stimuli(M_reduced, M_rd_active, M_rd_passive):

    # to check if it makes sense:
    fig, axs = plt.subplots(1, 3, figsize=(20,8))
    fig.suptitle('averaged population response', fontsize=26)

    im0 = axs[0].imshow(M_reduced)
    axs[0].set_title('full')
    axs[0].set_xlabel('stimulus presentations')
    axs[0].set_ylabel('VIsp Units')

    im1 = axs[1].imshow(M_rd_active)
    axs[1].set_title('active')
    axs[1].set_xlabel('stimulus presentations')
    axs[1].set_ylabel('VIsp Units')

    im2 = axs[2].imshow(M_rd_passive)
    axs[2].set_title('passive')
    axs[2].set_xlabel('stimulus presentations')
    axs[2].set_ylabel('VIsp Units')


def corrcoef_matrices_visualization(C_reduced, C_rd_active, C_rd_passive, half):

    fig, axs = plt.subplots(2, 3, figsize=(20,12))
    fig.title = 'averaged population response'

    im0 = axs[0, 0].imshow(C_reduced, origin='lower')
    axs[0, 0].set_title('full')
    axs[0, 0].set_xlabel('stimulus presentations')

    im1 = axs[0, 1].imshow(C_rd_active, origin='lower')
    axs[0, 1].set_title('active')
    axs[0, 1].set_xlabel('stimulus presentations')

    im2 = axs[0, 2].imshow(C_rd_passive, origin='lower')
    axs[0, 2].set_title('passive')
    axs[0, 2].set_xlabel('stimulus presentations')

    im3 = axs[1, 0].imshow(C_reduced, origin='lower')
    axs[1, 0].set_title('full')
    axs[1, 0].set_xlabel('stimulus presentations')

    im4 = axs[1, 1].imshow(C_reduced[half:, :half], origin='lower')
    axs[1, 1].set_title('??')
    axs[1, 1].set_xlabel('stimulus presentations')

    im5 = axs[1, 2].imshow(C_reduced[:half, half:], origin='lower')
    axs[1, 2].set_title('??')
    axs[1, 2].set_xlabel('stimulus presentations')

    #fig.colorbar(im0, shrink=0.1)


def plot_representational_similarity_over_time(timings, C_reduced, C_rd_active, across_blocks2, half):
    fig, axs = plt.subplots(1, 2, figsize=(20,6), gridspec_kw={'width_ratios': [1, 2]})

    axs[0].scatter(timings, C_reduced[0,:])
    axs[1].plot(timings, C_reduced[0,:], c='purple', alpha=0.2)
    axs[1].plot(timings[:half], C_rd_active[0,:], alpha=0.8, label='active block')
    axs[1].plot(timings[half:], across_blocks2[0,:], alpha=0.8, label='passive block')
    axs[1].set_xlabel('time in seconds')
    axs[1].set_ylabel('Representational Similarity')
    axs[1].legend()


def count_same_image_appearences(image_name, stimulus_presentations):
    """
    counts lengths of same stimulus (image) chains throughout stimulus presentations, while tracking the time medians
    of these chains
    """

    current_image = 'no'
    counts = []
    timings = []
    timings_start_end = []
    count = 0
    for i in range(len(stimulus_presentations)):
        stimulus = stimulus_presentations.iloc[i]
        if stimulus.image_name != current_image:     # if we have a new image
            if not (stimulus.image_name == 'omitted'):   # to skip the omitted pictures and count on
                if stimulus.image_name == image_name:        # if its the one we look for
                    count = 1
                    # indices.append(i)
                    time_begin = stimulus.start_time        # keep track of timings

                elif current_image == image_name:            # if its the last in row of the one we look for (previous-image)
                    counts.append(count)
                    timings.append(np.median([time_begin, current_time_end]))
                    timings_start_end.append([time_begin, current_time_end])

                current_image = stimulus.image_name
        else:                                        # no image change (previous is the same as current one)
            if stimulus.image_name == image_name:        # if its the one we look for
                # we are counting
                count += 1
                # indices.append(i)
                current_time_end = stimulus.stop_time    # keep track of timings

    if current_image == image_name:
        counts.append(count)
        timings.append(np.median([time_begin, current_time_end]))
        timings_start_end.append([time_begin, current_time_end])

    # indices
    return counts, timings, timings_start_end


def average_population_response_over_same_stimulus_presentations(M, image_name, stimulus_presentations):
    """
    calculating averages for same stimulus presentations, which means the repeated display of the same image (specified through image_name),
    seperated through the presentation of a different stimulus
    returning:
    - a reduced population response Matrix M
    - also split up in the passive and active block
    - as well as the time medians

    """
    # take population activity Matrix for fixed image
    M_image, M_image_b0, M_image_b5 = get_unit_activity_vectors_for_image(image_name, M, stimulus_presentations)

    # count in stimulus_presentations table how long each row of image presentation is,
    # also get median timings for each chain of images
    counts, timings, timings_start_end = count_same_image_appearences(image_name, stimulus_presentations)

    assert sum(counts) == M_image.shape[1]
    assert len(counts) == len(timings)

    b = int(len(counts)/2)

    # construct Matrix that averages presentations of same picture in a row
    M_avg = np.zeros((M_image.shape[0], len(counts)))
    M_avg_active = np.zeros((M_image.shape[0], b))
    M_avg_passive = np.zeros((M_image.shape[0], b))

    prev = 0
    for i, cnt in enumerate(counts):
        M_avg[:, i] = np.mean(M_image[:, prev: prev + cnt], axis=1)

        if i < b:
            # might not be necessary at this point
            M_avg_active[:, i] = np.mean(M_image_b0[:, prev:prev + cnt], axis=1)
            M_avg_passive[:, i] = np.mean(M_image_b5[:, prev:prev + cnt], axis=1)

        prev += cnt

    return M_avg, M_avg_active, M_avg_passive, timings, timings_start_end


def average_running_speed_all_stimuli(stimulus_presentations, running_speed, custom_timings=None):

    startTimes = stimulus_presentations['start_time'].to_numpy()
    endTimes = stimulus_presentations['stop_time'].to_numpy()
    if custom_timings is not None:
        startTimes = custom_timings[:, 0]
        endTimes = custom_timings[:, 1]

    running_timestamps = running_speed['timestamps'].to_numpy()
    speeds = running_speed['speed'].to_numpy()

    mean_running_speeds = []
    std_running_speeds = []

    for i, start in enumerate(startTimes):
        startInd = np.searchsorted(running_timestamps, start)
        endInd = np.searchsorted(running_timestamps, endTimes[i])

        mean_running_speeds.append(np.mean(speeds[startInd:endInd]))
        std_running_speeds.append(np.std(speeds[startInd:endInd]))

    # assert len(mean_running_speeds) == len(stimulus_presentations)
    return mean_running_speeds, std_running_speeds


def average_pupil_area_all_stimuli(stimulus_presentations, pupil_area, custom_timings=None):

    startTimes = stimulus_presentations['start_time'].to_numpy()
    endTimes = stimulus_presentations['stop_time'].to_numpy()
    if custom_timings.any():
        startTimes = custom_timings[:,0]
        endTimes = custom_timings[:,1]

    pupil_timestamps = pupil_area['timestamps'].to_numpy()
    area = pupil_area['pupil_area'].to_numpy()

    mean_pupil_areas = []
    std_pupil_areas = []

    for i, start in enumerate(startTimes):
        startInd = np.searchsorted(pupil_timestamps, start)
        endInd = np.searchsorted(pupil_timestamps, endTimes[i])

        mean_pupil_areas.append(np.mean(area[startInd:endInd]))
        std_pupil_areas.append(np.std(area[startInd:endInd]))

    # assert len(mean_running_speeds) == len(stimulus_presentations)
    return mean_pupil_areas, std_pupil_areas


def extract_speed_and_pupil_data(sp, timings_start_end, running_speed, eye_tracking):

    # sp = stimulus_presentations[stimulus_presentations.image_name == 'im111_r']
    timings_start_end = np.array(timings_start_end)

    # running speed:
    avg_running_speed, std_running_speed = average_running_speed_all_stimuli(sp, running_speed, timings_start_end)

    # pupil area:
    eye_tracking_noblinks = eye_tracking[~eye_tracking['likely_blink']]
    eye_track = eye_tracking_noblinks[['timestamps', 'pupil_area']]
    avg_pupil_area, std_pupil_area = average_pupil_area_all_stimuli(sp, eye_track, timings_start_end)

    return avg_running_speed, std_running_speed, avg_pupil_area, std_pupil_area


def running_speeds_and_pupil_areas(stimulus_presentations, running_speed, eye_tracking):
    """
    gets the unaveraged running speeds and pupil areas for given stimuli_presentations
    :param stimulus_presentations: should be fixed on an image here
    :param running_speed:
    :param pupil_area:
    :return: running and pupil timestamps for specific image stimuli
    """

    startTimes = stimulus_presentations['start_time'].to_numpy()
    endTimes = stimulus_presentations['stop_time'].to_numpy()

    running_timestamps = running_speed['timestamps'].to_numpy()
    speeds = running_speed['speed'].to_numpy()

    eye_tracking_noblinks = eye_tracking[~eye_tracking['likely_blink']]
    pupil_area = eye_tracking_noblinks[['timestamps', 'pupil_area']]
    pupil_timestamps = pupil_area['timestamps'].to_numpy()
    areas = pupil_area['pupil_area'].to_numpy()

    running_speeds = []
    pupil_areas = []

    for i, start in enumerate(startTimes):
        startInd_rs = np.searchsorted(running_timestamps, start)
        endInd_rs = np.searchsorted(running_timestamps, endTimes[i])

        startInd_pa = np.searchsorted(pupil_timestamps, start)
        endInd_pa = np.searchsorted(pupil_timestamps, endTimes[i])

        assert startInd_rs < endInd_rs
        assert startInd_pa <= endInd_pa

        if startInd_pa == endInd_pa:
            pupil_areas.append(areas[startInd_pa])
        else:
            pupil_areas.append(np.mean(areas[startInd_pa:endInd_pa]))

        running_speeds.append(np.mean(speeds[startInd_rs:endInd_rs]))

    # assert len(mean_running_speeds) == len(stimulus_presentations)
    return running_speeds, pupil_areas


def plot_single_behaviour(avg, std, half, y_title):
    fig, ax = plt.subplots(figsize=(15, 5))

    # ax.set_title(label='pupil')
    ax.plot(avg, label=y_title)
    error = np.array(std)
    ax.fill_between(np.arange(len(avg)), np.array(avg)-error, np.array(avg)+error, alpha=0.2)
    ax.axvline(half, c='r')

    ax.set_xlabel('Stimulus Presentations')
    ax.set_ylabel(y_title)
    ax.legend()


def plot_active_vs_passive_RS_running_speed_and_pupil_area_over_time(timings,
                                                                     C_rd_active,
                                                                     C_rd_passive,
                                                                     across_blocks2,
                                                                     hit_change_times,
                                                                     miss_change_times,
                                                                     avg_running_speed,
                                                                     std_running_speed,
                                                                     avg_pupil_area,
                                                                     std_pupil_area,
                                                                     half):

    fig, axs = plt.subplots(3, 2, figsize=(16,9), gridspec_kw={'width_ratios': [1, 2], 'height_ratios': [2,1,1]})
    fig.tight_layout()

    x = timings

    # plot representational similarity

    axs[0,0].plot(x[:half], C_rd_active[0,:], label='active block')           #[0, :] takes first row of matrix
    axs[0,0].plot(x[:half], C_rd_passive[0,:], label='passive block')
    axs[0,0].vlines(hit_change_times, 0.05, 0.1, linestyles ="solid", colors ="green")
    #axs[0,0].set_xlabel('stimulus presentations')
    axs[0,0].set_ylabel('Representational Similarity')
    axs[0,0].legend()

    #axs[0,1].plot(timings, C_reduced[0,:], c='purple', alpha=0.2)
    axs[0,1].vlines(hit_change_times, 0.05, 0.1, linestyles ="solid", colors ="green", label='hit trials')
    axs[0,1].vlines(miss_change_times, 0.00, 0.05, linestyles ="solid", colors ="red", label='miss trials')
    axs[0,1].plot(x[:half], C_rd_active[0,:], alpha=0.8, label='active block')
    axs[0,1].plot(x[half:], across_blocks2[0,:], alpha=0.8, label='passive block')
    #axs[0,1].set_xlabel('time in seconds')
    axs[0,1].set_ylabel('Representational Similarity')
    axs[0,1].legend()

    # plot running speed
    y = avg_running_speed
    error = np.array(std_running_speed)

    #axs[1,0].set_title(label='running speed')
    axs[1,0].plot(x[:half], y[:half], label='speed active block')
    axs[1,0].fill_between(x[:half], np.array(y[:half])-error[:half], np.array(y[:half])+error[:half], alpha=0.2)
    axs[1,0].plot(x[:half], y[-half:], label='speed passive block')
    axs[1,0].fill_between(x[:half], np.array(y[-half:])-error[-half:], np.array(y[-half:])+error[-half:], alpha=0.2)
    axs[1,0].set_xlabel('time in seconds')
    axs[1,0].set_ylabel('Running speed (cm/s)')
    axs[1,0].legend()

    #axs[1,1].set_title(label='running speed')
    axs[1,1].plot(x[:half], y[:half], label='run speed')
    axs[1,1].fill_between(x[:half], np.array(y[:half])-error[:half], np.array(y[:half])+error[:half], alpha=0.2)
    axs[1,1].plot(x[half:], y[half:], label='run speed')
    axs[1,1].fill_between(x[half:], np.array(y[half:])-error[half:], np.array(y[half:])+error[half:], alpha=0.2)
    #axs[1,1].axvline(half, c='r')
    axs[1,1].set_xlabel('time in seconds')
    axs[1,1].set_ylabel('Running speed (cm/s)')
    axs[1,1].legend()


    # plot pupil area
    error2 = np.array(std_pupil_area)
    y = avg_pupil_area

    axs[2,0].plot(x[:half], y[:half], label='pupil area - active block')
    axs[2,0].fill_between(x[:half], np.array(y[:half])-error2[:half], np.array(y[:half])+error2[:half], alpha=0.2)
    axs[2,0].plot(x[:half], y[-half:], label='pupil area - passive block')
    axs[2,0].fill_between(x[:half], np.array(y[-half:])-error2[-half:], np.array(y[-half:])+error2[-half:], alpha=0.2)
    axs[2,0].set_xlabel('Stimulus Presentations')
    axs[2,0].set_ylabel('Pupil Area')
    axs[2,0].legend()

    axs[2,1].plot(x[:half], y[:half], label='pupil area')
    axs[2,1].fill_between(x[:half], np.array(y[:half])-error2[:half], np.array(y[:half])+error2[:half], alpha=0.2)
    axs[2,1].plot(x[half:], y[half:], label='pupil area')
    axs[2,1].fill_between(x[half:], np.array(y[half:])-error2[half:], np.array(y[half:])+error2[half:], alpha=0.2)
    axs[2,1].set_xlabel('time in seconds')
    axs[2,1].set_ylabel('Pupil Area')
    axs[2,1].legend()

    return fig


def plot_active_vs_passive_RS_Z_Vector_over_time(timings, C_rd_active, C_rd_passive, across_blocks2,  hit_change_times,
                                                 miss_change_times, z_t_a, z_t_p, half, vol_x, vol_y):

    fig, axs = plt.subplots(2, 2, figsize=(20,9), gridspec_kw={'width_ratios': [1, 2], 'height_ratios': [5, 2]})
    fig.tight_layout()

    x = timings

    # plot representational similarity

    axs[0,0].plot(x[:half], C_rd_active[0,:], label='active block')           #[0, :] takes first row of matrix
    axs[0,0].plot(x[:half], C_rd_passive[0,:], label='passive block')
    # reward volume:
    axs[0,0].plot(vol_x, vol_y, label='reward volume', color='darkblue', alpha=0.1)
    # hit trials
    axs[0,0].vlines(hit_change_times, 0.05, 0.1, linestyles="solid", colors="limegreen")

    axs[0, 0].grid(axis='x', color='0.95')
    axs[0,0].set_ylabel('Representational Similarity and reward volume (ml)')
    axs[0,0].legend()

    axs[0,1].vlines(hit_change_times, 0.05, 0.1, linestyles="solid", colors="limegreen", label='hit trials')
    axs[0,1].vlines(miss_change_times, 0.00, 0.05, linestyles="solid", colors="red", label='miss trials')
    axs[0,1].plot(x[:half], C_rd_active[0,:], alpha=0.8, label='active block')
    axs[0,1].plot(x[half:], across_blocks2[0,:], alpha=0.8, label='passive block')
    # reward volume
    axs[0,1].plot(vol_x, vol_y, label='reward volume', color='darkblue', alpha=0.1)
    # axs[0,1].set_xlabel('time in seconds')
    axs[0, 1].grid(axis='x', color='0.95')
    axs[0,1].set_ylabel('Representational Similarity and reward volume (ml)')
    axs[0,1].legend()

    # axs[1,0].set_title(label='running speed')
    axs[1,0].plot(x[:half], z_t_a[:, 0], label='running speed active block', color='skyblue')
    axs[1,0].plot(x[:half], z_t_a[:, 1], label='pupil area active block', color='violet')
    axs[1,0].plot(x[:half], z_t_p[:, 0], label='running speed passive block', color='steelblue')
    axs[1,0].plot(x[:half], z_t_p[:, 1], label='pupil area passive block', color='mediumpurple')

    axs[1, 0].axhline(y=0, color='gray', linestyle='--', alpha=0.3)
    axs[1, 0].grid(axis='x', color='0.95')
    axs[1,0].set_xlabel('time in seconds')
    axs[1,0].set_ylabel('behaviour z-scored')
    axs[1,0].legend()

    # axs[1,1].set_title(label='running speed')
    axs[1,1].plot(x[:half], z_t_a[:, 0], label='running speed', color='skyblue')
    axs[1,1].plot(x[:half], z_t_a[:, 1], label='pupil area', color='violet')

    axs[1,1].plot(x[half:], z_t_p[:, 0], color='skyblue')
    axs[1,1].plot(x[half:], z_t_p[:, 1], color='violet')

    # axs[1,1].axvline(half, c='r')
    axs[1, 1].axhline(y=0, color='gray', linestyle='--', alpha=0.3)
    axs[1, 1].grid(axis='x', color='0.95')
    axs[1,1].set_xlabel('time in seconds')
    axs[1,1].set_ylabel('behaviour z-scored')
    axs[1,1].legend()

    return fig


def count_all_same_image_appearences(stimulus_presentations):
    """
    counts lengths of same stimulus (image) chains throughout stimulus presentations, while tracking the time medians
    of these chains
    """

    images = stimulus_presentations['image_name'].unique()
    im_counts = {}
    for image in images:
        im_counts[image] = -1

    image_chain_column = np.zeros((len(stimulus_presentations)))
    current_image = 'no'
    for i in range(len(stimulus_presentations)):
        stimulus = stimulus_presentations.iloc[i]
        if stimulus.image_name != current_image:     # if we have a new image
            if not (stimulus.image_name == 'omitted'):   # to skip the omitted pictures and count on
                im_counts[stimulus.image_name] += 1
                image_chain_column[i] = im_counts[stimulus.image_name]
                current_image = stimulus.image_name
        else:                                        # no image change (previous is the same as current one)
            image_chain_column[i] = im_counts[stimulus.image_name]

    return image_chain_column