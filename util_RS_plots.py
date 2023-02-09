from util_eda import average_population_response_over_same_stimulus_presentations, extract_speed_and_pupil_data, plot_active_vs_passive_RS_running_speed_and_pupil_area_over_time, running_speeds_and_pupil_areas, get_unit_activity_vectors_for_image
import numpy as np
import matplotlib.pyplot as plt
from itertools import combinations, product


def specific_image(image_name, M, stimulus_presentations, running_speed, eye_tracking, change_trials):
    """
    plots Representational Similarity, Running Speed and Pupil Area over time for one fixed image
    """

    # extract the full unit activity for certain image
    # complete, b0, b5 = get_unit_activity_vectors_for_image(image_name, M, stimulus_presentations)

    # dimension reduction: average population response over chains of same stimuli (binning and avg)
    M_reduced, M_rd_active, M_rd_passive, timings, timings_start_end = average_population_response_over_same_stimulus_presentations(M,
                                                                                                                                    image_name,
                                                                                                                                    stimulus_presentations)

    # visualization of process:
    # check_reduction_over_stimuli(M_reduced, M_rd_active, M_rd_passive)

    # get the Representational Similarity
    C_reduced = np.corrcoef(np.transpose(M_reduced))            # the Correlation Matrix with reduced stimuli
    C_rd_active = np.corrcoef(np.transpose(M_rd_active))        # within active block
    C_rd_passive = np.corrcoef(np.transpose(M_rd_passive))      # within passive block

    half = len(C_rd_active)

    across_blocks = C_reduced[half:, :half]                    # lower middle correlation matrix
    across_blocks2 = C_reduced[:half, half:]                   # passive block? - lower right corr matrix

    # visualize:
    # corrcoef_matrices_visualization(C_reduced, C_rd_active, C_rd_passive, half)

    # show only representational similarity over time:
    # plot_representational_similarity_over_time(timings, C_reduced, C_rd_active, across_blocks2, half)

    # trials - overall: not specific to image
    # get the hit trials
    hit_trials = change_trials[change_trials.hit == True]
    hit_change_times = hit_trials['change_time_no_display_delay'].to_numpy()
    # get the miss trials
    miss_trials = change_trials[change_trials.miss == True]
    miss_change_times = miss_trials['change_time_no_display_delay'].to_numpy()

    sp = stimulus_presentations[stimulus_presentations.image_name == image_name]
    avg_running_speed, std_running_speed, avg_pupil_area, std_pupil_area = extract_speed_and_pupil_data(sp, timings_start_end, running_speed, eye_tracking)

    # plot_single_behaviour(avg_running_speed, std_running_speed, half, 'running speed')
    # plot_single_behaviour(avg_pupil_area, std_pupil_area, half, 'pupil area')


    # final plot for image:
    fig = plot_active_vs_passive_RS_running_speed_and_pupil_area_over_time(timings,
                                                                           C_rd_active,
                                                                           C_rd_passive,
                                                                           across_blocks2,
                                                                           hit_change_times,
                                                                           miss_change_times,
                                                                           avg_running_speed,
                                                                           std_running_speed,
                                                                           avg_pupil_area,
                                                                           std_pupil_area,
                                                                           half)

    fig.suptitle(t='Representational Similarity and Behaviour over Time of '+ image_name, fontsize=22, y=1.01)


def plot_layered_RS_running_speed_and_pupil_area_all_images(images, M, stimulus_presentations, running_speed, eye_tracking, colors, change_trials, title):
    """
    plots Representational Similarity, Running Speed and Pupil Area over time for all presented images
    """

    fig, axs = plt.subplots(3, 1, figsize=(20,14), gridspec_kw={'height_ratios': [2,1,1]})
    fig.tight_layout()

    # trials - overall: not specific to image
    # get the hit trials
    hit_trials = change_trials[change_trials.hit == True]
    hit_change_times = hit_trials['change_time_no_display_delay'].to_numpy()
    # get the miss trials
    miss_trials = change_trials[change_trials.miss == True]
    miss_change_times = miss_trials['change_time_no_display_delay'].to_numpy()

    axs[0].vlines(hit_change_times, 0.05, 0.1, linestyles ="solid", colors ="green", label='hit trials')
    axs[0].vlines(miss_change_times, 0.00, 0.05, linestyles ="solid", colors ="red", label='miss trials')

    for i, image in enumerate(images):
        # get image specific unit activities
        M_reduced, M_rd_active, M_rd_passive, timings, timings_start_end = average_population_response_over_same_stimulus_presentations(M, image, stimulus_presentations)

        # get the Representational Similarity
        C_reduced = np.corrcoef(np.transpose(M_reduced))            # the Correlation Matrix with reduced stimuli
        C_rd_active = np.corrcoef(np.transpose(M_rd_active))        # within active block
        #C_rd_passive = np.corrcoef(np.transpose(M_rd_passive))      # within passive block

        half = len(C_rd_active)

        across_blocks2 = C_reduced[:half, half:]

        x = timings

        # plot representational similarity
        axs[0].plot(x[:half], C_rd_active[0,:], alpha=0.8, label=image, color=colors[i])
        axs[0].plot(x[half:], across_blocks2[0,:], alpha=0.8, color=colors[i])

        # behaviour:
        sp = stimulus_presentations[stimulus_presentations.image_name == image]
        avg_running_speed, std_running_speed, avg_pupil_area, std_pupil_area = extract_speed_and_pupil_data(sp, timings_start_end, running_speed, eye_tracking)

        # plot running speed
        y = avg_running_speed
        error = np.array(std_running_speed)

        axs[1].plot(x[:half], y[:half], label=image, color=colors[i])
        axs[1].fill_between(x[:half], np.array(y[:half])-error[:half], np.array(y[:half])+error[:half], alpha=0.2, color=colors[i])
        axs[1].plot(x[half:], y[half:], color=colors[i])
        axs[1].fill_between(x[half:], np.array(y[half:])-error[half:], np.array(y[half:])+error[half:], alpha=0.2, color=colors[i])

        # plot pupil area
        error2 = np.array(std_pupil_area)
        y = avg_pupil_area

        axs[2].plot(x[:half], y[:half], label=image, color=colors[i])
        axs[2].fill_between(x[:half], np.array(y[:half])-error2[:half], np.array(y[:half])+error2[:half], alpha=0.2, color=colors[i])
        axs[2].plot(x[half:], y[half:], color=colors[i])
        axs[2].fill_between(x[half:], np.array(y[half:])-error2[half:], np.array(y[half:])+error2[half:], alpha=0.2, color=colors[i])


    axs[0].set_ylabel('Representational Similarity')
    axs[0].legend()

    axs[1].set_xlabel('time in seconds')
    axs[1].set_ylabel('Running speed (cm/s)')
    axs[1].legend()

    axs[2].set_xlabel('time in seconds')
    axs[2].set_ylabel('Pupil Area')
    axs[2].legend()

    fig.suptitle(t=title, fontsize=22, y=1.02)
    plt.savefig(title+'.png', bbox_inches='tight')


def RS_vs_speed(image, M, stimulus_presentations, running_speed, eye_tracking):

    M_avg_im, M_avg_im_a, M_avg_im_passive, timings_im, timings_start_end_im = average_population_response_over_same_stimulus_presentations(M, image, stimulus_presentations)

    # get the Representational Similarity
    C_avg_im = np.corrcoef(np.transpose(M_avg_im))            # the Correlation Matrix with reduced stimuli

    sp = stimulus_presentations[stimulus_presentations.image_name == image]
    avg_running_speed, std_running_speed, avg_pupil_area, std_pupil_area = extract_speed_and_pupil_data(sp, timings_start_end_im, running_speed, eye_tracking)

    assert len(C_avg_im) == len(avg_running_speed)

    y_im = []
    x_im = []
    for pair in combinations(range(len(avg_running_speed)), 2):
        i, j = pair
        delta_V = np.abs(avg_pupil_area[i] - avg_pupil_area[j])
        RS = C_avg_im[i,j]
        x_im.append(delta_V)
        y_im.append(RS)

    fig, axs = plt.subplots(2, 2, figsize=(20, 7), gridspec_kw={'width_ratios': [4, 1], 'height_ratios': [1,2]})
    fig.tight_layout()
    axs[1,0].scatter(x_im, y_im, alpha=0.2, label=image)
    axs[1,0].legend()

    # plot histograms-----------

    # hist.,data, 30 bins, 0.5 opacity, label,       for normalized values??       5/4 of binWidth
    axs[0,0].hist(x_im, bins=30, alpha=0.5, density=True, rwidth=0.8)
    #plt.title(f"Projection to delta V histogramm plot", fontsize=20)

    axs[0,1].set_visible(False)

    # hist.,data, 30 bins, 0.5 opacity, label,       for normalized values??       5/4 of binWidth
    axs[1,1].hist(y_im, bins=30, alpha=0.5, density=True, rwidth=0.8, orientation='horizontal')
    #plt.title(f"Projection to RS histogramm plot", fontsize=20)


def RS_vs_speed_all_images(valid_images, M, stimulus_presentations, running_speed, eye_tracking):
    fig, axs = plt.subplots(1, 2, figsize=(20,14), gridspec_kw={'width_ratios': [2,1]})
    fig.tight_layout()

    for image in valid_images:
        print(image)
        M_avg_im, M_avg_im_a, M_avg_im_passive, timings_im, timings_start_end_im = average_population_response_over_same_stimulus_presentations(M, image, stimulus_presentations)

        # get the Representational Similarity
        C_avg_im = np.corrcoef(np.transpose(M_avg_im))            # the Correlation Matrix with reduced stimuli

        sp = stimulus_presentations[stimulus_presentations.image_name == image]
        avg_running_speed, std_running_speed, avg_pupil_area, std_pupil_area = extract_speed_and_pupil_data(sp, timings_start_end_im, running_speed, eye_tracking)

        assert len(C_avg_im) == len(avg_running_speed)

        y_im = []
        x_im = []
        for pair in combinations(range(int(len(avg_running_speed)/2)), 2):
            i, j = pair
            delta_V = np.abs(avg_pupil_area[i] - avg_pupil_area[j])
            RS = C_avg_im[i,j]
            x_im.append(delta_V)
            y_im.append(RS)

        if image == 'im114_r':
            axs[0].scatter(x_im, y_im, label=image+' active', alpha=0.2, color='blue')

        for pair in combinations(range(int(len(avg_running_speed)/2), len(avg_running_speed)), 2):
            i, j = pair
            delta_V = np.abs(avg_pupil_area[i] - avg_pupil_area[j])
            RS = C_avg_im[i,j]
            #print(delta_V, RS)
            x_im.append(delta_V)
            y_im.append(RS)

        if image == 'im114_r':
            axs[1].scatter(x_im, y_im, label=image+' passive', alpha=0.2, color='orange')


    axs[0].legend()
    axs[1].legend()


def RS_vs_speed_all_s(image, M, stimulus_presentations, running_speed, eye_tracking):

    # get population response for specific image
    complete, b0, b5 = get_unit_activity_vectors_for_image(image, M, stimulus_presentations)

    # get the Representational Similarity
    C = np.corrcoef(np.transpose(complete))            # the Correlation Matrix for one image

    sp = stimulus_presentations[stimulus_presentations.image_name == image]
    running_speeds, pupil_areas = running_speeds_and_pupil_areas(sp, running_speed, eye_tracking)

    assert len(C) == len(running_speeds)
    assert len(C) == len(pupil_areas)


    x_im = []
    y_im = []
    for pair in combinations(range(len(running_speeds)), 2):
        i, j = pair
        delta_V = np.abs(pupil_areas[i] - pupil_areas[j])
        RS = C[i,j]
        x_im.append(delta_V)
        y_im.append(RS)

    fig, axs = plt.subplots(2, 2, figsize=(20,14), gridspec_kw={'width_ratios': [4, 1], 'height_ratios': [1,2]})
    fig.tight_layout()
    axs[1,0].scatter(x_im, y_im, alpha=0.2, label=image)
    axs[1,0].legend()


    # plot histograms-----------

    # hist.,data, 30 bins, 0.5 opacity, label,       for normalized values??       5/4 of binWidth
    axs[0,0].hist(x_im, bins=30, alpha=0.5, density=True, rwidth=0.8)
    #plt.title(f"Projection to delta V histogramm plot", fontsize=20)

    axs[0,1].set_visible(False)

    # hist.,data, 30 bins, 0.5 opacity, label,       for normalized values??       5/4 of binWidth
    axs[1,1].hist(y_im, bins=30, alpha=0.5, density=True, rwidth=0.8, orientation='horizontal')
    #plt.title(f"Projection to RS histogramm plot", fontsize=20)


def RS_vs_speed_active_passive(image, M, stimulus_presentations, running_speed, eye_tracking, behaviour='running', block='both'):

    # get population response for specific image
    complete, b0, b5 = get_unit_activity_vectors_for_image(image, M, stimulus_presentations)
    sp = stimulus_presentations[stimulus_presentations.image_name == image]

    fig, axs = plt.subplots(2, 2, figsize=(20,18), gridspec_kw={'width_ratios': [4, 1], 'height_ratios': [1,2]})
    fig.tight_layout()
    axs[0, 1].set_visible(False)

    red_by = 10
    print(f'plotting every {red_by}-th datapoint')

    # get the Representational Similarity
    if (block == 'both') or (block == 'active-passive'):
        C = np.corrcoef(np.transpose(complete))            # the Correlation Matrix for one image
        running_speeds, pupil_areas = running_speeds_and_pupil_areas(sp, running_speed, eye_tracking)
        assert len(C) == len(running_speeds)
        assert len(C) == len(pupil_areas)
        n = len(C)

        if behaviour == 'pupil':
            behav = pupil_areas
        elif behaviour == 'running':
            behav = running_speeds

        x_im = []
        y_im = []

        a = range(int(n/2))
        p = range(int(n/2), n)
        for pair in product(a, p):
            i, j = pair
            delta_V = np.abs(behav[i] - behav[j])
            RS = C[i, j]
            x_im.append(delta_V)
            y_im.append(RS)

        axs[1, 0].scatter(x_im[::red_by], y_im[::red_by], alpha=0.2, label=image+" "+block)
        axs[1, 0].set_xlabel(behaviour + " difference")
        # plot histograms-----------
        axs[0, 0].hist(x_im[::red_by], bins=30, alpha=0.5, rwidth=0.8)
        axs[1, 1].hist(y_im[::red_by], bins=30, alpha=0.5, rwidth=0.8, orientation='horizontal')

    else:
        C_a = np.corrcoef(np.transpose(b0))
        sp_a = sp[sp.stimulus_block == 0]
        C_p = np.corrcoef(np.transpose(b5))
        sp_p = sp[sp.stimulus_block == 5]
        running_speeds_a, pupil_areas_a = running_speeds_and_pupil_areas(sp_a, running_speed, eye_tracking)
        running_speeds_p, pupil_areas_p = running_speeds_and_pupil_areas(sp_p, running_speed, eye_tracking)

        assert len(C_a) == len(running_speeds_a)
        assert len(C_a) == len(pupil_areas_a)
        assert len(C_a) == len(running_speeds_p)
        assert len(C_p) == len(pupil_areas_p)
        assert len(C_a) == len(C_p)
        n = len(C_a)

        if behaviour == 'pupil':
            behav_a = pupil_areas_a
            behav_p = pupil_areas_p
        elif behaviour == 'running':
            behav_a = running_speeds_a
            behav_p = running_speeds_p

        x_im_a = []
        y_im_a = []
        x_im_p = []
        y_im_p = []

        for pair in combinations(range(n), 2):
            i, j = pair
            delta_V = np.abs(behav_a[i] - behav_a[j])
            RS = C_a[i, j]
            x_im_a.append(delta_V)
            y_im_a.append(RS)
            x_im_p.append(np.abs(behav_p[i] - behav_p[j]))
            y_im_p.append(C_p[i, j])

        x_im_a = x_im_a[::red_by]
        y_im_a = y_im_a[::red_by]
        x_im_p = x_im_p[::red_by]
        y_im_p = y_im_p[::red_by]

        print(len(x_im_a), len(y_im_a))
        axs[1, 0].scatter(x_im_a, y_im_a, alpha=0.1, label=image+" active")
        axs[1, 0].scatter(x_im_p, y_im_p, alpha=0.1, label=image+" passive")

        axs[1, 0].set_xlabel(behaviour + " difference")
        # plot histograms-----------
        axs[0, 0].hist(x_im_a, bins=30, alpha=0.5, rwidth=0.8)
        axs[1, 1].hist(y_im_a, bins=30, alpha=0.5, rwidth=0.8, orientation='horizontal')
        axs[0, 0].hist(x_im_p, bins=30, alpha=0.5, rwidth=0.8)
        axs[1, 1].hist(y_im_p, bins=30, alpha=0.5, rwidth=0.8, orientation='horizontal')

    fig.legend()
    fig.suptitle(t=f'Representational Similarity and {behaviour} Difference of {image} in block: {block}', fontsize=22,
                 y=1.02)


def RS_vs_speed_avg_active_passive(image, M, stimulus_presentations, running_speed, eye_tracking, behaviour):

    M_avg_im, M_avg_im_a, M_avg_im_passive, timings_im, timings_start_end_im = average_population_response_over_same_stimulus_presentations(M, image, stimulus_presentations)

    # get the Representational Similarity
    C_avg = np.corrcoef(np.transpose(M_avg_im))            # the Correlation Matrix with reduced stimuli
    C_avg_a = np.corrcoef(np.transpose(M_avg_im_a))
    C_avg_p = np.corrcoef(np.transpose(M_avg_im_passive))

    sp = stimulus_presentations[stimulus_presentations.image_name == image]
    avg_running_speed, std_running_speed, avg_pupil_area, std_pupil_area = extract_speed_and_pupil_data(sp, timings_start_end_im, running_speed, eye_tracking)

    assert len(C_avg) == len(avg_running_speed)

    assert len(C_avg) == len(avg_running_speed)
    assert len(C_avg_a) == len(avg_running_speed)/2
    assert len(C_avg_p) == len(avg_running_speed)/2

    n = len(C_avg)
    half = int(n/2)

    if behaviour == 'pupil':
        avg_behaviour_a = avg_pupil_area[:half]
        avg_behaviour_p = avg_pupil_area[half:]
    elif behaviour == 'running':
        avg_behaviour_a = avg_running_speed[:half]
        avg_behaviour_p = avg_running_speed[half:]



    x_im_a = []
    y_im_a = []
    x_im_p = []
    y_im_p = []
    x_a_p = []
    y_a_p = []

    for pair in combinations(range(half), 2):
        i, j = pair

        # active
        x_im_a.append(np.abs(avg_behaviour_a[i] - avg_behaviour_a[j]))
        y_im_a.append(C_avg_a[i,j])
        # passive
        x_im_p.append(np.abs(avg_behaviour_p[i] - avg_behaviour_p[j]))
        y_im_p.append(C_avg_p[i,j])


    # across blocks
    #for pair in combinations(range(n), 2):
    #    i, j = pair
    #    delta_V = np.abs(avg_pupil_area[i] - avg_pupil_area[j])
    #    RS = C_avg[i,j]
    #    x_a_p.append(delta_V)
    #    y_a_p.append(RS)

    print(image)

    fig, axs = plt.subplots(2, 2, figsize=(20, 7), gridspec_kw={'width_ratios': [4, 1], 'height_ratios': [1,2]})
    fig.tight_layout()

    # active
    axs[1,0].scatter(x_im_a, y_im_a, alpha=0.2)
    # axs[1,0].legend()
    # hist.,data, 30 bins, 0.5 opacity, label,       for normalized values??       5/4 of binWidth
    axs[0,0].hist(x_im_a, bins=30, alpha=0.3, rwidth=0.8, label='active')
    #plt.title(f"Projection to delta V histogramm plot", fontsize=20)
    axs[0,1].set_visible(False)
    # hist.,data, 30 bins, 0.5 opacity, label,       for normalized values??       5/4 of binWidth
    axs[1,1].hist(y_im_a, bins=30, alpha=0.3, rwidth=0.8, orientation='horizontal')
    #plt.title(f"Projection to RS histogramm plot", fontsize=20)

    # passive
    axs[1,0].scatter(x_im_p, y_im_p, alpha=0.2)
    # axs[1,0].legend()
    # hist.,data, 30 bins, 0.5 opacity, label,       for normalized values??       5/4 of binWidth
    axs[0,0].hist(x_im_p, bins=30, alpha=0.3, rwidth=0.8, label='passive')
    #plt.title(f"Projection to delta V histogramm plot", fontsize=20)
    # hist.,data, 30 bins, 0.5 opacity, label,       for normalized values??       5/4 of binWidth
    axs[1,1].hist(y_im_p, bins=30, alpha=0.3, rwidth=0.8, orientation='horizontal')
    #plt.title(f"Projection to RS histogramm plot", fontsize=20)

    axs[1,0].set_xlabel(behaviour + " difference")

    #axs[1,0].scatter(x_a_p, y_a_p, alpha=0.2)
    #axs[0,0].hist(x_im_p, bins=30, alpha=0.3, rwidth=0.8, label='across')
    #axs[1,1].hist(y_im_p, bins=30, alpha=0.3, rwidth=0.8, orientation='horizontal')

    fig.legend()
    fig.suptitle(t=f'Averaged Representational Similarity and {behaviour} Difference of {image} ', fontsize=22, y=1.05)


def RS_vs_time_avg_active_passive(image, M, stimulus_presentations):

    M_avg_im, M_avg_im_a, M_avg_im_passive, timings_im, timings_start_end_im = average_population_response_over_same_stimulus_presentations(M, image, stimulus_presentations)

    # get the Representational Similarity
    C_avg = np.corrcoef(np.transpose(M_avg_im))            # the Correlation Matrix with reduced stimuli
    C_avg_a = np.corrcoef(np.transpose(M_avg_im_a))
    C_avg_p = np.corrcoef(np.transpose(M_avg_im_passive))

    assert len(C_avg) == len(timings_im)
    assert len(C_avg_a) == len(timings_im)/2
    assert len(C_avg_p) == len(timings_im)/2

    n = len(C_avg)
    half = int(n/2)

    med_timings_a = timings_im[:half]
    med_timings_p = timings_im[half:]


    x_im_a = []
    y_im_a = []
    x_im_p = []
    y_im_p = []

    for pair in combinations(range(half), 2):
        i, j = pair

        # active
        x_im_a.append(np.abs(med_timings_a[i] - med_timings_a[j]))
        y_im_a.append(C_avg_a[i, j])
        # passive
        x_im_p.append(np.abs(med_timings_p[i] - med_timings_p[j]))
        y_im_p.append(C_avg_p[i, j])


    print(image)

    fig, axs = plt.subplots(2, 2, figsize=(20, 7), gridspec_kw={'width_ratios': [4, 1], 'height_ratios': [1,2]})
    fig.tight_layout()
    axs[0,1].set_visible(False)

    # active
    axs[1,0].scatter(x_im_a, y_im_a, alpha=0.2)
    axs[0,0].hist(x_im_a, bins=30, alpha=0.3, rwidth=0.8, label='active')
    axs[1,1].hist(y_im_a, bins=30, alpha=0.3, rwidth=0.8, orientation='horizontal')

    # passive
    axs[1,0].scatter(x_im_p, y_im_p, alpha=0.2)
    axs[0,0].hist(x_im_p, bins=30, alpha=0.3, rwidth=0.8, label='passive')
    axs[1,1].hist(y_im_p, bins=30, alpha=0.3, rwidth=0.8, orientation='horizontal')

    fig.legend()
    fig.suptitle(t='Averaged Representational Similarity and Time Difference of '+ image, fontsize=22, y=1.05)


def RS_vs_time_active_passive_within_blocks(image, M, stimulus_presentations):

    # get population response for specific image
    complete, b0, b5 = get_unit_activity_vectors_for_image(image, M, stimulus_presentations)
    sp = stimulus_presentations[stimulus_presentations.image_name == image]
    timings = get_time_medians(sp)

    fig, axs = plt.subplots(2, 2, figsize=(20,18), gridspec_kw={'width_ratios': [4, 1], 'height_ratios': [1,2]})
    fig.tight_layout()
    axs[0, 1].set_visible(False)

    red_by = 10
    print(f'plotting every {red_by}-th datapoint')

    C = np.corrcoef(np.transpose(complete))
    C_a = np.corrcoef(np.transpose(b0))
    C_p = np.corrcoef(np.transpose(b5))

    assert len(C_a) == len(C_p)
    assert len(C) == len(timings)
    assert len(C_a) == len(timings)/2
    assert len(C_p) == len(timings)/2
    n = len(C)
    half = int(n/2)

    med_timings_a = timings[:half]
    med_timings_p = timings[half:]

    x_im_a = []
    y_im_a = []
    x_im_p = []
    y_im_p = []

    for pair in combinations(range(half), 2):
        i, j = pair
        # active
        x_im_a.append(np.abs(med_timings_a[i] - med_timings_a[j]))
        y_im_a.append(C_a[i, j])
        # passive
        x_im_p.append(np.abs(med_timings_p[i] - med_timings_p[j]))
        y_im_p.append(C_p[i, j])

    # active
    axs[1,0].scatter(x_im_a[::red_by], y_im_a[::red_by], alpha=0.2)
    axs[0,0].hist(x_im_a[::red_by], bins=30, alpha=0.3, rwidth=0.8, label='active')
    axs[1,1].hist(y_im_a[::red_by], bins=30, alpha=0.3, rwidth=0.8, orientation='horizontal')
    # passive
    axs[1,0].scatter(x_im_p[::red_by], y_im_p[::red_by], alpha=0.2)
    axs[0,0].hist(x_im_p[::red_by], bins=30, alpha=0.3, rwidth=0.8, label='passive')
    axs[1,1].hist(y_im_p[::red_by], bins=30, alpha=0.3, rwidth=0.8, orientation='horizontal')

    fig.legend()
    fig.suptitle(t='Representational Similarity and Time Difference of '+ image, fontsize=22, y=1.05)


def get_time_medians(stimulus_presentations):
    timings_start_end = stimulus_presentations[['start_time', 'stop_time']].to_numpy()
    timings = np.median(timings_start_end, axis=1)
    return timings


def corr2_coeff(A, B):
    # Rowwise mean of input arrays & subtract from input arrays themeselves
    A_mA = A - A.mean(1)[:, None]
    B_mB = B - B.mean(1)[:, None]

    # Sum of squares across rows
    ssA = (A_mA**2).sum(1)
    ssB = (B_mB**2).sum(1)

    # Finally get corr coeff
    return np.dot(A_mA, B_mB.T) / np.sqrt(np.dot(ssA[:, None],ssB[None]))


# averaged:
def correlation_between_picture_pair(imageA, imageB, M, stimulus_presentations):
    M_im, M_im_b0, M_im_b5, t, t_S_E = average_population_response_over_same_stimulus_presentations(M, imageA, stimulus_presentations)
    M_im2, M_im2_b0, M_im2_b5, t2, t2_S_E = average_population_response_over_same_stimulus_presentations(M, imageB, stimulus_presentations)

    C = np.corrcoef(np.transpose(M_im))
    minC = np.min(C)
    C2 = np.corrcoef(np.transpose(M_im2))
    minC2 = np.min(C2)
    C_pic_compare = corr2_coeff(np.transpose(M_im), np.transpose(M_im2))
    minC_pic_c = np.min(C_pic_compare)

    min_ = np.min([minC, minC2, minC_pic_c])

    fig, axs = plt.subplots(1, 3, figsize=(20, 20))
    fig.tight_layout()

    im0 = axs[0].imshow(C, vmin= min_, vmax=1)
    axs[0].set_title(imageA, fontsize=18)

    im1 = axs[1].imshow(C2, vmin= min_, vmax=1)
    axs[1].set_title(imageB, fontsize=18)

    im2 = axs[2].imshow(C_pic_compare, vmin= min_, vmax=1)
    axs[2].set_title(f'{imageA} and {imageB}', fontsize=18)

    cax = axs[2].inset_axes([1.1,0,0.05,1])
    fig.colorbar(im2, shrink=0.2, cax=cax)