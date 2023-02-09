# function collection for setup and plotting

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.patches import Rectangle
from allensdk.brain_observatory.behavior.behavior_project_cache. \
    behavior_neuropixels_project_cache \
    import VisualBehaviorNeuropixelsProjectCache


def set_up_cache():
    """
    specifying data cache directory, import the <code>VisualBehaviorNeuropixelsProjectCache</code>  class and instantiate it
    """
    cache_dir = "/Users/kiracordes/allenSDK/VBNcache"

    cache = VisualBehaviorNeuropixelsProjectCache.from_s3_cache(cache_dir=cache_dir)
    if cache.latest_manifest_file() != cache.current_manifest():
        cache.load_manifest(cache.latest_manifest_file())
    
    assert cache.latest_manifest_file() == cache.current_manifest()

    return cache


def get_session(session_id=1053941483, cache=None):
    """
    shortcut for setting up cache and loading session
    (default is a novel sst mice session)
    """
    if cache is None:
        cache = set_up_cache()

    session = cache.get_ecephys_session(ecephys_session_id=session_id)
    return session


def main_meta_data_description():
    cache = set_up_cache()
    ecephys_sessions_table = cache.get_ecephys_session_table()
    ecephys_sessions_table.head()
    print('how many novel and familar sessions?')
    print(ecephys_sessions_table.experience_level.value_counts())
    print('how many G and H sessions?')
    print(ecephys_sessions_table['image_set'].value_counts())
    print('how many sessions were NOVEL or FAMILIAR for each image set?')
    print(ecephys_sessions_table[['experience_level', 'image_set']].value_counts())  # argument normalize=True to get percentages
    print('how many unique genotypes?')
    print(ecephys_sessions_table.genotype.unique())
    print('how many sessions for each?')
    print(ecephys_sessions_table.genotype.value_counts())

    # filter for novel sessions:
    # novel_sessions = ecephys_sessions_table[ecephys_sessions_table['experience_level']=='Novel']
    # novel_sessions.head()

    # filter for novel sessions in sst mice:
    # sst_novel_sessions = ecephys_sessions_table[(ecephys_sessions_table['genotype'].str.contains('Sst')) & (ecephys_sessions_table['experience_level']=='Novel')]
    # sst_novel_sessions.head()


def filtered_spike_times_for_unit(u_id, spike_times, firstXsecs, session):

    units = session.get_units()
    channels = session.get_channels()
    units = units.merge(channels, left_on='peak_channel_id', right_index=True)
    
    brain_structure_ac = units['structure_acronym'][u_id]

    unit_spikes = spike_times[u_id]
    secs = np.where(unit_spikes < (firstXsecs + 0.5))
    secs = secs[0][-1:][0]

    return brain_structure_ac, secs, unit_spikes


def plot_unit_spikes(u_id, spike_times, firstXsecs, session):
    """
    Plot a raster of the spike times for unit {u_id} over the first seconds of the recording.
    Where is this unit located? Put the unit id and the brain region in the plot title.

    u_id: unit that should be observed
    :param firstXsecs: for the first x seconds
    :param session: session object
    :return: plots spike times
    """
    brain_structure_ac, secs, unit_spikes = filtered_spike_times_for_unit(u_id, spike_times, firstXsecs, session)

    fig, ax = plt.subplots()
    fig.set_size_inches(18, 4)

    ax.set_title(f'Brain Region {brain_structure_ac} with Unit ID: {u_id}')
    ax.set_xlabel('seconds')
    ax.axes.get_yaxis().set_visible(False)

    ax.eventplot(unit_spikes[:secs])
    plt.show()

    return


def trials_data_description(trials):
    print(trials.value_counts('go'))
    print(trials.value_counts('catch'))

    # Get number of go trials
    total_go_trials = trials.go.value_counts()[1]
    print(f'Total go trials: {total_go_trials}')

    print(trials.value_counts('hit'))
    print(trials.value_counts('miss'))
    print(trials.value_counts('aborted'))
    print(trials.value_counts('false_alarm'))
    print(trials.value_counts('correct_reject'))
    print(trials.value_counts('auto_rewarded'))

    # Get number of hits
    hit_trials = trials[trials['hit']]
    print(f'Total hits: {len(hit_trials)}')

    # Get hit rate
    print('hit rate:')
    print(len(hit_trials) / total_go_trials)


def determine_color(trial):
    if trial.hit:
        return 'hit', 'tab:green'
    elif trial.miss:
        return 'miss', 'tab:olive'
    elif trial.false_alarm:
        return 'false alarm', 'tab:orange'
    elif trial.correct_reject:
        return 'correct reject', 'tab:blue'
    elif trial.aborted:
        return 'aborted', 'tab:gray'
    else:
        return 'auto rewarded', 'tab:purple'


def plot_trials_chron(trials):
    fig, ax = plt.subplots(figsize=(len(trials), len(trials)))

    yticks = []
    ylabels = []

    for i in range(len(trials)):
        trial = trials.iloc[i]
        l_times = trial.lick_times
        trial_type, color = determine_color(trial)
        ax.broken_barh([(trial.start_time, trial.stop_time-trial.start_time)], ((i+1)*5, 5), facecolors=color)

        for lick in l_times:
            ax.broken_barh([(lick, 0.01)], ((i+1)*5, 5), facecolors='tab:red')

        yticks = yticks + [7.5 + i*5]
        ylabels = ylabels + [f'trial {i}']

    ax.set_ylim(5, 70)
    min_x = int(trials.iloc[0].start_time)
    max_x = round(trials.iloc[-1].stop_time)
    ax.set_xlim(min_x, max_x)
    ax.set_xlabel('seconds since start')
    ax.set_yticks(yticks)
    ax.set_xticks(range(min_x, max_x, 5))
    ax.set_yticklabels(labels=ylabels)     # Modify y-axis tick labels
    ax.grid(True)
    # ax.set_labels(labels=['First line', 'Second line'])
    legend_elements = [Patch(facecolor='tab:gray', edgecolor='b', label='aborted'),
                       Patch(facecolor='tab:green', edgecolor='b', label='hit'),
                       Patch(facecolor='tab:olive', edgecolor='b', label='miss'),
                       Patch(facecolor='tab:orange', edgecolor='b', label='false alarm'),
                       Patch(facecolor='tab:blue', edgecolor='b', label='correct reject'),
                       Patch(facecolor='tab:purple', edgecolor='b', label='auto rewarded')]

    ax.legend(handles=legend_elements, loc='upper left', fontsize='large')
    # labels=['lick', 'aborted'], labelcolor=['tab:red', 'tab:gray'])

    plt.show()


def plot_trials_timeline(trials):
    fig, ax = plt.subplots(figsize=(len(trials)*8,4))
    ax.set(title="trials timeline")

    for i in range(len(trials)):
        trial = trials.iloc[i]
        l_times = trial.lick_times
        trial_type, trial_color = determine_color(trial)
        ax.broken_barh([(trial.start_time, trial.stop_time-trial.start_time)], (5, 1), facecolors=trial_color)
        for lick in l_times:
            ax.broken_barh([(lick, 0.01)], (5, 1), facecolors='tab:red')

    ax.set_ylim(4.5, 6.5)
    min_x = int(trials.iloc[0].start_time)
    max_x = round(trials.iloc[-1].stop_time)
    ax.set_xlim(min_x, max_x)
    ax.set_xlabel('seconds since start')
    ax.set_xticks(range(min_x, max_x))

    ax.grid(True)

    plt.show()


def plot_one_trial(trial, size=(8.8, 2), length=15, dense_view=False):

    fig, ax = plt.subplots(figsize=size, constrained_layout=True)
    trial_type, trial_color = determine_color(trial)
    ax.add_patch(Rectangle((trial.start_time, 0), trial.trial_length, 1.5, facecolor=trial_color, alpha=0.2, label=trial_type+' trial'))

    ax.vlines([trial.start_time, trial.stop_time], 0, 1.5, alpha=0.1)

    ax.vlines(trial.lick_times, 0, 1, color="tab:red", label='licks')

    trial_kind = ''
    if trial.go:
        ax.vlines(trial.change_time_no_display_delay, 0, 1.5, color='darkslateblue', label='image change no dd')
        ax.vlines(trial.response_time, 0, 1, color='dodgerblue', label='response', linestyles=':')
        ax.scatter(trial.reward_time, 0.5, 9, color='dodgerblue', label='reward', marker='*')
        trial_kind = 'go'
    elif trial.auto_rewarded:
        ax.vlines(trial.change_time_no_display_delay, 0, 1.5, color='darkslateblue', label='image change no dd')
        ax.vlines(trial.response_time, 0, 1, color='dodgerblue', label='response', linestyles=':')
        ax.scatter(trial.reward_time, 0.5, 9, color='dodgerblue', label='reward', marker='*')
        trial_kind = 'a_r'
    elif trial.catch:
        trial_kind = 'catch'

    # ax.set(title=f"{trial_kind} trial nr.{trial.name}: {trial_type}", loc='left')
    if not dense_view:
        ax.set_title(f"{trial_kind} trial nr.{trial.name}: {trial_type}")

    # remove y-axis and spines
    ax.yaxis.set_visible(False)
    ax.set_xticks(np.arange(trial.start_time, trial.start_time + length, step=1))
    ax.spines[["left", "top", "right"]].set_visible(False)
    ax.grid(True)
    ax.legend()
    plt.show()


def plot_trials(trials, number_of_trials, size=(13, 1.2), length=15, dense_view=True):
    for i in range(number_of_trials):
         plot_one_trial(trials.iloc[i], size=size, dense_view=dense_view)


def horizontal_bar_plot_trials(trials):
    # Horizontal bar plot with gaps
    fig, ax = plt.subplots(figsize=(17, 7))

    yticks = []
    ylabels = []

    for i in range(17):
        trial = trials.iloc[i]
        l_times = trial.lick_times
        trial_type, trial_color = determine_color(trial)
        ax.broken_barh([(trial.start_time, trial.stop_time-trial.start_time)], ((i+1)*5, 5), facecolors=trial_color)
        for lick in l_times:
            ax.broken_barh([(lick, 0.01)], ((i+1)*5, 5), facecolors='tab:red')

        yticks = yticks + [7.5 + i*5]
        ylabels = ylabels + [f'trial {i}']

    ax.set_ylim(5, 70)
    ax.set_xlim(25, 86)
    ax.set_xlabel('seconds since start')
    ax.set_yticks(yticks)
    ax.set_xticks(range(25, 86, 5))
    ax.set_yticklabels(labels=ylabels)     # Modify y-axis tick labels
    ax.grid(True)                                       # Make grid lines visible
    # ax.annotate('trial aborted', (61, 25),
    #            xytext=(0.8, 0.9), textcoords='axes fraction',
    #            arrowprops=dict(facecolor='black', shrink=0.05),
    #            fontsize=10,
    #            horizontalalignment='right', verticalalignment='top')

    plt.show()


def plot_one_trial_advanced(trial, change_time, size=(8.8, 2), length=15, dense_view=False):

    fig, ax = plt.subplots(figsize=size, constrained_layout=True)
    trial_type, trial_color = determine_color(trial)
    ax.add_patch(Rectangle((trial.start_time, 0), trial.trial_length, 1.5, facecolor=trial_color, alpha=0.2, label=trial_type+' trial'))

    ax.vlines([trial.start_time, trial.stop_time], 0, 1.5, alpha=0.1)

    ax.vlines(trial.lick_times, 0, 1, color="tab:red", label='licks')
    
    if trial.is_change:
        delay = change_time - trial.change_time_no_display_delay
        ax.vlines(change_time, 0, 1.5, color='hotpink', label=f'change time with display delay: {round(delay, 4)}')

    trial_kind = ''
    if trial.go:
        ax.vlines(trial.change_time_no_display_delay, 0, 1.5, color='darkslateblue', label='image change no dd')
        ax.vlines(trial.response_time, 0, 1, color='dodgerblue', label='response', linestyles=':')
        ax.scatter(trial.reward_time, 0.5, 9, color='dodgerblue', label='reward', marker='*')
        trial_kind = 'go'
    elif trial.auto_rewarded:
        ax.vlines(trial.change_time_no_display_delay, 0, 1.5, color='darkslateblue', label='image change no dd')
        ax.vlines(trial.response_time, 0, 1, color='dodgerblue', label='response', linestyles=':')
        ax.scatter(trial.reward_time, 0.5, 9, color='dodgerblue', label='reward', marker='*')
        trial_kind = 'a_r'
    elif trial.catch:
        trial_kind = 'catch'

    # ax.set(title=f"{trial_kind} trial nr.{trial.name}: {trial_type}", loc='left')
    if not dense_view:
        ax.set_title(f"{trial_kind} trial nr.{trial.name}: {trial_type}")

    # remove y-axis and spines
    ax.yaxis.set_visible(False)
    ax.set_xticks(np.arange(trial.start_time, trial.start_time + length, step=1))
    ax.spines[["left", "top", "right"]].set_visible(False)
    ax.grid(True)
    ax.legend()
    plt.show()
    

def plot_all_trials(trials, change_times, number_of_trials, size=(13, 1.2), length=15, dense_view=True):
    change_index = 0
    for i in range(number_of_trials):
        plot_one_trial_advanced(trials.iloc[i], change_times.iloc[change_index], size=size, dense_view=dense_view)
        if trials.iloc[i].is_change:
            change_index += 1


def get_real_change_times(psth_trials, stimulus_presentations):
    
    #Get the change frames for these trials
    change_frames = psth_trials.change_frame.values
    
    #Find the flashes in the stimulus_presentations table that started on these frames
    flashes = stimulus_presentations[np.isin(stimulus_presentations.start_frame, change_frames)]
    
    #Get the display start times for these flashes
    change_times = flashes.start_time.values
    
    return change_times


def makePSTH(spikes, startTimes, windowDur, binSize=0.001):
    '''
    Convenience function to compute a peri-stimulus-time histogram
    (see section 7.2.2 here: https://neuronaldynamics.epfl.ch/online/Ch7.S2.html)
    INPUTS:
        spikes: spike times in seconds for one unit
        startTimes: trial start times in seconds; the first spike count
            bin will be aligned to these times
        windowDur: trial duration in seconds
        binSize: size of spike count bins in seconds
    OUTPUTS:
        Tuple of (PSTH, bins), where:
            PSTH gives the trial-averaged spike rate for
                each time bin aligned to the start times;
            bins are the bin edges as defined by numpy histogram
    '''
    bins = np.arange(0,windowDur+binSize,binSize)
    counts = np.zeros(bins.size-1)
    for start in startTimes:
        startInd = np.searchsorted(spikes, start)
        endInd = np.searchsorted(spikes, start+windowDur)
        counts = counts + np.histogram(spikes[startInd:endInd]-start, bins)[0]

    counts = counts/len(startTimes)
    return counts/binSize, bins[:-1]


def plot20unitsPSTH(trial_start_times, units=None, windowDur=1, binSize=0.01, title='', layered=None, label=''):
    '''
    plots 20 PSTHs for different units at once with given change times
    INPUTS:
        units: optional, the 20 unit ids we'd like to see plotted, by default takes the first 20 'good' VISps
        trial_start_times: trial start times in seconds; the first spike count
                            bin will be aligned to these times
        windowDur: trial duration in seconds -> PSTH argument
        binSize: size of spike count bins in seconds -> PSTH argument
    OUTPUTS:
        PSTH Diagrams for 20 units
    '''
    if units==None:
        units = good_units[good_units['structure_acronym']=='VISp'].index.values[:20] #just take the first twenty good VISp units

    fig, axes = plt.subplots(nrows=5, ncols=4, figsize=(21,9), sharex=True, sharey=True)
    for i, ax in enumerate(axes.flatten()):
        unit_id = units[i]

        # get spike times of iths unit
        unit_spike_times = spike_times[unit_id]

        # Compute the PSTH
        unit_psth, bins = makePSTH(unit_spike_times, trial_start_times, windowDur, binSize)

        if layered is not None:
            other_unit_psth, other_bins = makePSTH(unit_spike_times, layered, windowDur, binSize)
            ax.plot(other_bins, other_unit_psth, label='miss', alpha=0.7)
            #ax.plot(other_bins, np.abs(unit_psth - other_unit_psth), label='diff', alpha=0.5)


        ax.plot(bins, unit_psth, label=label, alpha=0.9)
        ax.set_title(f'unit id: {unit_id}')
        #ax.set_xticks(df.iloc[:,0])
        ax.legend()

    fig.suptitle(f'{title} trials PSTH (averaged spike times per unit)', fontsize=26)


    plt.show()


def plot20units_over_time(trial_start_times, units=None, windowDur=0.25, binSize=0.01, title='', layered=None, label=''):
    '''
    plots 20 PSTHs for different units at once with given change times
    INPUTS:
        units: optional, the 20 unit ids we'd like to see plotted, by default takes the first 20 'good' VISps
        trial_start_times: trial start times in seconds; the first spike count
                            bin will be aligned to these times
        windowDur: trial duration in seconds -> PSTH argument
        binSize: size of spike count bins in seconds -> PSTH argument
    OUTPUTS:
        PSTH Diagrams for 20 units
    '''
    if units==None:
        units = good_units[good_units['structure_acronym']=='VISp'].index.values[:20] #just take the first twenty good VISp units

    alpha_steps = 1/len(trial_start_times)

    fig, axes = plt.subplots(nrows=5, ncols=4, figsize=(21,9), sharex=True, sharey=True)
    for i, ax in enumerate(axes.flatten()):
        unit_id = units[i]

        # get spike times of iths unit
        unit_spike_times = spike_times[unit_id]

        unit_psth, bins = makePSTH(unit_spike_times, trial_start_times, windowDur, binSize)
        ax.plot(bins, unit_psth, label='psth', alpha=0.9)

        alph = 1.0
        for start_time in trial_start_times:
            sInd = np.searchsorted(unit_spike_times, start_time)
            eInd = np.searchsorted(unit_spike_times, start_time+windowDur)

            ax.vlines(unit_spike_times[sInd:eInd]-start_time, 0, 5, alpha=alph, colors=['r', 'g', 'b'])
            alph = alph - alpha_steps

        if layered is not None:
            other_unit_psth, other_bins = makePSTH(unit_spike_times, layered, windowDur, binSize)
            ax.plot(other_bins, other_unit_psth, label='miss', alpha=0.7)
            # ax.plot(other_bins, np.abs(unit_psth - other_unit_psth), label='diff', alpha=0.5)

        ax.set_title(f'unit id: {unit_id}')
        ax.legend()

    fig.suptitle(f'{title} trials PSTH (averaged spike times per unit)', fontsize=26)

    plt.show()


# stimulus presentations plot:
def plot_stimulus_presentations(stimulus_presentations, size=(180, 2), block=0, activities=None):

    if activities is not None:
        fig, (ax0, ax1) = plt.subplots(2, figsize=size) # sharex=True
    else:
        fig, ax0 = plt.subplots(figsize=size)

    for i in range(len(stimulus_presentations)):
        stimulus = stimulus_presentations.iloc[i]


        color = util.image_to_color(stimulus.image_name)
        ax0.add_patch(Rectangle((stimulus.start_time, 0), stimulus.duration, 0.1, facecolor=color, alpha=0.8))

    # ax.vlines([trial.start_time, trial.stop_time], 0, 1.5, alpha=0.1)
    # ax.vlines(trial.lick_times, 0, 1, color="tab:red", label='licks')

    # remove y-axis and spines
    # ax.yaxis.set_visible(False)
    ax0.set_yticks([0, 1])
    ax0.set_xticks(np.arange(25.0, 800.0, step=1))
    ax0.spines[["left", "top", "right"]].set_visible(False)
    ax0.grid(True)
    ax0.margins(x=0, y=0)

    # plt.ylim(0, 1)
    plt.tight_layout()
    # ax.legend()

    if activities is not None:
        # plt.figure(figsize=(15, 1), dpi=180)
        ax1.imshow(activities, aspect='auto')


# stimulus presentations plot:
def plot_stimulus_presentations_complete_block(stimulus_presentations, size=(180, 70), block=0):

    number_of_subplots = int(len(stimulus_presentations)/200)
    print(number_of_subplots)

    fig, axs = plt.subplots(number_of_subplots, figsize=size)
    fig.suptitle(f"Stimulus Presentation Block {block}", fontsize=150)

    for sub in range(number_of_subplots):
        for i in range(200*sub, (sub+1)*200):
            stimulus = stimulus_presentations.iloc[i]

            color = util.image_to_color(stimulus.image_name)
            axs[sub].add_patch(Rectangle((stimulus.start_time, 0), stimulus.duration, 0.1, facecolor=color, alpha=0.8))


        # remove y-axis and spines
        # ax.yaxis.set_visible(False)
        axs[sub].set_yticks([0, 1])
        axs[sub].set_xticks(np.arange(25.0, 800.0, step=1))
        axs[sub].spines[["left", "top", "right"]].set_visible(False)
        axs[sub].grid(True)
        axs[sub].margins(x=0, y=0)
        #axs[sub].ylim(lower_limit, upper_limit)
        #ax.legend()

    #ax.vlines([trial.start_time, trial.stop_time], 0, 1.5, alpha=0.1)
    #ax.vlines(trial.lick_times, 0, 1, color="tab:red", label='licks')
    patches = []

    patches.append(Patch(color='mediumpurple', label='im_111_r'))
    patches.append(Patch(color='mediumorchid', label='im_083_r'))
    patches.append(Patch(color='crimson', label='im_104_r'))
    patches.append(Patch(color='salmon', label='im_114_r'))
    patches.append(Patch(color='orangered', label='im_024_r'))
    patches.append(Patch(color='firebrick', label='im_005_r'))
    patches.append(Patch(color='indianred', label='im_087_r'))
    patches.append(Patch(color='tomato', label='im_034_r'))

    plt.legend(handles=patches, bbox_to_anchor=(1.05, 1.04), fontsize=150)

    plt.show()


def image_to_color(image):
    if image == 'im036_r':
        return 'lightsteelblue'
    elif image == 'im012_r':
        return 'cornflowerblue'
    elif image == 'im115_r':
        return 'royalblue'
    elif image == 'im047_r':
        return 'darkblue'
    elif image == 'im044_r':
        return 'lightskyblue'
    elif image == 'im078_r':
        return 'powderblue'
    elif image == 'im111_r':
        return 'mediumpurple'
    elif image == 'im083_r':
        return 'mediumorchid'
    elif image == 'im104_r':
        return 'crimson'
    elif image == 'im114_r':
        return 'salmon'
    elif image == 'im024_r':
        return 'orangered'
    elif image == 'im005_r':
        return 'firebrick'
    elif image == 'im087_r':
        return 'indianred'
    elif image == 'im034_r':
        return 'tomato'
    else:
        return 'b'
    
