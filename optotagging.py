import os
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm

def get_square_pulse_optotag_trials(session):
    """
    Returns the trials with square pulses of light from optotagging session.
    """
    square_pulses = session.optotagging_table[session.optotagging_table['stimulus_name']=='pulse']
    return square_pulses

def get_mean_pulse_duration_and_min_break_length_optotagging_session(session):
    """
    Calculates the mean pulse duration of square pulses and the breaks between all stimuli for
    an optoptagging session.
    """
    square_pulses = get_square_pulse_optotag_trials(session)
    mean_duration_square_pulse = np.mean(square_pulses['duration'].values)
    break_lengths = session.optotagging_table['start_time'].values[1:] - session.optotagging_table['stop_time'].values[:-1]
    return mean_duration_square_pulse, break_lengths.min()

def get_aligned_spike_matrix(trials, start_time_label, units, spike_times, inclusive_bin_edges):
    """
    Aligns spikes for reach trial and unit. Uses time stored in start_time_label as reference and
    bins in inclusive_bin_edges.
    """
    spike_matrix = np.zeros((len(trials),  len(units), len(inclusive_bin_edges) - 1))
    print(f'Number of trials: {len(trials)}')
    for pulse_idx, pulse_id in enumerate(trials.index.values):       
  # for pulse_idx, pulse_id in tqdm(enumerate(trials.index.values)):
        for unit_idx, unit_id in enumerate(units.index.values):
            relative_spike_times = spike_times[unit_id] - trials.loc[pulse_id, start_time_label]
            spike_counts_in_range, bins = np.histogram(relative_spike_times, inclusive_bin_edges)
            spike_matrix[pulse_idx, unit_idx, :] = spike_counts_in_range
    return spike_matrix

def convert_spike_matrix_into_firing_rates(spike_matrix, time_resolution):
    """
    Averages over trials for each unit and converts spike count into average firing rate.
    """
    firing_rates = np.mean(spike_matrix, axis=0)/time_resolution
    return firing_rates

def get_firing_rates_in_window(firing_rates, inclusive_bin_edges, window):
    """
    Select only part of firing rate matrix, specified by an interval (window).
    """
    in_range = np.argwhere(np.logical_and(inclusive_bin_edges >= min(window),
                                          inclusive_bin_edges < max(window))).flatten()
    return firing_rates[:, in_range]

def get_cre_pos_unit_ids_from_optotagging(session, structure_acronym=None, **kwargs):
    """
    Extracts all unit ids which fulfill the criteria for cre+ units. Should be run with parameters tested on a few
    sessions by using the rest of this notebook.
    """
    params_keys = ['threshold_increase', 'threshold_mean_evoked_rate', 'epsilon',
                   'left_edge', 'right_edge', 'time_resolution',
                   'evoked_window', 'baseline_window']
    default_params = {'threshold_increase': 4., 'threshold_mean_evoked_rate': 10., 'epsilon': 1e-03,
                      'left_edge': -0.01, 'right_edge': 0.025, 'time_resolution': 0.0005,
                      'evoked_window': [0.002, 0.008], 'baseline_window': [-0.01, -0.002]}
    default_params.update(**kwargs)

    (threshold_increase, threshold_mean_evoked_rate,
     epsilon, left_edge, right_edge, time_resolution,
     evoked_window, baseline_window) = [default_params[key] for key in params_keys]

    inclusive_bin_edges = np.arange(left_edge, right_edge + time_resolution, time_resolution)


    channels = session.get_channels()
    units = session.get_units()
    units = units.merge(channels, left_on='peak_channel_id', right_index=True)

    units = units[(units['quality']=='good') &
                  (units['firing_rate']>1) &
                  (units['snr']>1) &
                  (units['isi_violations']<1)]
    # Select specific units
    if structure_acronym is None:
        selected_units = units
    elif isinstance(structure_acronym, str):
        selected_units = units[units['structure_acronym'].str.contains(structure_acronym)]
    else:
        raise KeyError

    spike_times = session.spike_times
    square_pulses = get_square_pulse_optotag_trials(session)

    aligned_spike_matrix = get_aligned_spike_matrix(square_pulses, 'start_time',
                                                    selected_units, spike_times,
                                                    inclusive_bin_edges)

    firing_rates = convert_spike_matrix_into_firing_rates(aligned_spike_matrix, time_resolution)
    evoked_firing_rates, baseline_firing_rates = [get_firing_rates_in_window(firing_rates,
                                                                             inclusive_bin_edges,
                                                                             window)
                                                  for window in [evoked_window, baseline_window]]

    mean_baseline_rates, mean_evoked_rates = [np.mean(rates, axis=1) for rates in [baseline_firing_rates,
                                                                                   evoked_firing_rates]]
    evoked_mean_ratio = mean_evoked_rates/(mean_baseline_rates + epsilon)

    have_min_ratio = evoked_mean_ratio >= threshold_increase
    have_min_evoked_rate = mean_evoked_rates >= threshold_mean_evoked_rate
    cre_pos_unit_indices = np.argwhere(np.logical_and(have_min_ratio, have_min_evoked_rate))
    cre_pos_unit_ids = selected_units.index.values[cre_pos_unit_indices]

    return cre_pos_unit_ids