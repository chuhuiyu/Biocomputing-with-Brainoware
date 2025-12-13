#!/usr/bin/env python
# -------------------------------------------------------------
# Author: Hongwei Cai, Huiyu Chu
# Date:   2025-06-06
# Description: Biocomputing with Brainoware
#              Procedure 3 - Brainoware software framework
#              2. Electrical stimulation and recording
#              This script describes how to stimulate organoids with the selected electrodes
#              and encoded speech information.
# -------------------------------------------------------------

import maxlab
import maxlab.system
import maxlab.chip
import maxlab.util
import os
import time
import pickle

## Define global parameters ##
event_counter = (
    1  # increment when a new pulse is generated, useful in the decoding part
)
audio_number = (
    240  # <-- number of audio clips, modify if using your own datasets
)
stim_amp_mV = 75  # stimulation amplitude in mV

stim_amp_mV = 75  # stimulation amplitude in mV
stim_amp_bits = int(stim_amp_mV / 2.92)  # convert stim amp to DAC bits

with open(
    "./1_Japanese_Vowels_dataset/processed_dataset/selected_stimulation_electrode_indices.pkl",
    "rb",
) as f:
    stimulation_electrode_indices = pickle.load(
        f
    )  # read the two-dimensional index list generated in the last section

name_of_configuration = "/home/mxwbio/configs/251008/25990_1008.cfg"  # <-- modify the path to your saved configuration
stim_electrodes = [
    21672,
    22338,
    21463,
    21026,
    20808,
    12642,
    13731,
    12015,
    9809,
    15128,
    14917,
    18902,
]  # <-- modify the stimulation electrodes based on your own selection


## 0. Initialize the system, enable stimulation power.
maxlab.util.initialize()
maxlab.send(maxlab.chip.Core().enable_stimulation_power(True))
## 1. Create a new array "stimulation" on the server,
##    load the recording electrode configuration,
##    and select stimulation electrodes.
array = maxlab.chip.Array("stimulation")
array.load_config(name_of_configuration)
array.select_stimulation_electrodes(stim_electrodes)

## 2. Connect stimulation electrodes to stimulation units.
stimulation_units = []

for stim_el in stim_electrodes:
    array.connect_electrode_to_stimulation(stim_el)
    stim = array.query_stimulation_at_electrode(stim_el)
    print(f"{stim_el} - Stim Unit: {stim}")
    if len(stim) == 0:
        raise RuntimeError(
            f"No stimulation channel can connect to electrode: {str(stim_el)}"
        )
    stimulation_units.append(stim)
if len(set(stimulation_units)) != len(stimulation_units):
    raise RuntimeError(
        "Multiple stimulation electrodes connected to same stimulation unit."
    )

# Download the prepared array configuration to the chip
array.download()


## 3. Three Helper functions


# 3.1 Prepare commands to power up and power down the two stimulation units
def cmd_power_p(seq, pattern, stimulation_units):
    for num in pattern:
        sti_power = (
            maxlab.chip.StimulationUnit(stimulation_units[num])
            .power_up(True)
            .connect(True)
            .set_voltage_mode()
            .dac_source(0)
        )
        seq.append(sti_power)


def cmd_power_down_p(seq, pattern, stimulation_units):
    for num in pattern:
        sti_power_down = maxlab.chip.StimulationUnit(
            stimulation_units[num]
        ).power_up(False)
        seq.append(sti_power_down)


# 3.2 Prepare commands to append one pulse to an existing sequence
def append_stimulation_pulse(seq, amplitude):
    global event_counter
    event_counter += 1
    seq.append(
        maxlab.Event(
            0,
            1,
            event_counter,
            f"amplitude{amplitude} event_id{event_counter}",
        )
    )
    seq.append(maxlab.chip.DAC(0, 512 - amplitude))
    seq.append(maxlab.system.DelaySamples(15))
    seq.append(maxlab.chip.DAC(0, 512 + amplitude))
    seq.append(maxlab.system.DelaySamples(15))
    seq.append(maxlab.chip.DAC(0, 512))
    return seq


## 4. Construct stimulation command sequence from stimulation indices
sequence1 = maxlab.Sequence()
for num in range(0, audio_number):
    indices_per_audioclip = stimulation_electrode_indices[num]
    for timestep in range(0, 29):
        cmd_power_p(
            sequence1, indices_per_audioclip[timestep], stimulation_units
        )
        append_stimulation_pulse(sequence1, stim_amp_bits)
        cmd_power_down_p(
            sequence1, indices_per_audioclip[timestep], stimulation_units
        )
        sequence1.append(maxlab.system.DelaySamples(1970))
    sequence1.append(maxlab.system.DelaySamples(22000))
    sequence1.append(maxlab.system.DelaySamples(20000))

## 5. Start recording, offset the signal, send the command sequence, and stop recording.

### start recording ###
data_directory = "/home/mxwbio/Desktop/huiyu/projects/Protocol/p3_version2_251010/2_Electrical_stimulation_and_recording_data"  # <-- modify the path to your data directory
os.makedirs(data_directory, exist_ok=True)
s = maxlab.Saving()
s.open_directory(data_directory)
s.group_delete_all()
s.group_define(0, "routed")
print(f"MaxOne: Start recording at {data_directory}")
time_start = str(time.strftime("%Y-%m-%d-%H-%M-%S", time.gmtime()))
recording_file_name = f"speech_recognition_{time_start}"
s.start_file(recording_file_name)
s.start_recording([0])

### offset the signal ###
maxlab.offset()
time.sleep(10)

### send the pre-defined sequence ###
sequence1.send()

### stop recording when all pulses are delivered ###
time.sleep(240 * 5 + 10)
s.stop_recording()
s.stop_file()
