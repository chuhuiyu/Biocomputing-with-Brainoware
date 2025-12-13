#!/usr/bin/python
# -------------------------------------------------------------
# Author: Hongwei Cai, Huiyu Chu
# Date:   2025-06-06
# Description: Biocomputing with Brainoware
#              Procedure 2 - Reservoir computing hardware properties
#              5. Spatial information processing in organoid reservoir computing
#              This script describes spatial information processing of organoid
#              reservoir computing with complementary stimulation patterns.
# -------------------------------------------------------------
import maxlab
import maxlab.system
import maxlab.chip
import maxlab.util

import time

name_of_configuration = "/home/mxwbio/configs/Organoid_1/o1_pc.cfg"
sti_electrodes = [
    7397,
    11351,
    14023,
    16201,
    17971,
    14463,
    6101,
    9179,
    6109,
    5237,
    9191,
    5245,
]

pattern_1 = [
    0,
    2,
    5,
    7,
    8,
    10,
]  # <-- Modify indices if less or more stimulation electrodes are used
pattern_2 = [
    1,
    3,
    4,
    6,
    9,
    11,
]  # <-- Modify indices if less or more stimulation electrodes are used
patterns = [pattern_1, pattern_2]


## 0. Initialize system into a defined state

maxlab.util.initialize()
maxlab.send(maxlab.chip.Core().enable_stimulation_power(True))

## 1. Load a previously created configuration

array = maxlab.chip.Array("stimulation")
array.load_config(name_of_configuration)

## 2. Connect electrodes to stimulation units and power up stimulation units

stimulation_units = []

for stim_el in sti_electrodes:
    array.connect_electrode_to_stimulation(stim_el)
    stim = array.query_stimulation_at_electrode(stim_el)
    if stim:
        stimulation_units.append(stim)
    else:
        print(
            "No stimulation channel can connect to electrode: " + str(stim_el)
        )


# Download the prepared array configuration to the chip
array.download()

# Prepare commands to power up and power down the two stimulation units


def cmd_power_pattern(seq, pattern, stimulation_units):
    for num in pattern:
        sti_power = (
            maxlab.chip.StimulationUnit(stimulation_units[num])
            .power_up(True)
            .connect(True)
            .set_voltage_mode()
            .dac_source(0)
        )
        seq.append(sti_power)


def cmd_power_down_pattern(seq, pattern, stimulation_units):
    for num in pattern:
        sti_power_down = maxlab.chip.StimulationUnit(
            stimulation_units[num]
        ).power_up(False)
        seq.append(sti_power_down)


## 3. Prepare pulse trains of different patterns


def append_stimulation_pulse(seq, amplitude):
    seq.append(maxlab.chip.DAC(0, 512 - amplitude))
    seq.append(maxlab.system.DelaySamples(4))
    seq.append(maxlab.chip.DAC(0, 512 + amplitude))
    seq.append(maxlab.system.DelaySamples(4))
    seq.append(maxlab.chip.DAC(0, 512))
    return seq


amplitude = 150
sequence1 = maxlab.Sequence()
for pattern in patterns:
    for rep in range(0, 30):
        cmd_power_pattern(sequence1, pattern, stimulation_units)
        append_stimulation_pulse(sequence1, amplitude)
        # Wait 5 s between two pulses
        for _ in range(5):
            sequence1.append(maxlab.system.DelaySamples(20000))
        cmd_power_down_pattern(sequence1, pattern, stimulation_units)

# 4. Deliver pulse trains of all patterns

time.sleep(10)
sequence1.send()
