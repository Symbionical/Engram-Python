import sys
import time
import numpy as np
import keyboard
import pandas as pd


import brainflow
from brainflow.board_shim import BoardShim, BrainFlowInputParams, LogLevels, BoardIds
from brainflow.data_filter import DataFilter, FilterTypes, AggOperations, WindowFunctions, DetrendOperations

OS = " " 
left_mu_beta_ratio = 0

def get_os():
    platform = sys.platform

    if platform == "win32":
        OS = "Windows"
    if platform == "win64":
        OS = "Windows"

def main():
    board_choice = input("Press 1 for Synthetic or 2 for Cyton")

    if board_choice == "1":
        board_id = BoardIds.SYNTHETIC_BOARD.value
    if board_choice == "2":
        board_id = BoardIds.CYTON_BOARD.value

    BoardShim.enable_dev_board_logger()
    params = BrainFlowInputParams()

    if board_id == 0:
        get_os()
        if OS == "Windows":
            pass
        else:
            params.serial_port = "/dev/cu.usbserial-DM01MPXO"
            params.serial_port = "COM4"

    sampling_rate = BoardShim.get_sampling_rate(board_id)
    board = BoardShim(board_id, params)
    nfft = DataFilter.get_nearest_power_of_two(sampling_rate)

    def stop_stream():
        board.stop_stream()
        board.release_session()

    def start_stream():
        board.prepare_session()
        board.start_stream()
        BoardShim.log_message(LogLevels.LEVEL_INFO.value, 'starting stream')
        eeg_channels = BoardShim.get_eeg_channels(board_id)

        while True:
            time.sleep(60)
            data = board.get_board_data()
            right_prefrontal = eeg_channels[1]
            right_hand = eeg_channels[2]
            left_hand = eeg_channels[3]

            #detrend
            DataFilter.detrend(data[right_prefrontal], DetrendOperations.LINEAR.value)
            DataFilter.detrend(data[right_hand], DetrendOperations.LINEAR.value)
            DataFilter.detrend(data[left_hand], DetrendOperations.LINEAR.value)

            # BAND POWERS
            psd_rprefrontal = DataFilter.get_psd_welch(data[right_prefrontal], nfft, nfft // 2, sampling_rate, WindowFunctions.BLACKMAN_HARRIS.value)
            psd_rhand = DataFilter.get_psd_welch(data[right_hand], nfft, nfft // 2, sampling_rate, WindowFunctions.BLACKMAN_HARRIS.value)
            psd_lhand = DataFilter.get_psd_welch(data[left_hand], nfft, nfft // 2, sampling_rate, WindowFunctions.BLACKMAN_HARRIS.value)

            # CREATIVITY INDEX
            rp_band_power_alpha = DataFilter.get_band_power(psd_rprefrontal, 10.0, 12.0)
            rp_band_power_beta = DataFilter.get_band_power(psd_rprefrontal, 14.0, 30.0)
            # print("creativity:", rp_band_power_alpha / rp_band_power_beta) #given by alpha/beta

            # LEFT MOTOR CORTEX MU
            rh_band_power_mu = DataFilter.get_band_power(psd_rhand, 9, 11)
            rh_band_power_beta = DataFilter.get_band_power(psd_rhand, 14.0, 30.0)

            right_mu_beta_ratio = rh_band_power_mu / rh_band_power_beta

            print("right hand:", right_mu_beta_ratio)

            # RIGHT MOTOR CORTEX MU
            lh_band_power_mu = DataFilter.get_band_power(psd_lhand, 9, 11)
            lh_band_power_beta = DataFilter.get_band_power(psd_lhand, 14.0, 30.0)

            old_value_LMBR = left_mu_beta_ratio
            left_mu_beta_ratio = lh_band_power_mu / lh_band_power_beta

            left_diff = old_value_LMBR - left_mu_beta_ratio

            print("left hand:", left_diff)


            # if left_mu_beta_ratio + 3 > right_mu_beta_ratio:
            #     print("left")
            # else:
            #     print("right")


            # # fail test if ratio is not smth we expect
            # if (band_power_alpha / band_power_beta < 100):
            #     raise ValueError('Wrong Ratio')


    start_stream()

if __name__ == "__main__":
    main()