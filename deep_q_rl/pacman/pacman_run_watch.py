""" This script runs a pre-trained network with the game
visualization turned on.

Usage:

ale_run_watch.py NETWORK_PKL_FILE [ ROM ]
"""
import subprocess
import sys


def run_watch():
    command = [sys.executable, './run_pacman.py', '--steps-per-epoch', '0',
               '--test-length', '10000', '--nn-file', sys.argv[1],
               '--display-screen']

    if len(sys.argv) > 2:
        command.extend(['--rom', sys.argv[2]])
        command.extend(['--experiment-prefix', sys.argv[2]])
    if len(sys.argv) > 3:
        command.extend(sys.argv[3:])

    p1 = subprocess.Popen(command)

    p1.wait()


if __name__ == "__main__":
    run_watch()
