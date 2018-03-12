import math
import numpy as np
import os



root_folder = r'D:\dev\projects\deep_q_rl\deep_q_rl\results\ATARI'
output_full_path = r'D:\dev\projects\deep_q_rl\deep_q_rl\seq_cluster_full.txt'
output_path = r'D:\dev\projects\deep_q_rl\deep_q_rl\seq_cluster.txt'

all_games = {}

with open(output_full_path, 'w') as out_file:
    for subdir, dirs, files in os.walk(root_folder):
        for d in dirs:
            csv_path = "{}/{}".format(os.path.join(subdir, d), "results.csv")
            results = np.loadtxt(open(csv_path, "rb"), delimiter=",", skiprows=1)
            data = results[:, 3]
            data = (data - data.min())/ (data.max() - data.min())
            # np.convolve(data, kernel, mode='same')
            # off = int(round(len(data)*1.0/32))
            off = int(math.floor(len(data)*1.0/32))
            sub_sample_index = range(off, len(data), off)

            sub_sample = data[sub_sample_index[0:32]]
            print(csv_path, len(data), off, len(sub_sample_index), sub_sample)
            all_games[d] = sub_sample
            out_file.write(d[0:len(d)-21] + "\t")
            for sample in data:
                out_file.write(str(sample) + "\t")
            out_file.write("\n")

print(all_games)

with open(output_path, 'w') as out_file:
    for game in all_games:
        sub_sample = all_games[game]
        out_file.write(game[0:len(game)-21] + "\t")
        for sample in sub_sample:
            out_file.write(str(sample) + "\t")
        out_file.write("\n")
