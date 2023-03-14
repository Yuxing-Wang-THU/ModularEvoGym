import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import os

curr_dir = os.path.dirname(os.path.abspath(__file__))

sns.set_style("darkgrid")

def getdata(game_name=None, algo_name=None):
    base_files_path =curr_dir + "/saved_data/" + game_name
    files = os.listdir(base_files_path) 
    algo_files = []
    data = []
    for file_name in files:
        if algo_name in file_name:
            algo_files.append(file_name)
    for file in algo_files:
        f_path = os.path.join(base_files_path,file)
        d = pd.read_csv(f_path + "/" + "/fitness_log.csv",
                        header=None,names=['iters', 'timesteps','total_episode','eval_score','train_score']) 
        rewards = list(d['eval_score'])
        data.append(rewards)
    return data

def smooth(data, sm=1):
    if sm > 1:
        smooth_data = []
        for d in data:
            y = np.ones(sm)*1.0/sm
            d = np.convolve(y, d, "same")
            smooth_data.append(d)
    return smooth_data

if __name__ == "__main__":
    game = "Walker-v0"
    algos = ['transformer_PPO','fc_PPO']
    fig = plt.figure(figsize=(7,4))
    datas = []

    for i in range(len(algos)):
        # Get data
        data = getdata(game_name=game, algo_name=algos[i])
        data = smooth(data, sm=2)
        datas.append(data)

    for i in range(len(algos)):
        xdata = np.arange(0.0,1.01,0.01)
        ax = sns.tsplot(time=xdata, data=datas[i],condition=algos[i], color=sns.color_palette()[i], legend=True)

    plt.tick_params(labelsize=16)
    plt.ylabel("Performance", fontsize=16)
    plt.xlabel("Policy iterations $(\\times 10^3)$", fontsize=16)
    plt.title(game, fontsize=18)
    plt.rcParams.update({'font.size': 9.5})
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"./{game}_training_curves.png", dpi=600,pad_inches=0.0)