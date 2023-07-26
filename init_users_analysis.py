import system_setting_init as ss
from system_setting_init import AREA_SIZE

import numpy as np
import os
from itertools import combinations
import time
import main

import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, IndexLocator, FuncFormatter
from matplotlib.dates import MonthLocator, DateFormatter
plt.rcParams['pdf.fonttype'] = 42 # TrueType
plt.rcParams['ps.fonttype'] = 42 

N_USER = 21                                               # 15C5=3003
N_PILOT=int(N_USER/3)                                     # 
N_AP = 300                                              # [100, 300]

SEED = 100

def main():

    timestamp_started = time.strftime("%Y%m%d-%H%M%S")

    np.random.seed(SEED)
    init_users_list=get_init_users_list(N_USER, N_PILOT)
    n_comb = len(init_users_list)

    failure = np.loadtxt(f"indices_failed_SP_{N_AP}.{N_USER}.{N_PILOT}.seed{SEED}.csv")
    success = np.loadtxt(f"sumrates_SP_{N_AP}.{N_USER}.{N_PILOT}.seed{SEED}.csv")
    idx_success = np.argsort(success)

    mean_distance = np.zeros(shape=(n_comb,1))
    mean_distance_s = np.zeros(shape=(len(success),1))  #print(f"{len(success)}")
    i_f = 0 #print(f"{failure[0]}")
    i_s = 0

    user_positions = ss.generate_positions(N_USER)

    # 중간에 멈췄었음
    mean_distance[:116270,0]=np.loadtxt(f"mean_distance_{N_AP}.{N_USER}.{N_PILOT}.seed{SEED}.csv")
    mean_distance_s[:107342,0]=np.loadtxt(f"mean_distance(successed)_{N_AP}.{N_USER}.{N_PILOT}.seed{SEED}.csv")

    for i, init_users in enumerate(init_users_list):
        if i<116270: # True: # 중간에 멈췄었음
            i_f = 116270 - 107342 - 1
            i_s = 107341
            continue

        list(init_users)
        pair_init_users = list(combinations(init_users,2))
        distances = [get_distance(user_positions[pair[0],:],user_positions[pair[1],:]) for pair in pair_init_users]
        l1_d = np.linalg.norm(distances, 1)/len(pair_init_users)

        mean_distance[i]=l1_d
        np.savetxt(f"mean_distance_{N_AP}.{N_USER}.{N_PILOT}.seed{SEED}.csv", mean_distance[:i+1])
        
        if (i_f<len(failure)) and i==int(failure[i_f]):
            is_failed = True
            i_f += 1
        else:
            is_failed = False
            mean_distance_s[i_s] = l1_d
            np.savetxt(f"mean_distance(successed)_{N_AP}.{N_USER}.{N_PILOT}.seed{SEED}.csv", mean_distance_s[:i_s+1])
            i_s += 1

        print(
                f"Simul[{timestamp_started}] COMB #{i+1} .... {init_users}\n" 
                + f"mean distance: {l1_d}, " + f"failure? : {is_failed}" + f"진행률: {100*(i+1)/n_comb:.3f}%" 
            )

    plot_d(mean_distance, 'mean_distance') #namestr(mean_distance,globals())
    plot_d(mean_distance_s, 'mean_distance_s')
    sorted_mean_distance_s = mean_distance_s[idx_success]
    plot_d(sorted_mean_distance_s, 'sorted_mean_distance_s')

    timestamp_finished = time.strftime("%Y%m%d-%H%M%S")
    print(f"COMPLETED! \n {timestamp_started}~{timestamp_finished} \n SET: {N_AP}.{N_USER}.{N_PILOT}.seed{SEED}")


def get_init_users_list(n_user, n_pilot):
    user_list = np.linspace(0, n_user-1, n_user, dtype=int)
    combo = combinations(user_list, n_pilot)
    return list(combo)

def get_distance(user_pos1, user_pos2): # is_wrapped
    dx = abs(user_pos1[0] - user_pos2[0])
    dx = min(dx, AREA_SIZE[0]-dx)
    dy = abs(user_pos1[1] - user_pos2[1])
    dy = min(dy, AREA_SIZE[1]-dy)
    xyz_offsets = np.array([dx, dy])
    distance = np.linalg.norm(xyz_offsets, 2)
    return distance

def plot_d(x, x_name):
    _, axes = plt.subplots(nrows=1, ncols=1,
                           tight_layout=True)
    
    n_trial = np.size(x)
    cdf_bins = np.arange(n_trial)/float(n_trial-1)
    axes.set_yticks(np.linspace(0, 1, 11))
    axes.plot(x, cdf_bins, color='black', label="cdf")
    axes.xaxis.set_major_locator(MultipleLocator(5)) ## x값이 5의 배수일 때마다 메인 눈금 표시
    axes.xaxis.set_major_formatter('{x:.0f}') ## 메인 눈금이 표시될 형식 
    axes.xaxis.set_minor_locator(MultipleLocator(1)) ## 서브 눈금은 x값이 1의 배수인 경우마다 표시
    axes.set_ylim(0, 1)
    axes.set_xlabel("System throughput [Mbits/sec]")
    axes.set_ylabel("CDF")
    axes.grid(which='minor', alpha=0.5)
    axes.grid(which='major', alpha=1)

    plt.axvline(np.mean(x), color='red', label = "mean", linewidth=1.5)
    axes.legend() 

    #x_name = namestr(x,globals())
    #print(f"{x_name}")
    plt.savefig(f"{x_name}_{N_AP}.{N_USER}.{N_PILOT}.seed{SEED}.pdf")
    plt.savefig(f"{x_name}_{N_AP}.{N_USER}.{N_PILOT}.seed{SEED}.png")        
    return

def namestr(obj, namespace):
    return [name for name in namespace if namespace[name] is obj] #https://stackoverflow.com/questions/592746/how-can-you-print-a-variable-name-in-python
    #return [ k for k,v in locals().items() if v == obj][0]

if __name__=="__main__":
    main()