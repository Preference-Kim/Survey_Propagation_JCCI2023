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

    #failure = np.loadtxt(f"indices_failed_SP_{N_AP}.{N_USER}.{N_PILOT}.seed{SEED}.csv")
    sumrates = np.loadtxt(f"sumrates_SP_{N_AP}.{N_USER}.{N_PILOT}.seed{SEED}.csv")
    success = sumrates[~np.isnan(sumrates)]
    idx_success = np.argsort(success)

    k_means_distance = np.zeros(shape=(n_comb,1))
    k_means_obj = np.zeros(shape=(n_comb,1))
    k_means_distance_s = np.zeros(shape=(len(success),1))  #print(f"{len(success)}")
    k_means_obj_s = np.zeros(shape=(len(success),1))  #print(f"{len(success)}")
    i_s = 0

    user_positions = ss.generate_positions(N_USER)

    for i, init_users in enumerate(init_users_list):
        #if i<116270: # True: # 중간에 멈췄었음
        #    i_f = 116270 - 107342 - 1
        #    i_s = 107341
        #    continue

        list(init_users)

        k_distances = get_kmeansDistance(user_positions[init_users,:]) # user_positions[init_users,:]
        k_means_distance[i] = np.linalg.norm(k_distances, 1)/N_PILOT
        k_means_obj[i] = np.linalg.norm(k_distances, 2)**2


        np.savetxt(f"k_means_distance_{N_AP}.{N_USER}.{N_PILOT}.seed{SEED}.csv", k_means_distance[:i+1])
        np.savetxt(f"k_means_obj_{N_AP}.{N_USER}.{N_PILOT}.seed{SEED}.csv", k_means_obj[:i+1])
        
        if np.isnan(sumrates[i]):     #(i_f<len(failure)) and i==int(failure[i_f]):
            is_failed = True
                #i_f += 1
        else:
            is_failed = False
            k_means_distance_s[i_s] = k_means_distance[i]
            k_means_obj_s[i_s] = k_means_obj[i]
            np.savetxt(f"k_means_distance(successed)_{N_AP}.{N_USER}.{N_PILOT}.seed{SEED}.csv", k_means_distance_s[:i_s+1])
            np.savetxt(f"k_means_obj(successed)_{N_AP}.{N_USER}.{N_PILOT}.seed{SEED}.csv", k_means_obj_s[:i_s+1])
            i_s += 1

        print(
                f"Simul[{timestamp_started}] COMB #{i+1} .... {init_users}\n" 
                + f"distance: {k_means_distance[i]}, " + f"KMEANSobj: {k_means_obj[i]}"+ f"failure? : {is_failed}" + f"진행률: {100*(i+1)/n_comb:.3f}%" 
            )

    plot_d(k_means_distance, 'k_means_distance') #namestr(mean_distance,globals())
    plot_d(k_means_obj, 'k_means_obj')
    plot_d(k_means_distance_s, 'k_means_distance_s')
    plot_d(k_means_obj_s, 'k_means_obj_s')
    sorted_k_means_distance_s = k_means_distance_s[idx_success]
    plot_d(sorted_k_means_distance_s, 'sorted_k_means_distance_s')
    sorted_k_means_obj_s = k_means_obj_s[idx_success]
    plot_d(sorted_k_means_obj_s, 'sorted_k_means_obj_s')    

    timestamp_finished = time.strftime("%Y%m%d-%H%M%S")
    print(f"COMPLETED! \n {timestamp_started}~{timestamp_finished} \n SET: {N_AP}.{N_USER}.{N_PILOT}.seed{SEED}")


def get_init_users_list(n_user, n_pilot):
    user_list = np.linspace(0, n_user-1, n_user, dtype=int)
    combo = combinations(user_list, n_pilot)
    return list(combo)

def get_distance(user_pos1, user_pos2): # is_wrapped
    dx = abs(user_pos1[0] - user_pos2[0])
    dy = abs(user_pos1[1] - user_pos2[1])
    xyz_offsets = np.array([dx, dy])
    distance = np.linalg.norm(xyz_offsets, 2)
    return distance

def get_kmeansDistance(init_user_positions): # user_positions[init_users,:]
    centroid = np.mean(init_user_positions, axis=0)
    k_distance = [get_distance(centroid, init_user_positions[i,:]) for i in range(N_PILOT)]
    return k_distance

def plot_d(x, x_name): 
    _, axes = plt.subplots(nrows=1, ncols=1,
                           tight_layout=True)
    
    n_trial = np.size(x)
    cdf_bins = np.arange(n_trial)/float(n_trial-1)
    axes.set_xticks(np.linspace(0, 10, 101)) #axes.set_yticks(np.linspace(0, 1, 11))
    axes.plot(cdf_bins, x, 'bo', markersize=.1)
    axes.yaxis.set_major_locator(MultipleLocator(20)) ## x값이 5의 배수일 때마다 메인 눈금 표시
    axes.yaxis.set_major_formatter('{x:.0f}') ## 메인 눈금이 표시될 형식 
    axes.yaxis.set_minor_locator(MultipleLocator(5)) ## 서브 눈금은 x값이 1의 배수인 경우마다 표시
    axes.set_xlim(0, 1)
    axes.set_xlabel("Order(%)")
    axes.set_ylabel("Mean distance(m)")
    axes.grid(which='minor', alpha=0.5)
    axes.grid(which='major', alpha=1)

    if x_name=='sorted_mean_distance_s': 
        plt.axvline(.25, color='red', label = "Mean performance", linewidth=.4)
        plt.axvspan(0, .1, facecolor='gray', label = "Outage zone", alpha=0.15)
    axes.legend() 

    #x_name = namestr(x,globals())
    #print(f"{x_name}")
    plt.savefig(f"{x_name}_{N_AP}.{N_USER}.{N_PILOT}.seed{SEED}.pdf")
    plt.savefig(f"{x_name}_{N_AP}.{N_USER}.{N_PILOT}.seed{SEED}.png")

    return

# def namestr(obj, namespace):
#     return [name for name in namespace if namespace[name] is obj] #https://stackoverflow.com/questions/592746/how-can-you-print-a-variable-name-in-python
#     #return [ k for k,v in locals().items() if v == obj][0]

if __name__=="__main__":
    main()