import system_setting_init as ss
import survey_propagation as sp
import numpy as np
import os
from itertools import combinations
import time

timestamp_started = time.strftime("%Y%m%d-%H%M%S")

N_USER = 21                                               # 15C5=3003
N_PILOT=int(N_USER/3)                                     # 
N_AP = [300]                                              # [100, 300]

N_REPEAT = 1
SEED_STARTED = 100

def main():
    init_users_list=get_init_users_list(N_USER, N_PILOT)
    n_comb = len(init_users_list)
    
    sum_rates = np.zeros(shape=(n_comb, N_REPEAT))
    indices_failed = []
    best_rate = 0
    worst_rate = 0
    sum_results = 0
    n_successed = 0

    seed_used = 0
    new_seed = SEED_STARTED
    save_path = make_folder(timestamp_started)

    for n_repeat in range(N_REPEAT):

        max_iter = N_USER * 22                                                                               ####  max_iter = n_user * 30  
        print("."*20 + f"n_user: {N_USER}" + "."*20)
        
        while True:
            test_tup = init_users_list[0]
            test = list(test_tup)
            ss.main(n_user=N_USER, init_n_users=test, n_pilot=N_PILOT, n_ap=N_AP[n_repeat],
                    seed=new_seed, save_path=save_path)
            (convergence_time, n_iter,
             sum_rate_test) = sp.main(N_USER, N_PILOT, max_iter,
                                               damping=np.array([0, 0]), converge_thresh=10**-3,            ####### damping factor
                                               seed=new_seed, save_path=save_path)                                  
            if sum_rate_test!=None: #sum_rate_test == sum_rate_test:
                best_rate = sum_rate_test
                worst_rate = sum_rate_test
                break
            new_seed += 1
                                                                                                      ################ 터질때마다 seed 갱신할거임
        for i, init_users in enumerate(init_users_list):
            list(init_users)
            
            ss.main(n_user=N_USER, init_n_users=init_users, n_pilot=N_PILOT, n_ap=N_AP[n_repeat],
                    seed=new_seed, save_path=save_path)
            (convergence_time, n_iter,
             sum_rates[i, n_repeat]) = sp.main(N_USER, N_PILOT, max_iter,
                                               damping=np.array([0, 0]), converge_thresh=10**-3,            ####### damping factor
                                               seed=new_seed, save_path=save_path)
            
            result_i = sum_rates[i, n_repeat]
            is_not_converged = np.isnan(result_i)
            print(f"Simul#{i} - " +
                  f"SP({convergence_time:.2f}s/{n_iter}itr): {result_i:.2f}, NaN?: {is_not_converged}"+ 
                  f"SEEDS:{new_seed}")
    
            np.save(f"sumrates_SP_{N_AP[n_repeat]}.{N_USER}.{N_PILOT}.seed{new_seed}.npy", sum_rates[:i+1, n_repeat])
            np.savetxt(f"sumrates_SP_{N_AP[n_repeat]}.{N_USER}.{N_PILOT}.seed{new_seed}.csv", sum_rates[:i+1, n_repeat], delimiter=',')

            
            if is_not_converged:
                indices_failed.append(i)
                np.savetxt(f"indices_failed_SP_{N_AP[n_repeat]}.{N_USER}.{N_PILOT}.seed{new_seed}.csv", indices_failed, delimiter=',')
            else:
                best_rate = max(result_i, best_rate)
                worst_rate = min(result_i, worst_rate)
                sum_results += result_i
                n_successed += 1
            
            print(f"Worst: {worst_rate:.3f}, Best: {best_rate:.3f}, Mean: {sum_results/n_successed:.3f}Mbits/sec")
            print(f"worst to best ratio: {100*worst_rate/best_rate:.3f}%, 진행률: {100*(i+1)/n_comb:.3f}%")
        
        seed_used=new_seed
        new_seed += 1                          ############# seed 갱신
    
    timestamp_finished = time.strftime("%Y%m%d-%H%M%S")
    print(f"COMPLETED! \n {timestamp_started}~{timestamp_finished} \n SEED:{SEED_STARTED}~{seed_used}")

def make_folder(timestamp_now):
    pathname = os.path.join("simul_outputs", f"{timestamp_now}")
    if not os.path.exists(pathname):
        os.makedirs(pathname)
    return pathname

def get_init_users_list(n_user, n_pilot):
    user_list = np.linspace(0, n_user-1, n_user, dtype=int)
    combo = combinations(user_list, n_pilot)
    return list(combo)

if __name__=="__main__":
    main()
