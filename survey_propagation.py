import numpy as np
import json
import time
import os

np.set_printoptions(precision=2)
INFIN = 10**60

def main(n_user, n_pilot, max_iter, damping, converge_thresh, seed, save_path):
    # np.random.seed(seed)
    tic_readfiles = time.time()
    x, neighbor_mapping, x_j0, y, y_normalizer, occupancy = read_files(save_path, seed)
    # for idx, user_list in enumerate(x):
    #     print(f"index {idx}: user(s) {user_list}")
    # print("readfiles:", time.time()-tic_readfiles)
    dim_x = len(x)
    # tic_matrix = time.time()
    (neighbor_mapping_matrix,
     j_prime_matrix, j0_prime_matrix,
     row_window, j0_window) = preCalculate_matrices(dim_x, n_pilot,
                                                    neighbor_mapping, x_j0)
    # print("matrices calc:", time.time()-tic_matrix)
    alpha_tilde = np.zeros(shape=(max_iter, dim_x, n_pilot))
    alpha_bar = np.zeros(shape=(max_iter, dim_x, n_pilot))
    rho_tilde = np.zeros(shape=(max_iter, dim_x, n_pilot))
    rho_bar = np.zeros(shape=(max_iter, dim_x, n_pilot))
    allocation = np.zeros(shape=(max_iter, dim_x))
    tic = time.time()                                                                    
    damp_thresh = 0                                                                             ########<--- 댐핑 조정 여부(댐핑 두 계단으로 만듬=[전 녀석 댐핑, 그전 녀석 댐핑])
    closing = 0                                                                 # t_closing = [0, 0, 0]        ########<--- 댐핑 조정된 시점
    for t in range(1, max_iter):
        # tic_rho = time.time()
        rho_tilde[t], rho_bar[t] = update_rho(y, alpha_tilde[t], alpha_bar[t])
        if damping[1] == 0:     #t==1:                                                                            ##### 두 계단 만듬. t=1일때 조심
            rho_tilde[t] = damping[1]*(rho_tilde[t-1]) + (1-damping[1])*rho_tilde[t]
            rho_bar[t] = damping[1]*(rho_bar[t-1]) + (1-damping[1])* rho_bar[t]
        else:
            rho_tilde[t] = (damping[0]*rho_tilde[t-1]+damping[1]*rho_tilde[t-2]) + (1-np.sum(damping))*rho_tilde[t] ############ 두계단
            # print("rho:", time.time()-tic_rho)
            rho_bar[t] = (damping[0]*rho_bar[t-1]+damping[1]*rho_bar[t-2]) + (1-np.sum(damping))* rho_bar[t]
        if t < max_iter-1:
            # tic_alpha = time.time()
            alpha_tilde[t+1], alpha_bar[t+1] = update_alpha(
                neighbor_mapping_matrix, j_prime_matrix, j0_prime_matrix,
                row_window, j0_window, rho_tilde[t], rho_bar[t])
            if damping[1] == 0:                                                                        ##### 두 계단 만듬. t=1일때 조심
                alpha_tilde[t+1] =  damping[1]*(alpha_tilde[t]) + (1-damping[1])*alpha_tilde[t+1]
                alpha_bar[t+1] =  damping[1]*(alpha_bar[t]) + (1-damping[1])*alpha_bar[t+1]
            else:
                alpha_tilde[t+1] = damping[0]*alpha_tilde[t]+damping[1]*alpha_tilde[t-1] + (1-np.sum(damping))*alpha_tilde[t+1]    ########## 두계단
                # print("alpha:", time.time()-tic_alpha)
                alpha_bar[t+1] = (damping[0]*alpha_bar[t]+damping[1]*alpha_bar[t-1]) + (1-np.sum(damping))*alpha_bar[t+1] 
        #if save_path=="debug":                                                                                                             ############# 잠시 아래쪽에 둠
        #    allocation[t] = make_decision(x, alpha_tilde[t], alpha_bar[t], rho_tilde[t], rho_bar[t])
        #    print_allocation(x, t, allocation[t], n_user, n_pilot, occupancy)
        #    sum_throughput = get_sum_throughput_Mbps(y, allocation[t]) * y_normalizer
        #    print(f"SumThroughput: {sum_throughput:.2f} Mbps")
############## 댐핑 조정 ###################
        if 3*n_user<t and damp_thresh<3*n_user:
            if np.sum(damping)<.6:
                damping = damping + [.04, .01]
            else:
                damp_thresh += 1
        if damp_thresh == 3*n_user:
            damping = [0, 0]
        if 10*n_user<t:
            damping = [.5, 0]
        if t>200 and save_path=="debug":   #t in [1, 100, 200] or
            allocation[t] = make_decision(x, alpha_tilde[t], alpha_bar[t], rho_tilde[t], rho_bar[t])
            print_allocation(x, t, allocation[t], n_user, n_pilot, occupancy)
            sum_throughput = get_sum_throughput_Mbps(y, allocation[t]) * y_normalizer
            print(f"SumThroughput: {sum_throughput:.2f} Mbps")
            print(f"iter={t:.1f}, damping={damping}")

#        if t==n_user*10 and t_closing[1]==0:                                                ######## 댐핑 첫 조정
#            damping=[.2, .1]
#            t_closing[0]=t
#            print(f"1st adjustment at iter={t_closing[0]:.1f}")
#        is_closing = check_convergence(                                                      ######## 댐핑 두 번째 조정
#            t, alpha_tilde, alpha_bar, rho_tilde, rho_bar, 10**-2)
#        if is_closing and t_closing[1]==0:
#            damping=[.222, .111] 
#            t_closing[1] = t
#            print(f"2st adjustment at iter={t_closing[1]:.1f}, closing..")
#        is_very_closing = check_convergence(                                                  ######<--- 댐핑 세 번째 조정
#            t, alpha_tilde, alpha_bar, rho_tilde, rho_bar, 10**-3)
#        if t_closing[2]==0:                       #if is_very_closing
#            if is_very_closing or t>max_iter-100:
#                damping=[.3, .2] 
#                t_closing[2] = t
#                print(f"2st adjustment at iter={t_closing[1]:.1f}, Very closing..")
#################################            
        is_converged = check_convergence(
            t, alpha_tilde, alpha_bar, rho_tilde, rho_bar, converge_thresh)
        if is_converged:
            if closing ==0:
                print(f"closing at iter={t:.1f}...")
                closing = 1
            convergence_time = time.time() - tic
            allocation[t] = make_decision(
                x, alpha_tilde[t], alpha_bar[t], rho_tilde[t], rho_bar[t])
            is_completed = allocation_completed(x, n_user, n_pilot, allocation[t])       ############################################################ 수렴기준은 널널하게 하고 allocation 되면 끝내자
            if is_completed:
                sum_throughput = get_sum_throughput_Mbps(y, allocation[t]) * y_normalizer
                if save_path=="debug":
                    np.save(os.path.join(save_path, "msg_alpha_tilde.npy"), alpha_tilde[1:t+1, :, :])
                    np.save(os.path.join(save_path, "msg_alpha_bar.npy"), alpha_bar[1:t+1, :, :])
                    np.save(os.path.join(save_path, "msg_rho_tilde.npy"), rho_tilde[1:t+1, :, :])
                    np.save(os.path.join(save_path, "msg_rho_bar.npy"), rho_bar[1:t+1, :, :])
                return convergence_time, t, sum_throughput
    return time.time() - tic, max_iter, None


def read_files(save_path, seed):
    with open(os.path.join(save_path, f"x_neighbors_{seed}.json"), 'r') as f:
        neighbor_mapping = json.load(f)
    with open(os.path.join(save_path, f"x_j0_{seed}.json"), 'r') as f:
        x_j0 = json.load(f)
    occupancy = np.load(os.path.join(save_path, f"occupancy_{seed}.npy"))
    x = np.load(os.path.join(save_path, f"x_{seed}.npy"))
    y = np.load(os.path.join(save_path, f"y_{seed}.npy"))
    y_normalizer = np.max(y)
    y = y / y_normalizer
    return x, neighbor_mapping, x_j0, y, y_normalizer, occupancy


def preCalculate_matrices(dim_x, n_pilot, neighbor_mapping, x_j0):
    neighbor_mapping_matrix = np.zeros((dim_x, dim_x), dtype=int)
    j0_prime_matrix = np.zeros((dim_x, dim_x, dim_x), dtype=int)
    j0_window = np.zeros((dim_x, dim_x, n_pilot), dtype=int)
    for i in range(dim_x):
        neighbor_mapping_matrix[i, neighbor_mapping[i]] = 1
        j0_window[i, x_j0[i], :] = 1
        for j0 in x_j0[i]:
            j0_prime = list(set(neighbor_mapping[j0]) - set(neighbor_mapping[i]))
            j0_prime_matrix[i, j0, j0_prime] = 1
    # print(neighbor_mapping_matrix)
    j_prime_matrix = np.tile(neighbor_mapping_matrix, (dim_x, 1, 1))
    col_window = 1 - np.tile(np.expand_dims(np.eye(dim_x, dtype=int), axis=1), (1, dim_x, 1))
    j_prime_matrix[col_window==0] = 0
    row_window = 1 - np.tile(np.expand_dims(np.eye(dim_x), axis=2), n_pilot)
    return neighbor_mapping_matrix, j_prime_matrix, j0_prime_matrix, row_window, j0_window

def update_rho(y, alpha_tilde_now, alpha_bar_now):
    dim_x, n_pilot = np.shape(y)
    rho_bar_now = np.zeros(shape=(dim_x, n_pilot))
    rho_tilde_now = np.zeros(shape=(dim_x, n_pilot))
    for r in range(n_pilot):
        tic = time.time()
        alpha_tilde_except_r = np.delete(alpha_tilde_now, r, axis=1)
        alpha_bar_except_r = np.delete(alpha_bar_now, r, axis=1)
        y_except_r = np.delete(y, r, axis=1)
        rho_tilde_now[:, r] = y[:, r] + np.sum(
            alpha_tilde_except_r - alpha_bar_except_r, axis=1)
        rho_bar_now[:, r] = y[:, r] - np.max(
            y_except_r + alpha_bar_except_r, axis=1)
    return rho_tilde_now, rho_bar_now


def update_alpha(neighbor_mapping_matrix, j_prime_matrix, j0_prime_matrix,
                 row_window, j0_window, rho_tilde_now, rho_bar_now):
    dim_x, _ = np.shape(rho_tilde_now)
    rho_BarMinusTilde_n = np.minimum(rho_bar_now - rho_tilde_now, 0)
    rho_BarMinusTilde_p = np.maximum(rho_bar_now - rho_tilde_now, 0)
    rho_bar_tile = np.tile(rho_bar_now, (dim_x, 1, 1))
    rho_BarMinusTilde_p_tile = np.tile(rho_BarMinusTilde_p, (dim_x, 1, 1))
    rho_BarMinusTilde_n_tile = np.tile(rho_BarMinusTilde_n, (dim_x, 1, 1))
    j_prime_term_tile = np.matmul(j_prime_matrix, rho_BarMinusTilde_n_tile)
    j0_prime_term_tile = np.matmul(j0_prime_matrix, rho_BarMinusTilde_n_tile)
    
    term1 = np.matmul(neighbor_mapping_matrix, rho_BarMinusTilde_n)
    term2 = -rho_bar_tile + rho_BarMinusTilde_p_tile - j_prime_term_tile
    term2[row_window==0] = INFIN
    alpha_tilde_next = term1 + np.minimum(np.min(term2, axis=1), 0)
    
    term3 = -rho_bar_tile + rho_BarMinusTilde_p_tile - j0_prime_term_tile
    term3[j0_window==0] = INFIN
    alpha_bar_next = np.minimum(np.min(term3, axis=1), 0)

    return alpha_tilde_next, alpha_bar_next


def make_decision(x, alpha_tilde_now, alpha_bar_now,
                  rho_tilde_now, rho_bar_now):
    dim_x = len(x)
    dim_r = np.size(alpha_tilde_now, axis=1)
    allocation = np.zeros(dim_x)

    b_tilde = alpha_tilde_now + rho_tilde_now
    b_bar = alpha_bar_now + rho_bar_now
    for i in range(dim_x):
        if np.max(b_tilde[i, :]) > 0:
            allocation[i] = np.argmax(b_bar[i, :])
        else:
            allocation[i] = None
    for r in range(dim_r):
        if np.count_nonzero(allocation==r) == 0:
            i_argmax = np.argmax(b_tilde[:, r])
            allocation[i_argmax] = r
    return allocation


def print_allocation(x, t, current_allocation, n_user, n_pilot, occupancy):
    print("."*10 + f"t={t}" + "."*10)
    used_resource_count = 0
    used_resource_list = np.array([], dtype=int)
    assigned_user_list = np.array([], dtype=int)
    for i in range(len(x)):
        if not (np.isnan(current_allocation[i])):
            pilot_no = int(current_allocation[i])
            print(f"pilot#{pilot_no}: user {occupancy[pilot_no]} + {x[i]}(idx {i})")
            used_resource_list = np.append(used_resource_list,
                                           int(current_allocation[i]))
            assigned_user_list = np.append(assigned_user_list, x[i])
            used_resource_count += 1
    used_resource_count_unique = np.size(np.unique(used_resource_list))
    n_duplicate_resource = used_resource_count - used_resource_count_unique
    assigned_user_count_unique = np.size(np.unique(assigned_user_list))
    n_duplicate_user = np.size(assigned_user_list) - assigned_user_count_unique
    print(f"#duplicate rsc: {n_duplicate_resource} || " + 
          f"#duplicate usr: {n_duplicate_user}\n" + 
          f"#unused rsc: {n_pilot-used_resource_count} || "
          f"#unassigned usr: {n_user-n_pilot-assigned_user_count_unique}")


def check_convergence(t, alpha_tilde, alpha_bar, rho_tilde, rho_bar, converge_thresh):
    alpha_tilde_converged = (np.abs(alpha_tilde[t] - alpha_tilde[t-1]) < converge_thresh).all()
    alpha_bar_converged = (np.abs(alpha_bar[t] - alpha_bar[t-1]) < converge_thresh).all()
    rho_tilde_converged = (np.abs(rho_tilde[t] - rho_tilde[t-1]) < converge_thresh).all()
    rho_bar_converged = (np.abs(rho_bar[t] - rho_bar[t-1]) < converge_thresh).all()
    alpha_converged = alpha_tilde_converged and alpha_bar_converged
    rho_converged = rho_tilde_converged and rho_bar_converged
    return alpha_converged and rho_converged

def allocation_completed(x, n_user, n_pilot, allocation_state):                                          ##############################################
    uRC = 0
    used_resource_list = np.array([], dtype=int)
    assigned_user_list = np.array([], dtype=int)
    for i in range(len(x)):
        if not (np.isnan(allocation_state[i])):
            used_resource_list = np.append(used_resource_list,
                                           int(allocation_state[i]))
            assigned_user_list = np.append(assigned_user_list, x[i])
            uRC += 1
    uRCU = np.size(np.unique(used_resource_list))
    nDR = uRC - uRCU
    aUCU = np.size(np.unique(assigned_user_list))
    nDU = np.size(assigned_user_list) - aUCU
    nUnaU = n_user-n_pilot-aUCU
    completed = (nDR==0) and (nDU==0) and (nUnaU==0)
    return completed                                                                 #######################

def get_sum_throughput_Mbps(y, converged_allocation):
    sum_rate = 0
    for i, r in enumerate(converged_allocation):
        if not (np.isnan(converged_allocation[i])):
            sum_rate += y[i, int(r)]
    return sum_rate * 10**-6


if __name__=="__main__":
    convergence_time, n_iter, sum_rate = main(
        n_user=30, n_pilot=10, max_iter=800, damping=np.array([0, 0]),     ############# np.array 해줘야함
        converge_thresh=10**-2, seed=0 , save_path="debug")
    print(f"converged in {convergence_time:.2f}s/{n_iter}itr")
