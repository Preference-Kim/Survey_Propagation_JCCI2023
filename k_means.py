import numpy as np
import os
from system_setting import AREA_SIZE, get_distances
from hungarian import get_sumThroughput_hung_Mbps as get_sumThroughput_km_Mbps
KP = 1000

def main(n_user, n_pilot, seed, save_path):
    np.random.seed(seed)
    user_positions = np.load(os.path.join(save_path, f"user_positions_{seed}.npy"))
    ap_positions = np.load(os.path.join(save_path, f"ap_positions_{seed}.npy"))
    beta = np.load(os.path.join(save_path, f"beta_{seed}.npy"))
    n_centroids = int(n_user / n_pilot)
    points_x = np.random.uniform(0, AREA_SIZE[0], KP)
    points_y = np.random.uniform(0, AREA_SIZE[1], KP)
    points = np.column_stack((points_x, points_y))
    centroids = np.random.choice(np.linspace(0, KP-1, KP, dtype=int),
                                 n_centroids, replace=False)
    point2ap_distances = get_distances(points, ap_positions, is_wrapped=True)

    # Centroid finder
    centroids = points[centroids]
    diff = np.ones(n_centroids)
    while(not np.all(diff < 0.001)):
        belonging_centroids = get_closest_centroid_idx(points, centroids, n_centroids)
        centroids_new = get_new_centroids(points, belonging_centroids, n_centroids)
        diff = np.abs(np.mean(centroids - centroids_new, axis=1))
        centroids = centroids_new

    # Allocate user to centroids
    allocated_users = []
    user_centroid_group = np.zeros((n_user), dtype=int)
    user2centroid_distances = get_distances(user_positions, centroids, is_wrapped=True)
    for centroid_idx in range(n_centroids):
        user2centroid_distances[:, centroid_idx]
        closest_users = get_nonOverlapping_closest_users(
            user2centroid_distances, centroid_idx, allocated_users, n_pilot)
        user_centroid_group[closest_users] = centroid_idx
        allocated_users.append(closest_users)

    # Allocate pilots to users
    allocation = np.ones(n_user, dtype=int) * 999 #placeholder
    pilot_list = np.linspace(0, n_pilot-1, n_pilot, dtype=int)
    allocation[user_centroid_group==0] = pilot_list
    for centroid_idx in range(1, n_centroids):
        centroid_belonging_users = np.argwhere(user_centroid_group==centroid_idx).reshape(-1)
        assigned_pilots = []
        for user in centroid_belonging_users:
            pos = user_positions[user]
            dist_comparison = []
            for pilot in pilot_list:
                other_pos = user_positions[allocation==pilot]
                dist_comparison.append(np.mean(np.linalg.norm(other_pos-pos, ord=2, axis=1)))
            best_pilot = get_nonOverlapping_best_pilot(dist_comparison, assigned_pilots)
            allocation[user] = best_pilot
            assigned_pilots.append(best_pilot)
    sum_throughput = get_sumThroughput_km_Mbps(allocation, n_pilot, beta)
    return sum_throughput


def get_closest_centroid_idx(points, centroids, n_centroids):
    points_tile = np.tile(np.expand_dims(points, 1), (1, n_centroids, 1))
    centroids_tile = np.tile(centroids, (KP, 1, 1))
    point2centroid_l2 = np.linalg.norm(points_tile-centroids_tile, axis=2)
    return np.argmin(point2centroid_l2, axis=1)


def get_new_centroids(points, belonging_centroids, n_centroids):
    centroids_list = np.linspace(0, n_centroids-1, n_centroids, dtype=int)
    new_centroids = np.zeros((n_centroids, 2))
    for centroid_idx in centroids_list:
        cluster_points = points[belonging_centroids==centroid_idx]
        new_centroids[centroid_idx] = np.mean(cluster_points, axis=0)
    return new_centroids


def get_nonOverlapping_closest_users(user2centroid_distances, centroid_idx, allocated_users, n_pilot):
    users = []
    for user in np.argsort(user2centroid_distances[:, centroid_idx]):
        if not np.any(allocated_users==user):
            users.append(user)
        if len(users)==n_pilot:
            return users


def get_nonOverlapping_best_pilot(dist_comparison, assigned_pilots):
    for pilot in np.argsort(dist_comparison):
        if not np.any(assigned_pilots==pilot):
            return pilot


if __name__=="__main__":
    sum_throughput = main(n_user=12, n_pilot=4,
                          seed=0, save_path="debug")
    print(f"K-means sumThroughput: {sum_throughput}Mbps")
