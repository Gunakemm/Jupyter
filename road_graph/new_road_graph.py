import pandas as pd
import numpy as np
import networkx as nx

from numba import njit
from utils import calculate_d_theta_sq
from scipy.spatial import KDTree
from tqdm import tqdm
from joblib import Parallel, delayed
from collections import deque

@njit
def _select_seeds_numba(xs: np.ndarray, ys: np.ndarray, heading: np.ndarray, seed_radius: float, theta_penalty: float):
    n_points = len(xs)
    radius_sq = seed_radius ** 2

    max_centroids = n_points
    cent_xs = np.empty(max_centroids, dtype=xs.dtype)
    cent_ys = np.empty(max_centroids, dtype=ys.dtype)
    cent_heading = np.empty(max_centroids, dtype=heading.dtype)

    cent_xs[0] = xs[0]
    cent_ys[0] = ys[0]
    cent_heading[0] = heading[0]
    count = 1

    for i in range(1, n_points):
        px = xs[i]
        py = ys[i]
        p_head = heading[i]

        d_theta_sq = calculate_d_theta_sq(
            px, py, p_head, 
            cent_xs[:count], cent_ys[:count], cent_heading[:count], 
            theta_penalty
        )

        any_less = False
        for j in range(count):
            if d_theta_sq[j] < radius_sq:
                any_less = True
                break

        if any_less:
            continue

        cent_xs[count] = px
        cent_ys[count] = py
        cent_heading[count] = p_head
        count += 1

    return cent_xs[:count], cent_ys[:count], cent_heading[:count]



def select_seeds(df: pd.DataFrame, seed_radius: float = 50, theta_penalty: float = 100) -> pd.DataFrame:
        
    xs = df['x'].values
    ys = df['y'].values
    heading = df['heading'].values
    n_points = len(df)

    c_xs, c_ys, c_heading = _select_seeds_numba(
        xs, ys, heading, seed_radius, theta_penalty
    )

    centroids = pd.DataFrame({
        'x': c_xs,
        'y': c_ys,
        'heading': c_heading
    })
    print(f'Выбрано {len(centroids)} центроид из {n_points} точек.')
    return centroids


def assign_grid_cells_overlap(df: pd.DataFrame, cell_size: float, overlap: float):
    """
    Разбивает данные df на клетки размером self.cell_size и зоной пересечения self.overlap.

    Args
    ----
    df : pd.DataFrame
        Исходные данные

    Returns
    -------
    grid_dict : dict
        Словарь с ключами - координатами ячейки вида 'cellX_cellY'; значениями - точками в данных ячейках (pd.DataFrame) 
    """

    x = df['x'].values
    y = df['y'].values
    
    i_base = np.floor(x / cell_size).astype(int)
    j_base = np.floor(y / cell_size).astype(int)
    
    dx = x % cell_size
    dy = y % cell_size

    all_i = []
    all_j = []
    all_idx = []

    all_i.append(i_base)
    all_j.append(j_base)
    all_idx.append(np.arange(len(df)))

    mask_left = dx < overlap
    if np.any(mask_left):
        all_i.append(i_base[mask_left] - 1)
        all_j.append(j_base[mask_left])
        all_idx.append(np.where(mask_left)[0])
        
    mask_right = dx > (cell_size - overlap)
    if np.any(mask_right):
        all_i.append(i_base[mask_right] + 1)
        all_j.append(j_base[mask_right])
        all_idx.append(np.where(mask_right)[0])

    mask_down = dy < overlap
    if np.any(mask_down):
        all_i.append(i_base[mask_down])
        all_j.append(j_base[mask_down] - 1)
        all_idx.append(np.where(mask_down)[0])
        
    mask_up = dy > (cell_size - overlap)
    if np.any(mask_up):
        all_i.append(i_base[mask_up])
        all_j.append(j_base[mask_up] + 1)
        all_idx.append(np.where(mask_up)[0])

    mask_ld = mask_left & mask_down
    if np.any(mask_ld):
        all_i.append(i_base[mask_ld] - 1)
        all_j.append(j_base[mask_ld] - 1)
        all_idx.append(np.where(mask_ld)[0])
        
    mask_rd = mask_right & mask_down
    if np.any(mask_rd):
        all_i.append(i_base[mask_rd] + 1)
        all_j.append(j_base[mask_rd] - 1)
        all_idx.append(np.where(mask_rd)[0])
        
    mask_lu = mask_left & mask_up
    if np.any(mask_lu):
        all_i.append(i_base[mask_lu] - 1)
        all_j.append(j_base[mask_lu] + 1)
        all_idx.append(np.where(mask_lu)[0])
        
    mask_ru = mask_right & mask_up
    if np.any(mask_ru):
        all_i.append(i_base[mask_ru] + 1)
        all_j.append(j_base[mask_ru] + 1)
        all_idx.append(np.where(mask_ru)[0])

    if not all_i:
        return {}
        
    final_i = np.concatenate(all_i)
    final_j = np.concatenate(all_j)
    final_idx = np.concatenate(all_idx)
    
    assign_df = pd.DataFrame({
        'orig_idx': final_idx,
        'cell_i': final_i,
        'cell_j': final_j
    })

    assign_df['cell_key'] = assign_df['cell_i'].astype(str) + '_' + assign_df['cell_j'].astype(str)
    
    grid_dict = {}
    grouped = assign_df.groupby('cell_key')['orig_idx'].apply(list)
    
    for key, indices in grouped.items():
        cell_df = df.iloc[indices].copy()
        cell_df = cell_df.sort_values(by=['t']).reset_index(drop=True)
        grid_dict[key] = cell_df
    return grid_dict


def kmeans(df: pd.DataFrame, centroids: pd.DataFrame, seed_radius: float = 50, theta_penalty: float = 100, max_iter: int = 300, tolerance: float = 1.0):
    """
    Усреднение центроид алгоритмом K-Means

    Args:
        df:
    """
    
    points_x = df['x'].values
    points_y = df['y'].values
    points_h = df['heading'].values

    tree = KDTree(np.column_stack((points_x, points_y)))
    n_points = len(df)

    cent_x = centroids['x'].values
    cent_y = centroids['y'].values
    cent_h = centroids['heading'].values

    n_clusters = len(centroids)
    labels = np.zeros(n_points, dtype=int)

    search_radius = seed_radius * 2
    radius_sq = seed_radius ** 2

    for iteration in tqdm(range(max_iter), desc='K-Means', unit='iter'):
        min_dists = np.full(n_points, np.inf)
        for k in range(n_clusters):
            center_coords = [cent_x[k], cent_y[k]]
            close_indices = tree.query_ball_point(center_coords, r=search_radius)

            if not close_indices:
                continue

            close_indices = np.array(close_indices)

            d_theta_sq = calculate_d_theta_sq(cent_x[k], cent_y[k], cent_h[k], points_x[close_indices], points_y[close_indices], points_h[close_indices], theta_penalty)
            closer_mask = d_theta_sq < min_dists[close_indices]
            labels[close_indices[closer_mask]] = k
            min_dists[close_indices[closer_mask]] = d_theta_sq[closer_mask]

        new_center_x = np.zeros(n_clusters)
        new_center_y = np.zeros(n_clusters)
        new_center_h_sin = np.zeros(n_clusters)
        new_center_h_cos = np.zeros(n_clusters)
        counts = np.zeros(n_clusters)

        for k in range(n_clusters):
            mask = (labels == k)
            count = np.sum(mask)
            counts[k] = count

            if count == 0:
                new_center_x[k] = cent_x[k]
                new_center_y[k] = cent_y[k]
                new_center_h_sin[k] = 0.0
                new_center_h_cos[k] = 0.0
                continue

            new_center_x[k] = np.mean(points_x[mask])
            new_center_y[k] = np.mean(points_y[mask])

            cluster_headings_rad = points_h[mask]
            new_center_h_sin[k] = np.mean(np.sin(cluster_headings_rad))
            new_center_h_cos[k] = np.mean(np.cos(cluster_headings_rad))
            # sin_vals = np.sin(cluster_headings_rad)
            # cos_vals = np.cos(cluster_headings_rad)
            # new_center_h_sin[k] = np.mean(sin_vals)
            # new_center_h_cos[k] = np.mean(cos_vals)

        # new_center_h = np.arctan2(new_center_h_sin, new_center_h_cos)
        raw_heading = np.arctan2(new_center_h_sin, new_center_h_cos)
        new_center_h = np.where(raw_heading < 0, raw_heading + np.pi, raw_heading)

        shift_x = np.max(np.abs(cent_x - new_center_x))
        shift_y = np.max(np.abs(cent_y - new_center_y))

        cent_x = new_center_x
        cent_y = new_center_y
        cent_h = new_center_h

        # print(f'Iteration {iteration+1}: Max shift x={shift_x:.2f}, y={shift_y:.2f}')
        if shift_x < tolerance and shift_y < tolerance:
            break


    new_centroids = pd.DataFrame({
        'x': cent_x,
        'y': cent_y,
        'heading': cent_h
    })
    return new_centroids, labels

def split_clusters(df: pd.DataFrame, centroids: pd.DataFrame, labels: np.ndarray, hv_threshold: float = np.radians(10.)):
    new_labels = labels.copy()
    new_centroids = centroids.copy()

    queue = deque(centroids.index.to_list())
    
    while queue:
        lbl = queue.popleft()

        mask = (new_labels == lbl)
        cluster_pts = df[mask]

        if len(cluster_pts) < 5:
            continue

        cent_h = new_centroids.loc[lbl]['heading']
        points_h = cluster_pts['heading'].values

        diff = np.abs(points_h - cent_h)
        diff = np.minimum(diff, np.pi - diff)
        mean_hv = np.mean(diff)

        if mean_hv <= hv_threshold:
            continue
        
        sin_diff = np.sin(points_h - cent_h)
        group_a_mask = sin_diff >= 0
        group_b_mask = ~group_a_mask

        idx_a = cluster_pts.index[group_a_mask]
        idx_b = cluster_pts.index[group_b_mask]

        if len(idx_a) == 0 or len(idx_b) == 0:
            print(f'Количество точек в кластере: {len(cluster_pts)}, однако невозможно разделить')
            continue

        pts_a = df.loc[idx_a]
        new_centroids.loc[lbl, 'x'] = pts_a['x'].mean()
        new_centroids.loc[lbl, 'y'] = pts_a['y'].mean()
        new_centroids.loc[lbl, 'heading'] = np.arctan2(
            np.mean(np.sin(pts_a['heading'])),
            np.mean(np.cos(pts_a['heading']))
        )

        new_lbl_b = new_centroids.index.max() + 1
        pts_b = df.loc[idx_b]
        new_row = pd.DataFrame({
            'x': [pts_b['x'].mean()],
            'y': [pts_b['y'].mean()],
            'heading': [np.arctan2(
                np.mean(np.sin(pts_b['heading'])),
                np.mean(np.cos(pts_b['heading']))
            )]
        }, index=[new_lbl_b])

        new_centroids = pd.concat((new_centroids, new_row))

        new_labels[idx_b] = new_lbl_b

        queue.append(lbl)
        queue.append(new_lbl_b)
    
    print(f"Split complete. Total clusters: {len(new_centroids)}")
    return new_centroids, new_labels

def parallel_kmeans(grid_data: dict, seed_radius: float, seed_theta_penalty: float, kmeans_radius: float, 
                    kmeans_theta_penalty: float, max_iter: int = 300, tolerance: float = 1.0, hv_threshold: float = np.radians(10.0)) -> pd.DataFrame:

    def process_cell(df_cell: pd.DataFrame, cell_key: str) -> pd.DataFrame:
        if df_cell.empty:
            return None
        
        centroids = select_seeds(df_cell, seed_radius=seed_radius, theta_penalty=seed_theta_penalty)
        centroids, labels = kmeans(df=df_cell, centroids=centroids, seed_radius=kmeans_radius, theta_penalty=kmeans_theta_penalty, max_iter=max_iter, tolerance=tolerance)
        # centroids, labels = split_clusters(df=df_cell, centroids=centroids, labels=labels, hv_threshold=hv_threshold)
        cell_x_min, cell_y_min = map(lambda x: int(x) * 1000, cell_key.split('_'))
        cell_x_max, cell_y_max = cell_x_min + 1000, cell_y_min + 1000
        mask = (centroids['x'] >= cell_x_min) & (centroids['x'] <= cell_x_max) & (centroids['y'] >= cell_y_min) & (centroids['y'] <= cell_y_max)
        centroids = centroids[mask]
        centroids = centroids.add_prefix(f'{cell_key}_', axis=0)
        return centroids

    cell_keys = list(grid_data.keys())
    dfs = [grid_data[k] for k in cell_keys]

    results = Parallel(n_jobs=-1, backend='loky', verbose=10)(
        delayed(process_cell)(df, cell_key) for df, cell_key in zip(dfs, cell_keys)
    )
    valid_results = [res for res in results if res is not None]

    if valid_results:
        return pd.concat(valid_results)
    else:
        return pd.DataFrame()
    
def constrained_clustering(centroids: pd.DataFrame, seed_radius: float = 50.0, theta_penalty: float = 100, max_per_cluster: int = 1, random_state: int = 42):
    points = centroids[['x', 'y']].values
    cell_ids = centroids.index.str.split('_').str[:2].str.join('_')
    headings = centroids['heading'].values

    n = len(points)
    labels = -1 * np.ones(n, dtype=int)

    tree = KDTree(points)

    rng = np.random.default_rng(random_state)
    shuffled_indices = rng.permutation(n)

    cluster_centers = []
    current_cluster_id = 0

    sq_sps_meters = seed_radius**2

    search_radius = seed_radius * 2

    for i in shuffled_indices:
        if labels[i] != -1:
            continue

        seed_cell = cell_ids[i]
        member_indices = [i]
        cell_count = {seed_cell: 1}

        neighbors_indices = tree.query_ball_point(points[i], r=search_radius)
        neighbors_indices.remove(i)

        px = points[i][0]
        py = points[i][1]
        p_head = headings[i]

        qx = points[neighbors_indices, 0]
        qy = points[neighbors_indices, 1]
        q_head = headings[neighbors_indices]

        dists = calculate_d_theta_sq(px, py, p_head, qx, qy, q_head, theta_penalty=theta_penalty)

        zipped_data = list(zip(neighbors_indices, dists))
        zipped_data = sorted(zipped_data, key=lambda x: x[1])

        for neighbor_idx, dist in zipped_data:
            if labels[neighbor_idx] != -1:
                continue
            
            nb_cell = cell_ids[neighbor_idx]

            current_count = cell_count.get(nb_cell, 0)

            if dist > sq_sps_meters:
                break
            
            if current_count < max_per_cluster:
                member_indices.append(neighbor_idx)
                cell_count[nb_cell] = current_count + 1

        for idx in member_indices:
            labels[idx] = current_cluster_id

        cluster_points = points[member_indices]
        cluster_headings = headings[member_indices]

        center_x = np.mean(cluster_points[:, 0])
        center_y = np.mean(cluster_points[:, 1])

        # sin_sum = np.mean(np.sin(cluster_headings))
        # cos_sum = np.mean(np.cos(cluster_headings))
        # center_heading = np.arctan2(sin_sum, cos_sum)

        sin_sum = np.mean(np.sin(2 * cluster_headings))
        cos_sum = np.mean(np.cos(2 * cluster_headings))
        center_heading = (np.arctan2(sin_sum, cos_sum) / 2.0) % np.pi

        cluster_centers.append({
            'x': center_x,
            'y': center_y,
            'heading': center_heading
            # 'cluster_id': current_cluster_id,
            # 'num_points': len(member_indices),
            # 'num_cell': len(cell_count)
        })

        current_cluster_id += 1

    return pd.DataFrame(cluster_centers)

def get_labels(df: pd.DataFrame, centroids: pd.DataFrame, seed_radius: float = 50, theta_penalty: float = 100):
    
    df = df.copy()
    df = df.sort_values(['trajectory_id', 't'])

    points_x = df['x'].values
    points_y = df['y'].values
    points_h = df['heading'].values

    tree = KDTree(np.column_stack((points_x, points_y)))
    n_points = len(df)
    labels = -1 * np.ones(n_points, dtype=int)

    n_centroids = len(centroids)
    min_dists = np.full(n_points, np.inf)

    cent_x = centroids['x'].values
    cent_y = centroids['y'].values
    cent_h = centroids['heading'].values

    search_radius = seed_radius * 2

    for k in range(n_centroids):
        center_coords = [cent_x[k], cent_y[k]]
        close_indices = tree.query_ball_point(center_coords, r=search_radius)

        if not close_indices:
            continue

        close_indices = np.array(close_indices)
        d_theta_sq = calculate_d_theta_sq(cent_x[k], cent_y[k], cent_h[k], points_x[close_indices], points_y[close_indices], points_h[close_indices], theta_penalty)
        closer_mask = d_theta_sq < min_dists[close_indices]
        labels[close_indices[closer_mask]] = k
        min_dists[close_indices[closer_mask]] = d_theta_sq[closer_mask]

    return labels

def edge_inference(df: pd.DataFrame, labels: np.ndarray, min_trajectories_per_edge: int = 1):
        
    df = df.copy()
    df = df.sort_values(['trajectory_id', 't'])
    df['label'] = labels
    
    df['prev_label'] = df.groupby('trajectory_id')['label'].shift(1)
    # df['prev_trajectory_id'] = df['trajectory_id'].shift(1)
    transitions = df[df['prev_label'].notna() & (df['label'] != df['prev_label'])].copy()
    transitions['node_min'] = transitions[['prev_label', 'label']].min(axis=1)
    transitions['node_max'] = transitions[['prev_label', 'label']].max(axis=1)
    edge_counts = transitions.groupby(['node_min', 'node_max']).size().reset_index(name='count')
    edge_counts.columns = ['source', 'target', 'count']

    cluster_frequancy = df['label'].value_counts().to_dict()
    valid_edges = []
    for _, row in edge_counts.iterrows():
        u = int(row['source'])
        v = int(row['target'])
        count = int(row['count'])
        
        if u == 329 or v == 329:
            print()

        freq_u = cluster_frequancy.get(u, 0)
        freq_v = cluster_frequancy.get(v, 0)

        if freq_u == 0 or freq_v == 0:
            continue
        min_freq = min(freq_u, freq_v)
        threshold = max(min_trajectories_per_edge, np.log(min_freq) - 1)

        if count >= threshold:
        #     if (v, u) in valid_edges:
        #         valid_edges.pop((v, u))
        #     else:
            valid_edges.append({'source': u, 'target': v, 'count': count})
            

    edges = pd.DataFrame(valid_edges)

    print(f'Количество ребер перед фильтрацией: {len(edge_counts)}')
    print(f'Количество ребер после фильтрации: {len(edges)}')
    return edges

