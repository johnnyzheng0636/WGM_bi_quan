import numpy as np
import torch
import torch.nn as nn
import heapq
from tqdm import tqdm


# nn.Module for pytorch cuda inbuilt acceleration?
# removed nn.Module
# if tensor is on cuda device, acceleration will be applied
# basline
class Binarization():
    def __init__(self, weight, quan_method='xnor', group_method='abs_mag_dist', k=10, col_block_len=2):
        self.weight = weight
        self.quan_method = quan_method
        self.group_method = group_method
        self.k = k
        self.col_block_len = col_block_len
        # store the group
        # mask is a list of boolean matrix with 1 represent using, 0 represent not using.
        self.group_masks = self.grouping()

    def grouping(self):
        '''
        self.group_method == 'abs_mag_dist_auto': 
            group matrix dynamically based on gaussian distribution of magnitude of weight
            moved into dynamicGroupingQuan and greedyGroupingQuan
        self.group_method == 'abs_mag_dist_k': 
            group matrix to k groups based on gaussian distribution of magnitude of weight
        self.group_method == 'abs_mag_dist': 
            split matrix based on gaussian distribution of magnitude of weight
        self.group_method == 'hes_dist': 
            split matrix based on gaussian distribution of hessian of weight
        self.group_method == 'col_wise_block': 
            split matrix shaped(row, col) into submatrix of shape (row, col_0), where
            col_0 < col and col is divisible by col_0 
        TODO
            Add a new method, columns wise blocking, i.e. use k columns as a group for XNOR
        '''
        # group the input matrix
        # return a list of boolean mask for each group
        # assume normal distributed
        # get the most concentrate data as a group
        # since shared scale will give a low error in this case
        # we repeatly do this until remaing data
        # number is below a threshold
        masks = []
        if self.group_method == 'no':
            masks.append(torch.full(self.weight.shape, True))
        elif self.group_method == 'abs_mag_dist':
            # a simple split
            abs_w = self.weight.abs()
            abs_w_mean = abs_w.mean()
            abs_w_std = self.weight.std()
            
            check_w = torch.full(self.weight.shape, abs_w_mean)
            large_w = torch.logical_and((check_w - abs_w) < abs_w_std, (check_w + abs_w) > abs_w_std)

            masks.append(large_w)
            masks.append(~large_w)
        elif self.group_method == 'abs_mag_dist_abs_scale':
            # this is wrose than direct scale for single matrix (normal) test
            # may change for LLM wieght (magnitude normal)
            # hypothesis abd scale should be better, since scale only matter for magnitude
            # a simple split
            abs_w = self.weight.abs()
            abs_w_mean = abs_w.mean()
            abs_w_std = abs_w.std()
            
            check_w = torch.full(self.weight.shape, abs_w_mean)
            large_w = torch.logical_and((check_w - abs_w) < abs_w_std, (check_w + abs_w) > abs_w_std)

            masks.append(large_w)
            masks.append(~large_w)
        elif self.group_method == 'abs_mag_dist_k':
            # split into k group as input
            remaining = torch.full(self.weight.shape, True)
            original_w = self.weight.abs().clone()
            abs_w_std = original_w.std()

            for i in range(self.k-1):
                abs_w = original_w * remaining
                abs_w_mean = abs_w.mean()
                # using damanic std caused worse grouping
                # because middle part already removed in previous iteration
                # so most values are not in the new range
                # abs_w_std = abs_w.std()
                check_w = torch.full(self.weight.shape, abs_w_mean) * remaining
                threshold = abs_w_std/self.k
                # threshold = abs_w_std/50
                # print(threshold)
                # print(abs_w_std)
                # print(abs_w_mean)
                current_group_mask = torch.logical_and(check_w - threshold < abs_w, abs_w < check_w + threshold) & remaining
                masks.append(current_group_mask)
                remaining = remaining & ~current_group_mask
                print('')

            masks.append(remaining)

            # print(current_group_mask.sum())
            # print('='*50)
            # print(abs_w)
            # print('='*50)
            # print(current_group_mask)
            # print('='*50)
            # print(abs_w*current_group_mask)
            # print('='*50)
        elif self.group_method == 'abs_mag_dist_auto':
            # split into group with an optimization
            pass
        elif self.group_method == 'col_wise_block':
            _, cols_no = self.weight.size()
            for i in range(0, cols_no, self.col_block_len):
                row_upper_bound = min(cols_no, i + self.col_block_len)
                current_mask = torch.zeros(self.weight.size())
                current_mask[:, i:row_upper_bound] = 1
                masks.append(current_mask)
        
        # check mask is correct, i.e. there is only a unique True mask
        # for each element of weight
        mask_sum = 0
        mask_only_one = torch.full(self.weight.shape, False)
        for m in masks:
            mask_sum += m.sum()
            mask_only_one = torch.logical_xor(mask_only_one, m)
        if mask_sum == torch.numel(self.weight) and torch.all(mask_only_one == True):
            pass
        else: 
            print('mask total true count correct? ', mask_sum == torch.numel(self.weight))
            print('mask is unique for each element? ', torch.all(mask_only_one == True))
            
        return masks

    def quantization(self):
        # quantization with input quan_method
        # just try simple xnor
        
        w = torch.zeros(self.weight.shape)
        if self.quan_method == 'xnor':
            for m in self.group_masks:
                if torch.all(m == False):
                    continue
                w_group = m*self.weight
                scale = w_group.abs().sum()/m.sum()
                binary_w = torch.sign(w_group)
                # print('scale: {}, binary: {}'.format(scale, binary_w))
                w += scale * binary_w
        
        return w


# # Example usage with lambda_reg=0.5
# if __name__ == "__main__":
#     A = torch.tensor([[1.0, -2.0, 0], [0, 3.0, -4.0], [5.0, 0, 6.0]], dtype=torch.float32)
#     lambda_reg = 0.5  # Hyperparameter to tune
#     min_cost, submatrices = optimal_grouping_with_submatrices(A, lambda_reg)
#     print(f"Minimum total cost (lambda={lambda_reg}):", min_cost)
#     for idx, submatrix in enumerate(submatrices):
#         print(f"Submatrix {idx + 1}:")
#         print(submatrix.to_dense())


# normalized    
# def compute_cost(j, k, prefix_sum_abs, prefix_sum_abs_sq, lambda_reg):
#     group_size = k - j
#     if group_size <= 0:
#         return float('inf')
#     sum_abs = prefix_sum_abs[k] - prefix_sum_abs[j]
#     sum_abs_sq = prefix_sum_abs_sq[k] - prefix_sum_abs_sq[j]
#     var = (sum_abs_sq / group_size) - (sum_abs / group_size) ** 2
#     return var + lambda_reg / (group_size ** 2)  # Normalized terms

def compute_cost(j, k, prefix_sum, squared_prefix_sum, lambda_reg, lambda_nor=False):
    """Compute the cost for grouping elements from index j to k."""
    group_size = k - j

    if lambda_nor:
        n = len(prefix_sum)-1
        lambda_min = ((prefix_sum[1] - (prefix_sum[2] - prefix_sum[1])) ** 2) / (3 * n)
        abs_sum_1 = prefix_sum[k] - prefix_sum[j]
        mean_1 = abs_sum_1 / group_size
        abs_sum_2 = prefix_sum[j] - prefix_sum[j+j-k]
        mean_2 = abs_sum_2 / group_size
        lambda_max = (n * (mean_1 - mean_2) ** 2) / 12
        lambda_reg = lambda_min + lambda_reg * (lambda_max - lambda_min)

    if group_size <= 0:
        return float('inf')
    elif group_size == 1:
        mean = prefix_sum[k] - prefix_sum[j]
        # numeric stability issue if directly calculating, forcing 0
        cost = 0
    else:
        abs_sum = prefix_sum[k] - prefix_sum[j]
        abs_sum_sqr = squared_prefix_sum[k] - squared_prefix_sum[j]
        mean = abs_sum / group_size
        var = (abs_sum_sqr / group_size) - (mean ** 2)
        cost = group_size * var
    cost_reg = cost + (lambda_reg / group_size)
    # print('='*25, 'cost debug', '='*25)
    # print('cost: {},\nabs_sum: {},\nabs_sum_sqr: {},\nmean: {},\nvar: {},\ncost_reg: {}'.format(
    #     cost, abs_sum, abs_sum_sqr, mean, var, cost_reg))
    return cost, cost_reg, mean

def sort_abs(A, zero_flag=True):
    # Find all non-zero element of A    
    rows, cols = torch.nonzero(A, as_tuple=True)
    values = A[rows, cols]
    n = values.numel()
    if n == 0:
        return None
    
    # sort non-zero partition for grouping
    abs_values = values.abs()
    sorted_abs, sorted_idx = torch.sort(abs_values, stable=True, descending=False)
    sorted_binary = torch.sign(values[sorted_idx])
    sorted_rows_idx = rows[sorted_idx]
    sorted_cols_idx = cols[sorted_idx]

    return sorted_abs, sorted_rows_idx, sorted_cols_idx, sorted_binary, n
    
def prefix_sum(abs_values):
    device = abs_values.device
    n = abs_values.numel()
    prefix_sum_abs = torch.zeros(n + 1, device=device)
    prefix_sum_abs[1:] = torch.cumsum(abs_values, dim=0)
    prefix_sum_sq_abs = torch.zeros(n + 1, device=device)
    prefix_sum_sq_abs[1:] = torch.cumsum(abs_values ** 2, dim=0)
    return prefix_sum_abs, prefix_sum_sq_abs

def optimalGrouping(dp, max_groups):
    # find the optimal grouping in the dp table
    min_total_cost, g_idx = torch.min(dp[1:, -1], dim=0)
    # print(dp[1:, -1])
    # print('min_total_cost: {}, g_idx: {}'.format(min_total_cost, g_idx))
    g = g_idx.item()
    # The row with optimal cost, assume last col (using all elements)
    return g

def backpointing(backpointers, backpointers_scaler, lastsplit, n):
    # look into bp start from lastsplit for totalsplit times
    # return all split points give the rightmost split
    # Backtrack to reconstruct groups

    current_k = -1
    splits = []
    scalers = []
    for m in range(lastsplit+1, 0, -1):
        best_j = backpointers[m, current_k].item()
        splits.append(best_j)
        scalers.append(backpointers_scaler[m, current_k].item())
        current_k = best_j
    splits = splits[::-1]
    scalers = scalers[::-1]
    splits = splits + [n]
    # print('splits: {}'.format(splits))
    # print('scalers: {}'.format(scalers))

    return splits, scalers

def bi_matrix(
        shape,
        row_idx, 
        col_idx, 
        binary, 
        split_pts, 
        scaler, 
        device,
        # origin_idx, 
        # simulation=False,
    ):
    """
    Construct the binary quantized matrix

    Args:
        shape (tuple): Shape of the original matrix.
        row_idx (torch.Tensor): Row indices of the non-zero elements.
        col_idx (torch.Tensor): Column indices of the non-zero elements.
        binary (torch.Tensor): Binary tensor representing the values sign.
        split_pts (list): List of split points for the binary quantized matrix.
        scaler (torch.Tensor): Tensor representing the scaling factors for each group.
        origin_idx (torch.Tensor): Original indices of the elements in the matrix.
        simulation (bool): Flag indicating simulation quantization, i.e. return binary * scale.
    Returns:
        biMat (torch.Tensor): The binary matrix.
        scaleMat (torch.Tensor): The scaler matrix.

    TODO:
        - return only biMat for acceleration
        - for scale, only return the scaler for each group with group eles index as dict for acceleration
    """
    # construct the binary quantized matrix

    biMat = torch.zeros(shape).to(device)
    scaleMat = torch.zeros(shape).to(device)

    for i in range(len(split_pts) - 1):
        # print('='*50)
        # print(i)
        start = split_pts[i]
        end = split_pts[i+1]
        # print('start:{}, end:{}'.format(start, end))
        group_rows = row_idx[start:end]
        group_cols = col_idx[start:end]
        group_sign = binary[start:end]
        # if group_rows.numel() == 0:
        #     continue
        # indices = torch.stack([group_rows, group_cols])
        # print('indices: \n', indices)
        # print('group_sign: \n', group_sign)
        # print('scale: ', scaler[i])
        scaleMat[group_rows, group_cols] = scaler[i]
        # print('scaleMat: \n',  scaleMat)
        biMat[group_rows, group_cols] = group_sign
        # print('biMat: \n',  biMat)

    # print('='*50)
    # print('biMat: \n', biMat)
    # print('scaleMat: \n', scaleMat)
    return biMat, scaleMat


class dynamicGroupingQuan():
    """
    Percise best grouping for quantization, time complxity O(n^2logn), n is the total number of elements

    """
    def __init__(self, weight, max_groups=20, lambda_reg=0.75, lambda_nor=False):
        self.weight = weight
        self.shape = self.weight.shape
        self.device = weight.device
        self.max_groups = max_groups
        self.lambda_reg = lambda_reg
        # self.n = weight.numel()     # the total number of elements
        self.sorted_abs, self.sorted_rows_idx, self.sorted_cols_idx, self.sorted_binary, self.n = sort_abs(weight)
        self.prefix_sum_abs, self.prefix_sum_sq_abs = prefix_sum(self.sorted_abs)
        self.quantized_bi, self.quantized_scale = None, None
        self.quantized_simu = None
        self.lambda_nor = lambda_nor

    def grouping(self):
        max_groups = min(self.max_groups, self.n)

        # dynamic programming table
        print(max_groups)
        dp = torch.full((max_groups + 1, self.n + 1), float('inf'))
        dp[0] = torch.zeros((self.n + 1,))
        dp[:,0] = torch.zeros((max_groups + 1,))

        # optimal split point
        backpointers = torch.full((max_groups + 1, self.n + 1), -1, dtype=torch.long, device=self.device)

        # mean at optimal split point (for scale)
        scaler = torch.full((max_groups + 1, self.n + 1), -1, dtype=torch.float, device=self.device)

        # Fill DP table
        for i in tqdm(range(1, max_groups + 1)):  # each possible row (no of groups)
            for j in range(1, self.n + 1):  # each possible col (no of possible split)
                min_cost = float('inf')
                best_split = -1
                best_avg = -1
                
                # skip used split for left part
                # follow is wrong
                # TODO: fix it, currently skip all split used for previously
                # left part cost in dp[i-1, k], 
                # the first split used is backpointers[i-1, k]
                # the left part
                # left_split = backpointers[:i, j-1]
                # No need this since we are using the smallest cost left part
                # and current split is the right remaining part
                # left_split = backpointing(backpointers, (i, j-1), i)
                # start at no. of elements > no. of groups (row), else no place for new split
                for k in range(i-1, j):       # each possible split point
                    # if k in left_split:
                    #     continue
                    # print('row: {}, col: {}, split:{}'.format(i, j, k))
                    # print('left_eles: {}-{}, right_eles: {}-{}'.format(1, k, k+1, j))
                    # if dp[m-1, j] < float('inf'):
                    # Pass lambda_reg to compute_cost
                    _, cost, avg = compute_cost(k, j, self.prefix_sum_abs, self.prefix_sum_sq_abs, self.lambda_reg, self.lambda_nor)
                    # print('cost: {}, avg: {}'.format(cost, avg))
                    # previous lowest cost from 0 to k plus k+1 to j
                    total_cost = dp[i-1, k] + cost
                    # print('total_cost: {}, left_cost: {}, right_cost: {}'.format(total_cost, dp[i-1, k], cost))
                    if total_cost < min_cost:
                        min_cost = total_cost
                        best_split = k
                        best_avg = avg
                        # print('='*50)
                        # print('min_cost:{}, best_split: {}, best_avg: {}'.format(min_cost, k, avg))
                        # print('='*50)
                    # 1 group is the whole matrix woth no split
                    if i == 1:
                        break
                # used up all possible split, just the last one
                # No need to pad, since will not be used anymore
                # if min_cost == float('inf'):
                #     dp[i, j] = dp[i-1, j]
                #     backpointers[i, j] = backpointers[i-1, j]
                #     scaler[i, j] = scaler[i-1, j]
                #     # print('+'*50)
                #     # print('min_cost:{}, best_split: {}, best_avg: {}'.format(min_cost, best_split, best_avg))
                #     # print('+'*50)
                #     continue

                # print('+'*50)
                # print('min_cost:{}, best_split: {}, best_avg: {}'.format(min_cost, best_split, best_avg))
                # print('+'*50)
                dp[i, j] = min_cost
                backpointers[i, j] = best_split
                scaler[i, j] = best_avg
    
        # print('dp: ', dp)
        # print('backpointers: ', backpointers)
        # print('scaler: ', scaler)

        best_split_row_idx = optimalGrouping(dp, self.max_groups)
        best_split, best_scale = backpointing(backpointers, scaler, best_split_row_idx, self.n)
        self.quantized_bi, self.quantized_scale = bi_matrix(
            self.shape,
            self.sorted_rows_idx,
            self.sorted_cols_idx,
            self.sorted_binary,
            best_split,
            best_scale,
            device=self.device,
        )
        self.quantized_simu = self.quantized_bi * self.quantized_scale
        print('quantized: \n', self.quantized_simu)



# # The cost is the same as dynamic spliting
# def compute_merge_cost(i, j, sorted_values, lambda_reg):
#     # Compute cost of merging groups [i, j)
#     group_size = j - i
#     sum_abs = torch.sum(sorted_values[i:j])
#     sum_abs_sq = torch.sum(sorted_values[i:j] ** 2)
#     var = (sum_abs_sq / group_size) - (sum_abs / group_size) ** 2
#     regularization = lambda_reg / group_size
#     return group_size * var + regularization


# def approximate_grouping(A, target_groups, lambda_reg=1.0):
#     # Extract and sort non-zero elements
#     nonzero_mask = A != 0
#     nonzero_values = A[nonzero_mask]
#     if len(nonzero_values) == 0:
#         return []
#     sorted_abs, sorted_idx = torch.sort(nonzero_values.abs())
#     sorted_values = nonzero_values[sorted_idx]
#     n = len(sorted_values)
    
#     # Initialize groups and compute initial costs
#     groups = [{'start': i, 'end': i+1, 'cost': lambda_reg} for i in range(n)]
#     heap = []
    
#     # Precompute merge costs for adjacent groups
#     for i in range(n-1):
#         cost_diff = compute_merge_cost(i, i+1, sorted_values, lambda_reg)
#         heapq.heappush(heap, (cost_diff, i, i+1))
    
#     # Merge until target_groups is reached
#     current_groups = n
#     while current_groups > target_groups:
#         # Pop the best merge (lowest cost_diff)
#         cost_diff, left, right = heapq.heappop(heap)
#         if groups[left] is None or groups[right] is None:
#             continue  # Already merged
        
#         # Merge groups[left] and groups[right]
#         merged_group = {
#             'start': groups[left]['start'],
#             'end': groups[right]['end'],
#             'cost': cost_diff + groups[left]['cost'] + groups[right]['cost']
#         }
        
#         # Update adjacency and heap
#         prev_group = groups[merged_group['start'] - 1] if merged_group['start'] > 0 else None
#         next_group = groups[merged_group['end']] if merged_group['end'] < n else None
        
#         if prev_group:
#             new_cost = compute_merge_cost(prev_group['start'], merged_group['start'], sorted_values, lambda_reg)
#             heapq.heappush(heap, (new_cost, prev_group['start'], merged_group['start']))
#         if next_group:
#             new_cost = compute_merge_cost(merged_group['start'], next_group['start'], sorted_values, lambda_reg)
#             heapq.heappush(heap, (new_cost, merged_group['start'], next_group['start']))
        
#         # Mark old groups as merged
#         groups[left] = None
#         groups[right] = None
#         groups[merged_group['start']] = merged_group
#         current_groups -= 1
    
#     # Extract final groups
#     final_groups = [g for g in groups if g is not None]
#     return final_groups


def groups_from_merge_candidate(
        merge_candidate,
        mid = -1,
):
    """
    Extrct groups from the merge candidate, for n-1 candidate, output n groups,
    each interception of two consective candidate is a group, plus the two ends
    candidate minus their right and left groups. One candidate is a special case
    and need a mid point to split the group.

    Args:
        merge_candidate (list): list of dict, each dict is a candidate merge
        mid (int): when only one candidate in the list, we need the mid point to split
    """
    if mid != -1:
        return [0, mid, merge_candidate[0]['end']]
    else:
        group_bound = []
        # extract tuple of start and end of each candidate merge
        # print([(m['start'], m['end']) for m in merge_candidate])
        boundaries = sorted([(m['start'], m['end']) for m in merge_candidate])

        # print(boundaries)

        for i in range(len(boundaries)):
            # print(i)
            group_bound.append(boundaries[i][0])
        if len(boundaries)>1:
            group_bound.append(boundaries[-2][1])
        group_bound.append(boundaries[-1][1])
    return group_bound
    

# only return the scale
def compute_scale(j, k, prefix_sum):
    """Compute the scale for grouping elements from index j to k."""
    group_size = k - j
    if group_size <= 0:
        return float('inf')
    elif group_size == 1:
        mean = prefix_sum[k] - prefix_sum[j]
    else:
        abs_sum = prefix_sum[k] - prefix_sum[j]
        mean = abs_sum / group_size
    return mean


def group_scale(
        groups_boundary,
        prefix_sum,
):
    """
    Given groups boundary, output scale for each group
    """
    scaler = []
    for i in range(len(groups_boundary) - 1):
        scaler.append(compute_scale(groups_boundary[i], groups_boundary[i+1], prefix_sum))
    return scaler

def windowed_groups_initialization(n, k):
    """
    Return merge candidate given total number of element n and windwo size k
    """
    # print('='*50)
    # print('windowed_groups_initialization started')
    # groups = [
    #         {'cost': float('inf'), 
    #          'start': i, 
    #          'end': i+2,                
    #          'leftNeighbourStart': i-1, # Find left group to edit when using this maerge
    #          'rightNeighbourStart': i+1,# Find right group to edit when using this maerge 
    #         } for i in range(self.n-1)]
    groups = list(range(0, n, k))
    # print('init groups: \n', groups)
    valid_starts = groups[:-1]
    groups.append(n)
    groups.insert(0, -1)
    # print('proposed start: \n', valid_starts)
    # print('full boundary: \n', groups)

    candidate_merges = [
        {
            'cost': float('inf'), 
            'start': groups[i], 
            'end': groups[i+2],                
            'leftNeighbourStart': groups[i-1], # Find left group to edit when using this maerge
            'rightNeighbourStart': groups[i+1],# Find right group to edit when using this maerge 
        } for i in range(1, len(groups)-2)]

    # print('init candidate_merges: \n', candidate_merges)
    
    # fill in None for each index with out candidatae_merge with corresponding start
    all_merges = []
    nones = [None] * (k-1)
    for i in candidate_merges:
        all_merges.append(i)
        all_merges.extend(nones)

    if len(all_merges) < n-1:
        all_merges.extend([None] * (n - 1 - len(all_merges)))
    elif len(all_merges) > n-1:
        del all_merges[-(len(all_merges) - (n-1)):]
    
    return all_merges, valid_starts


class greedyGroupingQuan():
    """
    Approximation of dynamicGroupingQuan of O(nlogn) time complexty, n is the total number of elements

    """
    def __init__(self, weight, max_groups=20, lambda_reg=0.75, window=None, lambda_nor=False, process_bar=False):
        self.lambda_nor = lambda_nor
        self.weight = weight
        self.shape = self.weight.shape
        self.device = weight.device
        self.max_groups = max_groups
        self.lambda_reg = lambda_reg
        # print process bar for each matrix quantization
        self.process_bar = process_bar
        # self.n = weight.numel()     # the total number of elements
        # moved into sort_abs since it gives the number of non zero elements
        self.sorted_abs, self.sorted_rows_idx, self.sorted_cols_idx, self.sorted_binary, self.n = sort_abs(weight)


        if window is None:
            self.window = max(2, self.n//100)
        elif isinstance(window, float):
            if window >= 1 or window <= 0:
                self.window = self.n//100
            else:
                self.window = self.n // (1 / window)
        else:
            if window >= self.n or window <= 0:
                self.window = 2
            else:
                self.window = window

        self.prefix_sum_abs, self.prefix_sum_sq_abs = prefix_sum(self.sorted_abs)
        self.quantized_bi, self.quantized_scale = None, None
        self.quantized_simu = None
        self.quantized_scale_meta = {}

    def grouping(self, ):
        # Extract and sort non-zero elements
        # nonzero_mask = A != 0
        # nonzero_values = A[nonzero_mask]
        # if len(nonzero_values) == 0:
        #     return []
        # sorted_abs, sorted_idx = torch.sort(nonzero_values.abs())
        # sorted_values = nonzero_values[sorted_idx]
        # n = len(sorted_values)
        
        # Initialize groups and compute initial costs
        # groups is a list of dict showing current group status
        # each dict is next candidate merge group
        # start is the first element of the candidate group
        # end is the ending boundary of the candidate group
        # cost is the cost of merging the candidate group
        # 
        # And we access each candidate group by its start index
        # which is the same as its index in the list
        # 
        # initially it is all possible groups of two elements
        # after a merge, we update the groups
        # 
        # Firstly, update two new candidate groups merging the just merged group
        # it is the left and right candidate group of the just merged group
        # we update the right boundary and start of the right and left 
        # neigouring candidate group
        # 
        # Secondly, invalid the merged candidate group, by setting all element 
        # from right boundary to right boundary of the right candidate group to None
        # 
        # TODO
        # How to find right and left neighbouring candidate group?
        # add two new key to the dict?
        # or find it with algo
        # current best:
        #     - use end as right candidate and add a new key for left candidate
        #     - must use key forright candidate too, since there may right neighbour
        #            may not start just after the current candidate

        groups = [
            {'cost': float('inf'), 
             'start': i, 
             'end': i+2,                
             'leftNeighbourStart': i-1, # Find left group to edit when using this maerge
             'rightNeighbourStart': i+1,# Find right group to edit when using this maerge 
            } for i in range(self.n-1)]
        heap = []
        # for special case of only one candidate group
        # we will use this to split into two groups
        mid = -1
        
        # Precompute merge costs for adjacent groups
        # self.process_bar
        for i in (tqdm(range(self.n-1), desc="Initialization") if self.process_bar else range(self.n-1)):
            _, cost, _ = compute_cost(i, i+2, self.prefix_sum_abs, self.prefix_sum_sq_abs, self.lambda_reg, self.lambda_nor)
            groups[i]['cost'] = cost
            # groups[i]['scale'] = scale
            # heap only have cost for priority queue, left and right bound for checking existence, 
            # scale for quantized value
            heapq.heappush(heap, (cost, i, i+2))
        # print('groups: ', groups)
        # print('heap: ', heap)
        # print('groups first: ', groups[1])

        if self.process_bar:
            p_bar = tqdm(range(self.n - self.max_groups))
        # split = 0
        
        # Merge until target_groups is reached
        current_groups = self.n
        while current_groups > self.max_groups:
            # print('='*50)
            # print('current_groups: ', current_groups)
            # Pop the best merge (lowest cost), i.e. use this merge
            cost, start, end = heapq.heappop(heap)
            # print('poped: ', (cost, start, end))

            # check left and right beighbour to see if the group still exist
            # after each merge, its left and right group will be updated
            # it can be the old cnadidate merge before update
            # try:
            # no group exist with current start or
            # start exist but end mismatched
            if groups[start] is None or groups[start]['end'] != end:
                # print('poped not exist')    
                continue  # Already merged
            # except:
            #     if groups[left] is None or groups[min(right-1,self.n-2)] is None:
            #         continue  # Already merged

            # special case for current_groups == 3
            # after pop (merge), now only two group left, after processing below, only
            # one candidate group left, we can't get full list of groups boundary from
            # only one candidate group, so we set the mid point to split the group
            if current_groups == 3:
                if start == 0:
                    mid = end
                else:
                    mid = start
            
            # after merging there are two new candidate groups
            # update the left and right group of the merged group

            # left
            # print('poped exists')

            # getting left group starting bound, check for leftmost group
            # print('testing left boundary')
            # print('left neighbour start: ', groups[start]['leftNeighbourStart'])
            left_candidate_start = groups[start]['leftNeighbourStart'] if groups[start]['leftNeighbourStart'] >= 0 else None
            # print('left_candidate_start: ', left_candidate_start)
            if left_candidate_start is not None:
                # print('have left candidate')
                # calculate the new cost and scale
                _, cost_left_new, scale_left_new = compute_cost(
                    left_candidate_start, 
                    end, 
                    self.prefix_sum_abs, 
                    self.prefix_sum_sq_abs, 
                    self.lambda_reg,
                    self.lambda_nor,
                )

                # update the new group status
                groups[left_candidate_start]['end'] = end
                groups[left_candidate_start]['cost'] = cost_left_new
                if groups[start]['end'] == self.n:
                    groups[left_candidate_start]['rightNeighbourStart'] = groups[start]['rightNeighbourStart']
                # groups[left_candidate_start]['scale'] = scale_left_new

                # update heapq
                heapq.heappush(heap, (cost_left_new, left_candidate_start, end))

                # no update neighbour start since only end has been updated

                # print('after update left neighbour: \n', groups)

            # right

            # getting right group starting bound, check for rightmost group
            # print('testing right boundary')
            right_candidate_start = groups[start]['rightNeighbourStart'] if end < self.n else None
            if right_candidate_start:
                # print('have right candidate')
                # calculate the new cost and scale
                _, cost_right_new, scale_right_new = compute_cost(
                    start, 
                    groups[right_candidate_start]['end'], 
                    self.prefix_sum_abs, 
                    self.prefix_sum_sq_abs, 
                    self.lambda_reg,
                    self.lambda_nor,
                )
                # update the new group status
                groups[start]['end'] = groups[right_candidate_start]['end']
                groups[start]['cost'] = cost_right_new
                # groups[start]['scale'] = scale_right_new

                # update heapq
                heapq.heappush(heap, (cost_right_new, start, groups[right_candidate_start]['end']))

                # Update the leftNeighbourStart of the right neighbour of the
                # old right neighbour, it is now the start of just merged group, 
                # 
                # And update the rightNeighbourStart of the just merged group to the 
                # start of the right right neighbour
                # 
                # If the right neighbour of the old right neighbour exist
                # i.e. the old right neighbour is not the rightmost
                if groups[right_candidate_start]['end'] < self.n:
                    groups[groups[right_candidate_start]['rightNeighbourStart']]['leftNeighbourStart'] = start
                    groups[start]['rightNeighbourStart'] = groups[groups[right_candidate_start]['rightNeighbourStart']]['start']
                else:
                    # have right neighbour but no right right neighbour
                    groups[start]['rightNeighbourStart'] = groups[right_candidate_start]['rightNeighbourStart']
                # invalid the old right group
                # because new left use the same start and new right use the 
                # start of the merged group, so only old right group start not in used
                groups[right_candidate_start] = None

                # print('after update right neighbour: \n', groups)

            # check for merging rightmost candidate, this needs to invalid
            # the merged group only since it have no right neighbour to merge and reuse start
            if end == self.n:
                groups[start] = None

            # Decreament real group in heap
            current_groups -= 1
            # print('current grous: \n', groups)
            # print('current heap: \n', heap)
            # split += 1
            if self.process_bar:
                p_bar.update(1)
            # p_bar.refresh()

        # print('='*50)
        
        # Extract final groups

        # get corresponding scale for each group
        # print('groups: \n', groups)
        final_groups = [g for g in groups if g is not None]
        # print('final_groups: \n', final_groups)
        groups_boundaries = groups_from_merge_candidate(final_groups, mid)
        group_scalers = group_scale(groups_boundaries, self.prefix_sum_abs)
        # print('groups_boundaries: \n', groups_boundaries)
        # print('group_scalers: \n', group_scalers)
        self.quantized_bi, self.quantized_scale = bi_matrix(
            self.shape,
            self.sorted_rows_idx, 
            self.sorted_cols_idx, 
            self.sorted_binary,
            groups_boundaries,
            group_scalers,
            device=self.device,
        )
        self.quantized_simu = self.quantized_bi * self.quantized_scale
        self.quantized_scale_meta = {
            'sorted_rows_idx': self.sorted_rows_idx, 
            'sorted_cols_idx': self.sorted_cols_idx,
            'groups_boundaries': groups_boundaries,
            'group_scalers': group_scalers,
        }
        # print('quantized_simu: \n', self.quantized_simu)


    def window_grouping(self, ):
        """
        This is the window initialization version of the normal grouping.
        The normal version start with n group each sized 1, so with time
        complexity of O(nlogn). This window grouping start with n/k group
        each with size k, so with time complexity of O((n/k)log(n/k)). 
        Notice this sacrificed accuracy for speed so it is less accurate
        but faster
        """

        # Redirect to normal grouping is window is too small
        # say not enough to give all group two element at least

        # print('windowed greedy grouping')
        # print('window size: ', self.window)
        # print('input total no of element: ', self.n)
        # print('prefix sum len: ', len(self.prefix_sum_abs))
        # print('first prefix sum: ', self.prefix_sum_abs[0])
        # Just use different initialization, other are the same

        # groups structure        
        # groups = [
            # {'cost': float('inf'),      # merging cost
        #      'start': i,                # start index of the candidate group
        #      'end': i+2,                # end index of the candidate group
        #      'leftNeighbourStart': i-1, # Find left group to edit when using this maerge
        #      'rightNeighbourStart': i+1,# Find right group to edit when using this maerge 
        #     } for i in range(self.n-1)]

        groups, valid_starts = windowed_groups_initialization(self.n, self.window)
        # print('groups: \n', groups)
        # print('valid_starts: \n', valid_starts)

        # print('last candidate merge: ', groups[valid_starts[-1]])

        heap = []
        # for special case of only one candidate group
        # we will use this to split into two groups
        mid = -1
        
        # Precompute merge costs for adjacent groups
        # TODO: only calculate cost for exist group
        for i in (tqdm(valid_starts, desc="Initialization") if self.process_bar else valid_starts):
            _, cost, _ = compute_cost(
                i,                      # groups[i]['start'] 
                groups[i]['end'], 
                self.prefix_sum_abs, 
                self.prefix_sum_sq_abs, 
                self.lambda_reg,
                self.lambda_nor,
            )
            groups[i]['cost'] = cost
            heapq.heappush(
                heap, 
                (
                    cost, 
                    i,                   # groups[i]['start']
                    groups[i]['end'],
                )
            )

        # print('groups: \n', groups)
        # print('heap: \n', heap)

        if self.process_bar:
            p_bar = tqdm(range(len(valid_starts) + 1 - self.max_groups))
        
        # Merge until target_groups is reached
        current_groups = len(valid_starts) + 1
        print('starting no of grooups: ', current_groups)
        while current_groups > self.max_groups:
            cost, start, end = heapq.heappop(heap)
            if groups[start] is None or groups[start]['end'] != end:
                continue  # Already merged

            if current_groups == 3:
                if start == 0:
                    mid = end
                else:
                    mid = start
            
            left_candidate_start = groups[start]['leftNeighbourStart'] if groups[start]['leftNeighbourStart'] >= 0 else None
            if left_candidate_start is not None:
                _, cost_left_new, scale_left_new = compute_cost(
                    left_candidate_start, 
                    end, 
                    self.prefix_sum_abs, 
                    self.prefix_sum_sq_abs, 
                    self.lambda_reg,
                    self.lambda_nor,
                )

                # update the new group status
                groups[left_candidate_start]['end'] = end
                groups[left_candidate_start]['cost'] = cost_left_new
                if groups[start]['end'] == self.n:
                    groups[left_candidate_start]['rightNeighbourStart'] = groups[start]['rightNeighbourStart']

                # update heapq
                heapq.heappush(heap, (cost_left_new, left_candidate_start, end))

            # right

            # getting right group starting bound, check for rightmost group
            right_candidate_start = groups[start]['rightNeighbourStart'] if end < self.n else None
            if right_candidate_start:
                # calculate the new cost and scale
                _, cost_right_new, scale_right_new = compute_cost(
                    start, 
                    groups[right_candidate_start]['end'], 
                    self.prefix_sum_abs, 
                    self.prefix_sum_sq_abs, 
                    self.lambda_reg,
                    self.lambda_nor,
                )
                # update the new group status
                groups[start]['end'] = groups[right_candidate_start]['end']
                groups[start]['cost'] = cost_right_new

                # update heapq
                heapq.heappush(heap, (cost_right_new, start, groups[right_candidate_start]['end']))

                if groups[right_candidate_start]['end'] < self.n:
                    groups[groups[right_candidate_start]['rightNeighbourStart']]['leftNeighbourStart'] = start
                    groups[start]['rightNeighbourStart'] = groups[groups[right_candidate_start]['rightNeighbourStart']]['start']
                else:
                    # have right neighbour but no right right neighbour
                    groups[start]['rightNeighbourStart'] = groups[right_candidate_start]['rightNeighbourStart']
                groups[right_candidate_start] = None


            # check for merging rightmost candidate, this needs to invalid
            # the merged group only since it have no right neighbour to merge and reuse start
            if end == self.n:
                groups[start] = None

            # Decreament real group in heap
            current_groups -= 1
            if self.process_bar:
                p_bar.update(1)

        # Extract final groups

        # get corresponding scale for each group
        final_groups = [g for g in groups if g is not None]
        groups_boundaries = groups_from_merge_candidate(final_groups, mid)
        tot_ele = groups_boundaries[-1] - groups_boundaries[0]
        groups_len = [groups_boundaries[i+1] - groups_boundaries[i] for i in range(len(groups_boundaries)-1)]
        # avg_bit = 0
        # for i in groups_len:
        #     avg_bit += 1 + 16/i
        # avg_bit /= len(groups_len)
        print('final_groups: ', len(final_groups))
        print('self.shape: ', self.shape)
        print('tot_ele: ', tot_ele)
        print('groups_len: ', len(groups_len))
        print('self.max_groups: ', self.max_groups)
        print('current_groups: ', current_groups)
        print('valid_starts: ', len(valid_starts))
        avg_bit = (16 * current_groups + tot_ele) / tot_ele
        print('Average bit length: ', avg_bit)
        group_scalers = group_scale(groups_boundaries, self.prefix_sum_abs)
        print('group_scalers len: ', len(group_scalers))
        self.quantized_bi, self.quantized_scale = bi_matrix(
            self.shape,
            self.sorted_rows_idx, 
            self.sorted_cols_idx, 
            self.sorted_binary,
            groups_boundaries,
            group_scalers,
            device=self.device,
        )
        self.quantized_simu = self.quantized_bi * self.quantized_scale
        self.quantized_scale_meta = {
            'sorted_rows_idx': self.sorted_rows_idx, 
            'sorted_cols_idx': self.sorted_cols_idx,
            'groups_boundaries': groups_boundaries,
            'group_scalers': group_scalers,
        }
        # print('quantized_simu: \n', self.quantized_simu)
        return final_groups

class greedyAproxSplitingQuan():
    def __init__(self, weight, max_groups=20, lambda_reg=0.75, splitFactor=10):
        """
        TODO:
        Don't do this,
            Since this has more time complexity O(n(log(n)^2)) caused by
            Looking for best split for new group after split, so this is actually 
            slower than greedymerging above
        """
        pass

class Quant1Linear(nn.Module):
    def __init__(self, infeatures, outfeatures, faster=False):
        super().__init__()
        # Per-output channel parameters
        self.register_buffer('zeros', torch.zeros((outfeatures, 1)))
        self.register_buffer('scales', torch.zeros((outfeatures, 1)))
        self.register_buffer('bias', torch.zeros(outfeatures))
        
        # Packed weights: 1 element per bit, 32 elements per int32
        self.register_buffer('qweight', 
            torch.zeros((infeatures // 32, outfeatures), dtype=torch.int)
        )
        self.faster = faster

    def pack(self, linear, scales, zeros):
        # Store quantization parameters
        self.zeros = zeros * scales
        self.scales = scales.clone()
        
        if linear.bias is not None:
            self.bias = linear.bias.clone()

        # Quantize weights to 1-bit (0/1)
        intweight = torch.round((linear.weight.data + self.zeros) / self.scales)
        intweight = intweight.clamp(0, 1).to(torch.int32)
        
        # Transpose for CUDA memory layout
        intweight = intweight.t().contiguous().numpy().astype(np.uint32)
        
        # Pack 32 elements per int32
        qweight = np.zeros(
            (intweight.shape[0] // 32, intweight.shape[1]), 
            dtype=np.uint32
        )
        
        for row in range(qweight.shape[0]):
            # Pack 32 consecutive bits
            start = row * 32
            end = start + 32
            qweight[row] = np.packbits(
                intweight[start:end, :], 
                axis=0, 
                bitorder='little'
            ).view(np.uint32)

        self.qweight = torch.from_numpy(qweight.astype(np.int32))

    def forward(self, x):
        if x.shape[-1] == x.numel():
            outshape = list(x.shape)
            y = self.bias.clone()
            outshape[-1] = self.bias.numel()
            dtype = x.dtype
            
            if self.faster:
                x = x.half()
                quant_cuda.vecquant1matmul_faster(
                    x, self.qweight, y, self.scales, self.zeros
                )
            else:
                x = x.float()
                quant_cuda.vecquant1matmul(
                    x, self.qweight, y, self.scales, self.zeros
                )
            
            return y.to(dtype).reshape(outshape)
        raise ValueError('Only supports single token processing')

def make_quant1(module, names, name='', faster=False):
    if isinstance(module, Quant1Linear):
        return
    
    for attr in dir(module):
        tmp = getattr(module, attr)
        full_name = f"{name}.{attr}" if name else attr
        if full_name in names:
            new_layer = Quant1Linear(
                tmp.in_features, 
                tmp.out_features, 
                faster=faster
            )
            setattr(module, attr, new_layer)
    
    for name_child, child in module.named_children():
        make_quant1(
            child, 
            names, 
            f"{name}.{name_child}" if name else name_child, 
            faster=faster
        )