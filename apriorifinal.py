import numpy as np
import pandas as pd
import time

class HashTable:
    def __init__(self, size):
        self.size = size
        self.table = [[] for _ in range(size)]

    def hash_function(self, itemset):
        return sum(ord(item) for item in ','.join(itemset)) % self.size

    def insert(self, itemset):
        hash_value = self.hash_function(itemset)
        self.table[hash_value].append(itemset)

    def lookup(self, itemset):
        hash_value = self.hash_function(itemset)
        return itemset in self.table[hash_value]

def has_infrequent_subset(itemset, prev_itemsets, hash_table):
    for item in itemset:
        subset = itemset.copy()
        subset.remove(item)
        if not hash_table.lookup(subset):
            return True
    return False

def generate_candidate_itemsets(itemset, k, hash_table):
    candidate_itemsets = []
    for i in range(len(itemset)):
        for j in range(i + 1, len(itemset)):
            if itemset[i][:-1] == itemset[j][:-1]:
                candidate = sorted(set(itemset[i]) | set(itemset[j]))
                if len(candidate) == k and '' not in candidate:
                    if not has_infrequent_subset(candidate, itemset, hash_table):
                        candidate_itemsets.append(candidate)
                        hash_table.insert(candidate)
    return candidate_itemsets

def prune_itemsets(itemset, candidate_itemsets, min_support, min_confidence):
    freq_itemsets = []
    item_counts = {}

    for transaction in itemset:
        for candidate in candidate_itemsets:
            if set(candidate).issubset(set(transaction)):
                item_counts[str(candidate)] = item_counts.get(str(candidate), 0) + 1

    num_transactions = len(itemset)
    for candidate in candidate_itemsets:
        support = item_counts.get(str(candidate), 0) / num_transactions
        if support >= min_support:
            freq_itemsets.append(candidate) 
            for i in range(1, len(candidate)):
                antecedent = candidate[:i]
                consequent = candidate[i:]

                antecedent_support = item_counts.get(str(antecedent), 0) / num_transactions
                confidence = support / antecedent_support
                if confidence >= min_confidence:
                    freq_itemsets.append([antecedent, consequent, support, confidence])

    return freq_itemsets

def apriori(itemset, min_support, min_conf):
    freq_itemsets = []
    k = 1

    unique_items = set([item for sublist in itemset for item in sublist])
    unique_itemsets = [[item] for item in unique_items]
    hash_table = HashTable(1000)  # Initialize hash table
    freq_itemsets.append(prune_itemsets(itemset, unique_itemsets, min_support, min_conf))

    while freq_itemsets[-1] != []:
        candidate_itemsets = generate_candidate_itemsets(freq_itemsets[-1], k + 1, hash_table)
        freq_itemsets.append(prune_itemsets(itemset, candidate_itemsets, min_support, min_conf))
        k += 1

    return freq_itemsets[:-1]

if __name__ == "__main__":
    df = pd.read_csv('bread-basket.csv')

    # Split the "Item" column by commas and convert it to a list of lists
    product_column = df["Item"].str.split(',').tolist()

    # Pad each transaction to have the same length using a placeholder value (e.g., 'None')
    max_length = max(len(transaction) for transaction in product_column)
    padded_product_column = [transaction + [''] * (max_length - len(transaction)) for transaction in product_column]

    # Convert the padded_product_column to a NumPy array
    product_array = np.array(padded_product_column)

    # Initialize hash table
    table_size = 1000  # Adjust the size based on dataset characteristics
    hash_table = HashTable(table_size)

    min_support = 0.01
    min_confidence = 0.01

    # Mesurer le temps d'exécution
    start_time = time.time()
    freq_itemsets = apriori(product_array, min_support, min_confidence)
    end_time = time.time()

    execution_time = end_time - start_time
    print("Temps d'exécution de l'algorithme Apriori:", execution_time, "secondes")
    
    # Print frequent itemsets with support and confidence
    for level, itemsets in enumerate(freq_itemsets):
        print(f"Level {level + 1} Itemsets:")
        for itemset in itemsets:
            if isinstance(itemset[0], list):  # If it's an association rule
                antecedent = ', '.join(itemset[0])
                consequent = ', '.join(itemset[1])
                support = itemset[2]
                confidence = itemset[3]
                print(f"  Rule: {antecedent} => {consequent}, Support: {support}, Confidence: {confidence}")
            else:  # If it's a frequent itemset
                support = itemset[-1]
                print(f"  Itemset: {itemset}")
