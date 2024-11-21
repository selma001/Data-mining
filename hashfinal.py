import numpy as np
import pandas as pd

class HashTable:
    def __init__(self, size):
        self.size = size
        self.table = [[] for _ in range(size)]

    def hash_function(self, itemset):
        return sum(hash(item) for item in itemset) % self.size

    def insert(self, itemset):
        hash_value = self.hash_function(itemset)
        self.table[hash_value].append(itemset)

    def lookup(self, itemset):
        hash_value = self.hash_function(itemset)
        for element in self.table[hash_value]:
            if all(x in itemset for x in element) and all(x in element for x in itemset):
                return True
        return False



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

def prune_itemsets(itemset, candidate_itemsets, min_support, min_confidence, hash_table):
    freq_itemsets = []
    item_counts = {}

    for transaction in itemset:
        hash_table.insert(transaction)  # Insert transaction into hash table

    num_transactions = len(itemset)
    for candidate in candidate_itemsets:
        if hash_table.lookup(candidate):  # Check if candidate is frequent
            support = sum(1 for transaction in itemset if set(candidate).issubset(set(transaction))) / num_transactions

            # Check if the support meets the minimum support threshold
            if support >= min_support:
                # Calculate confidence for association rules
                for i in range(1, len(candidate)):
                    antecedent = candidate[:i]
                    consequent = candidate[i:]

                    antecedent_support = sum(1 for transaction in itemset if set(antecedent).issubset(set(transaction))) / num_transactions
                    confidence = support / antecedent_support

                    # Check if confidence meets the minimum confidence threshold
                    if confidence >= min_confidence:
                        freq_itemsets.append((candidate, support, confidence))

    return freq_itemsets

def apriori(itemset, min_support, min_conf):
    freq_itemsets = []
    k = 1

    unique_items = set([item for sublist in itemset for item in sublist])
    unique_itemsets = [[item] for item in unique_items]
    hash_table = HashTable(1000)  # Initialize hash table
    freq_itemsets.append(prune_itemsets(itemset, unique_itemsets, min_support, min_conf, hash_table))

    while freq_itemsets[-1] != []:
        candidate_itemsets = generate_candidate_itemsets(freq_itemsets[-1], k + 1, hash_table)
        freq_itemsets.append(prune_itemsets(itemset, candidate_itemsets, min_support, min_conf, hash_table))
        k += 1

    return freq_itemsets[:-1]

if __name__ == "__main__":
    df = pd.read_csv('Retail_Transactions_Dataset.csv')

    # Split the "Item" column by commas and convert it to a list of lists
    product_column = df["Product"].str.split(',').tolist()

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

    freq_itemsets = apriori(product_array, min_support, min_confidence)
    
    # Print frequent itemsets with support and confidence
    for level, itemsets in enumerate(freq_itemsets):
        print(f"Level {level + 1} Itemsets:")
        for itemset in itemsets:
            if isinstance(itemset[0], list):  # If it's an association rule
                antecedent = ', '.join(itemset[0])
                consequent = ', '.join(itemset[1])
                confidence = itemset[2]
                support = itemset[3]
                print(f"  Rule: {antecedent} => {consequent}, Support: {support}, Confidence: {confidence}")
            else:  # If it's a frequent itemset
                support = itemset[-1]
                print(f"  Itemset: {itemset}, Support: {support}")
