from typing import List


class Solution:
    def get_merges(self, corpus: str, num_merges: int) -> List[List[str]]:
        # 1. Split corpus into a list of individual characters
        # 2. For each merge step:
        #    a. Count frequency of all adjacent token pairs
        #    b. Find the most frequent pair (break ties lexicographically)
        #    c. Merge all non-overlapping occurrences left to right
        #    d. Record the merge as [token_a, token_b]
        # 3. Return the list of merges performed
        list_corpus = list(corpus)
        list_merge = []
        for i in range(num_merges):
            hashmap = defaultdict(int)
            n = len(list_corpus)
            max_pair = [0, "", ""]
            for i in range(n-1):
                hashmap[(list_corpus[i], list_corpus[i+1])] += 1
                
            for (a, b), count in hashmap.items():
                if count > max_pair[0] or (count == max_pair[0] and (a, b) < (max_pair[1], max_pair[2])):
                    max_pair = [count, a, b]
            list_corpus_cur = []
            i = 0
            list_merge.append([max_pair[1], max_pair[2]])
            while i < n :
                if i + 1 < n and list_corpus[i] == max_pair[1] and list_corpus[i+1] == max_pair[2]:
                    list_corpus_cur.append(list_corpus[i] + list_corpus[i+1])
                    i += 1
                else:
                    list_corpus_cur.append(list_corpus[i])
                i += 1
            list_corpus = list_corpus_cur
        return list_merge
        


                
            


