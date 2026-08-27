from typing import List

class Solution:
    def merge(self, intervals: List[List[int]]) -> List[List[int]]:
        intervals.sort(key=lambda x:x[0])
        res=[]
        i=0
        cur=intervals[i]
        nxt=intervals[i+1]
        while i+1<len(intervals):
            nxt=intervals[i+1]
            if cur[1]>=nxt[0]:
                cur=[cur[0],max(cur[1],nxt[1])]
            else:
                res.append(cur)
                cur=nxt
            i+=1
        return res
    
if __name__ == "__main__":
    sol = Solution()
    intervals = [[1,3],[2,6],[8,10],[15,18]]
    print(sol.merge(intervals))  # Output: [[1,6],[8,10],[15,18]]