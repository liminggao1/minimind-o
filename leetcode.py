from typing import List

class Solution:
    def firstMissingPositive(self, nums: List[int]) -> int:
        while 1:
            for i,item in enumerate(nums):
                if item<=len(nums):
                    temp=nums[item-1]
                    nums[item-1]=item
                    nums[i]=temp
            for i,item in enumerate(nums):
                if item!=i+1:
                    return i+1 
    
if __name__ == "__main__":
    solution = Solution()
    nums = [3, 4, -1, 1]
    result = solution.firstMissingPositive(nums)
    print(result)  # Output: 2