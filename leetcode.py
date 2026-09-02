from typing import List

class Solution:
    def decodeString(self, s: str) -> str:
        stack=[]
        alpha=""
        for i,item in enumerate(s):
            if item!="]":
                stack.append(item)
            else:#发现符号"]"
                stack1=[]
                while stack[-1] !="[":
                    stack1.append(stack.pop())
                stack.pop()#弹出"["
                #检查数字再复制
                stack2=[]
                while stack and stack[-1].isdigit():
                    stack2.append(stack.pop())
                stack.append(int("".join(reversed(stack2)))*"".join(reversed(stack1)))  # 复制
        return "".join(stack)
            
        
                
if __name__ == "__main__":
    s = "3[a]2[bc]"
    solution = Solution()
    result = solution.decodeString(s)
    print(result)  # Output: "aaabcbc"