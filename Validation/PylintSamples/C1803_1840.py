# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def longestZigZag(self, root: TreeNode) -> int:
        if root == None:
            return None
        
        maxlength = 0
        stack = collections.deque()
        if root.left:
            stack.append((1, 1, root.left))
        if root.right:
            stack.append((1, 0, root.right))
        while stack:
            length, isleft, node = stack.pop()
            if isleft:
                if node.right:
                    stack.append((length + 1, 0, node.right))
                else:
                    maxlength = max(maxlength, length)
                if node.left:
                    stack.append((1, 1, node.left))
            else:
                if node.left:
                    stack.append((length + 1, 1, node.left))
                else:
                    maxlength = max(maxlength, length)
  
                if node.right:
                    stack.append((1, 0, node.right))
            
        return maxlength
            

# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def longestZigZag(self, root: TreeNode) -> int:
        stack = [(root, 0, 'left')]
        res = 0
        while stack:
            node, length, d = stack.pop()
            res = max(res, length)
            if node.left:
                if d != 'left':
                    stack.append((node.left, length + 1, 'left'))
                else:
                    stack.append((node.left, 1, 'left'))

            if node.right:
                if d != 'right':
                    stack.append((node.right, length + 1, 'right'))
                else:
                    stack.append((node.right, 1, 'right'))
        return res

# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
from collections import deque
class Solution:
    def longestZigZag(self, root: TreeNode) -> int:
        # dequeu node : [curr_node, state, longestdistance]
        ans = 0
        q = deque([[root, 'A', 0]])
        while (len(q) != 0):
            top, state, dist = q[0]
            ans = max(ans, dist)
            q.popleft()
            if state == 'A':
                if top.left:
                    q.append([top.left, 'L', 1])
                if top.right:
                    q.append([top.right, 'R', 1])
            else:
                if state == 'L':
                    if top.left:
                        q.append([top.left, 'L', 1])
                    if top.right:
                        q.append([top.right, 'R', dist+1])
                if state == 'R':
                    if top.left:
                        q.append([top.left, 'L', dist+1])
                    if top.right:
                        q.append([top.right, 'R', 1])
        return ans

# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def longestZigZag(self, root: TreeNode) -> int:
        def dfs(node):
            if not node:
                return [-1, -1, -1]
            left = dfs(node.left)
            right = dfs(node.right)
            return [left[1]+1, right[0]+1, max(left[1]+1, right[0]+1, left[2], right[2])]
        return dfs(root)[-1]
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def longestZigZag(self, root: TreeNode) -> int:
        
        maxlen = 0
        
        dt = collections.defaultdict(lambda : [0,0])
        
        def dfs(root,dt):
            nonlocal maxlen
            
            if root.left:
                dfs(root.left,dt)
                dt[root][0] = dt[root.left][1] + 1
            else:
                dt[root][0] = 0
                
            if root.right:
                dfs(root.right,dt)
                dt[root][1] = dt[root.right][0] + 1
            else:
                dt[root][1] = 0
            
            maxlen = max(maxlen, dt[root][0], dt[root][1])
            
        dfs(root,dt)
        
        return maxlen
        
                

# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def helper(self, node: TreeNode, lastDirection: str, cache) -> int:
        if not node:
            return 0
        
        if (node, lastDirection) in cache:
            return cache[(node, lastDirection)]
        
        count = 1
        childCount = float('-inf')
        if lastDirection == 'right':
            childCount = max(childCount, self.helper(node.left, 'left', cache))
        else:
            childCount = max(childCount, self.helper(node.right, 'right', cache))
        
        count += childCount
        cache[(node, lastDirection)] = count
        return count
    
    def longestZigZag(self, root: TreeNode) -> int:
        maxCount = float('-inf')
        cache = {}
        stack = [root]
        while stack:
            node = stack.pop()
            maxCount = max(maxCount, self.helper(node, 'left', cache))
            maxCount = max(maxCount, self.helper(node, 'right', cache))
            
            if node.left:
                stack.append(node.left)
            if node.right:
                stack.append(node.right)
        return maxCount - 1
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def longestZigZag(self, root: TreeNode) -> int:
        self.ans = 0
        memo = {}
        
        def dfs2(node, direction):
            if not node:
                return 0
            
            if (node, direction) not in memo:
                if direction == False:
                    memo[(node, direction)] = 1 + dfs2(node.right, True)
                else:
                    memo[(node, direction)] = 1 + dfs2(node.left, False)
            
            return memo[(node, direction)]
        
        def dfs1(node):
            if not node:
                return
            
            self.ans = max(self.ans, dfs2(node, True) - 1, dfs2(node, False) - 1)
            
            dfs1(node.left)
            dfs1(node.right)
        
        dfs1(root)
        return self.ans
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def longestZigZag(self, root: TreeNode) -> int:
        if not root:
            return 0
        max_zig_zag = 0
        
        def dfs(node):
            if not node:
                return (0, 0)
            
            left = dfs(node.left)
            right = dfs(node.right)
            
            nonlocal max_zig_zag
            max_zig_zag = max(1 + left[1], 1 + right[0], max_zig_zag)
            
            return (1 + left[1], 1 + right[0])
    
        dfs(root)
        return max_zig_zag - 1
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def longestZigZag(self, root: TreeNode) -> int:
        self.res = 0;
        if root.left:
            self.dfs(root.left, 1, True, False)
        if root.right:
            self.dfs(root.right, 1, False, True)
        return self.res
        
    def dfs(self, node, count, prevL, prevR):
        if not node:
            return
        self.res = max(self.res, count)

        if prevL:
            self.dfs(node.left, 1, True, False)
            self.dfs(node.right, count + 1, False, True)
        if prevR:
            self.dfs(node.left, count + 1, True, False)
            self.dfs(node.right, 1, False, True)

# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def longestZigZag(self, root: TreeNode) -> int:
        self.res = float(-inf)
        def helper(root, left, right):
            self.res = max(self.res, max(left, right))
            if root == None:
                return
            if root.left:
                helper(root.left, right + 1, 0)
            if root.right:
                helper(root.right, 0, left + 1)
            return
    
        helper(root, 0, 0)
        return self.res
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def longestZigZag(self, root: TreeNode) -> int:
        ans = [-1]
        
        def aux(root, isleft, ans):
            if not root:
                return -1
            left = aux(root.left, 1, ans) + 1
            right = aux(root.right, 0, ans) + 1
            ans[0] = max(ans[0], left, right)
            if isleft:
                return right
            else:
                return left
            
        if not root:
            return 0
        aux(root, 0, ans)
        return ans[0]
        
#         def aux(root, isleft, memo):
#             if not root:
#                 return 0
#             if root in memo:
#                 if isleft:
#                     return memo[root][1]
#                 else:
#                     return memo[root][0]
#             memo[root] = [0, 0]
#             memo[root][0] = aux(root.left, 1, memo) + 1
#             memo[root][1] = aux(root.right, 0, memo) + 1
#             self.ans = max(self.ans, memo[root][1], memo[root][0])
#             if isleft:
#                 return memo[root][1]
#             else:
#                 return memo[root][0]
            
#         if not root:
#             return 0
#         memo = {}
#         aux(root, 0, memo)
#         return self.ans - 1

# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def helper(self,node,prenode,ans):
        if not node:return
        self.res=max(self.res,ans)
        if node==prenode.left:
            self.helper(node.right,node,ans+1)
            self.helper(node.left,node,1)
        elif node==prenode.right:
            self.helper(node.left,node,ans+1)
            self.helper(node.right,node,1)
            
    def longestZigZag(self, root: TreeNode) -> int:
        self.res=0
        self.helper(root.left,root,1)
        self.helper(root.right,root,1)
        return self.res
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def longestZigZag(self, root: TreeNode) -> int:
        def zigzag(node: TreeNode) -> tuple:
            if not node: return 0, 0
            _, lr = zigzag(node.left)
            rl, _ = zigzag(node.right)
            self.max_path = max(self.max_path, lr + 1, rl + 1)
            return lr + 1, rl + 1

        self.max_path = 0
        zigzag(root)
        return self.max_path - 1
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def longestZigZag(self, root: TreeNode) -> int:
        if root == None:
            return 0
        maxZigZag = 0
        def zigZagStart(root):##direction will correspond to -1 for left and 1 for right
            nonlocal maxZigZag
            if root == None or (root.left == None and root.right == None):
                return [0,0]
            ll,lr = zigZagStart(root.left)
            rl,rr = zigZagStart(root.right)
            bestLeft = 0
            bestRight = 0
            if root.left:
                bestLeft = 1+lr
            if root.right:
                bestRight = 1+rl
            maxZigZag = max(maxZigZag,bestLeft,bestRight)
            return [bestLeft,bestRight]
        zigZagStart(root)
        return maxZigZag
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def longestZigZag(self, root: TreeNode) -> int:
# recursive
# use a direction to mark curr direction
# if dir is right: call 1+ helper(root.left, left)
        self.longest = 0
        def dfs(node, dirc):
            if not node:return 0
            left = dfs(node.left, "left")
            right = dfs(node.right, "right")
            self.longest = max(self.longest, left, right)
            if dirc == "right":
                return 1 + left
            else:
                return 1 + right

        dfs(root,"left")
        return self.longest
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def longestZigZag(self, root: TreeNode) -> int:
        def ZigZag(node): # -> [left_len, right_len, max_len]
            if not node:
                return [-1, -1, -1]
            left = ZigZag(node.left)
            right = ZigZag(node.right)
            return [left[1] + 1, right[0] + 1, max(left[1] + 1, right[0] + 1, left[2], right[2])]
        return ZigZag(root)[-1]

# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def longestZigZag(self, root: TreeNode) -> int:
        self.longest = 0
        self.dfs(root, 0, 0)
        return self.longest
        
    def dfs(self, node, longest_left, longest_right):
        self.longest = max(self.longest, longest_left, longest_right)
        if node.left:
            self.dfs(node.left, longest_right+1, 0)
        if node.right:
            self.dfs(node.right, 0, longest_left+1)

# 1372. Longest ZigZag Path in a Binary Tree

# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right

class Solution:
    def longestZigZag(self, root: TreeNode) -> int:
        
        def dfs(root):
            if not root: 
                # [left, right, res]
                return [-1, -1, -1]
            left, right = dfs(root.left), dfs(root.right)
            # [left.right + 1, right.left + 1, max(left.right + 1, right.left + 1, left.res, right.res)]
            return [left[1] + 1, right[0] + 1, max(left[1] + 1, right[0] + 1, left[2], right[2])]
        
        return dfs(root)[-1]
    
# Explanation
# Recursive return [left, right, result], where:
# left is the maximum length in direction of root.left
# right is the maximum length in direction of root.right
# result is the maximum length in the whole sub tree.


# Complexity
# Time O(N)
# Space O(height)

# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def longestZigZag(self, root: TreeNode) -> int:
        if not root: return
        root.val = 0
        
        result = [0]
        self.zigZagHelper(root.left, "L", root.val+1, result)
        self.zigZagHelper(root.right, "R", root.val+1, result)
        return result[0]
        
    def zigZagHelper(self, root, prev_dir, prev_val, result):
        if not root: 
            return
        
        root.val = prev_val
        result[0] = max(result[0], root.val)
        
        if prev_dir == "L":
            self.zigZagHelper(root.right, "R", root.val + 1, result)
            self.zigZagHelper(root.left, "L", 1, result)
        else:
            self.zigZagHelper(root.right, "R", 1, result)
            self.zigZagHelper(root.left, "L", root.val + 1, result)
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def longestZigZag(self, root):
        def dfs(root):
            if not root: return [-1, -1, -1]
            left, right = dfs(root.left), dfs(root.right)
            return [left[1] + 1, right[0] + 1, max(left[1] + 1, right[0] + 1, left[2], right[2])]
        return dfs(root)[-1]
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def longestZigZag(self, root: TreeNode) -> int:
        # Recursive return [left, right, result], where:
        # left is the maximum length in direction of root.left
        # right is the maximum length in direction of root.right
        # result is the maximum length in the whole sub tree
        def dfs(root):
            if not root:
                return [-1, -1, -1]
            left, right = dfs(root.left), dfs(root.right)
            return [left[1] + 1, right[0] + 1, max(left[1] + 1, right[0] + 1, left[2], right[2])]
        return dfs(root)[2]

# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def longestZigZag(self, root: TreeNode) -> int:
        _, max_depth = self.zigzag(root)
        return max_depth
    
    def zigzag(self, node: TreeNode, return_left=False) -> int:
        if node is None:
            return -1, 0
        left_depth, left_max = self.zigzag(node.left)
        right_depth, right_max = self.zigzag(node.right, return_left=True)

        left_depth += 1
        right_depth += 1
        max_depth = max(left_depth, right_depth, left_max, right_max)
        return left_depth if return_left else right_depth, max_depth
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def longestZigZag(self, root: TreeNode) -> int:
        
        def aux(root):
            if not root:
                return [-1, -1, -1]
            left = aux(root.left)
            right = aux(root.right)
            ans = max(max(left[1], right[0])+1, left[2], right[2])
            return [left[1]+1, right[0]+1, ans]
        
        return aux(root)[2]
        
        
        
        
#         ans = [-1]
        
#         def aux(root, isleft, ans):
#             if not root:
#                 return -1
#             left = aux(root.left, 1, ans) + 1
#             right = aux(root.right, 0, ans) + 1
#             ans[0] = max(ans[0], left, right)
#             if isleft:
#                 return right
#             else:
#                 return left
            
#         if not root:
#             return 0
#         aux(root, 0, ans)
#         return ans[0]
        
#         def aux(root, isleft, memo):
#             if not root:
#                 return 0
#             if root in memo:
#                 if isleft:
#                     return memo[root][1]
#                 else:
#                     return memo[root][0]
#             memo[root] = [0, 0]
#             memo[root][0] = aux(root.left, 1, memo) + 1
#             memo[root][1] = aux(root.right, 0, memo) + 1
#             self.ans = max(self.ans, memo[root][1], memo[root][0])
#             if isleft:
#                 return memo[root][1]
#             else:
#                 return memo[root][0]
            
#         if not root:
#             return 0
#         memo = {}
#         aux(root, 0, memo)
#         return self.ans - 1

# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    m = 0
    def longestZigZag(self, root: TreeNode) -> int:
        def longest(node: TreeNode, state: int, acc: int) -> int:
            self.m = max(self.m, acc)
            if not node:
                return 0
            if state == 0:
                longest(node.right, 1, acc + 1), longest(node.left, 0, 0)
            else:
                longest(node.left, 0, acc + 1), longest(node.right, 1, 0)
            
        longest(root.left, 0, 0), longest(root.right, 1, 0)
        return self.m
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def longestZigZag(self, root: TreeNode) -> int:
        self.max_zigzag = 0
        
        def dfs(node):
            if not node:
                # [left, right, max]
                return [-1, -1, -1]
            left = dfs(node.left)
            right = dfs(node.right)
            return [left[1] + 1, right[0] + 1, max(left[2], right[2], left[1]+1, right[0]+1)]
        

        return dfs(root)[2]
    
        

class Solution:
        
    def longestZigZag(self, root: TreeNode) -> int:
        if not root: return 0
        q = deque()
        max_depth = 0
        if root.left: q.append((root.left, True, 1))
        if root.right: q.append((root.right, False, 1))
        
        while q: 
            n, is_left, depth = q.popleft()
            max_depth = max(depth, max_depth)
            if n.left:
                if is_left: 
                    q.append((n.left, True, 1))
                else: 
                    q.append((n.left, True, depth+1))
            if n.right: 
                if is_left: 
                    q.append((n.right, False, depth+1))
                else:
                    q.append((n.right, False, 1))
                    

        return max_depth
        

# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def longestZigZag(self, root: TreeNode) -> int:
        
        q = collections.deque()
        res = 0
        q.append((root, 0, 0))
        
        while q:
            size = len(q)
            for _ in range(size):
                node, l, r = q.popleft()
                if node.left:
                    q.append((node.left, r+1, 0))
                    res = max(res, r+1)
                if node.right:
                    q.append((node.right, 0, l+1))
                    res = max(res, l+1)
        
        return res
                             
                            
        
                             
                          


# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def longestZigZag(self, root: TreeNode) -> int:
        *_, max_depth = self.zigzag(root)
        return max_depth
    
    def zigzag(self, node: TreeNode) -> int:
        if node is None:
            return -1, -1, 0
        _, left_depth, left_max = self.zigzag(node.left)
        right_depth, _, right_max = self.zigzag(node.right)

        left_depth += 1
        right_depth += 1
        max_depth = max(left_depth, right_depth, left_max, right_max)
        return left_depth, right_depth, max_depth
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def longestZigZag(self, root: TreeNode) -> int:
        def zigzag(node,left,acum):
            if node is None:
                return acum - 1
            if left:
                return max(zigzag(node.left,False,acum + 1), zigzag(node.left,True,0))
            else:
                return max(zigzag(node.right,True,acum + 1), zigzag(node.right,False,0))
        return max(zigzag(root,True,0),zigzag(root,False,0))
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def longestZigZag(self, root: TreeNode) -> int:
        
        self.max_len = 0

        def subTree(root, indir, flip):
            if root is None:
                return 0
            if indir == -1:
                subcount = subTree(root.right, 1, True) 
                self.max_len = max(subcount + 1, self.max_len)
            else:
                subcount = subTree(root.left, -1, True)
                self.max_len = max(subcount + 1, self.max_len)

            if flip:
                subTree(root, -indir, False)
            return subcount + 1

        subTree(root, 1, True)
        return self.max_len - 1
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def longestZigZag(self, root: TreeNode) -> int:
        l=[[root,0]]
        r=[[root,0]]
        m=0
        while(True):
            if(l==[])and(r==[]):
                return m
            l1=[]
            r1=[]
            for i in l:
                m=max(m,i[1])
                if(i[0].right!=None):
                    r1.append([i[0].right,i[1]+1])
                if(i[0].left!=None):
                    r1.append([i[0].left,0])
            for i in r:
                m=max(m,i[1])
                if(i[0].left!=None):
                    l1.append([i[0].left,i[1]+1])
                if(i[0].right!=None):
                    l1.append([i[0].right,0])
            r=r1
            l=l1
            
            

# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def longestZigZag(self, root: TreeNode) -> int:
        
        
        def help(root):
            if not root:
                return [-1, -1, -1]
            left, right = help(root.left), help(root.right)
            return [ left[1]+1, right[0]+1, max(left[-1], right[-1], left[1]+1, right[0]+1)  ]
        
        return help(root)[-1]
        
            

# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def longestZigZag(self, root: TreeNode) -> int:
        def helper(root):
            stack = [[root, True, 0], [root, False, 0]]
            ans = 0
            while stack:
                root, right, length = stack.pop()
                if root:
                    ans = max(length, ans)
                    stack.append((root.right if right else root.left, not right, length +1))
                    stack.append((root.left if right else root.right, right, 1))    
            return ans
        return helper(root)
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def __init__(self):
        self.max_length = float('-inf')
    
    def longestZigZag(self, root):
        return self.dfs(root)[2] - 1
    
    def dfs(self, root):
        if root is None: return [0, 0, 0]
        
        left_res = self.dfs(root.left)
        right_res = self.dfs(root.right)
        
        maxForSubtree = max(left_res[1], right_res[0]) + 1
        return [left_res[1] + 1, right_res[0] + 1, max(maxForSubtree, left_res[2], right_res[2])]    
    
#     def __init__(self):
#         self.max_length = float('-inf')
#         self.memo = {}

#     def longestZigZag(self, root: TreeNode) -> int:
#         self.dfs(root)
#         return self.max_length if self.max_length != float('-inf') else 0
# # O(n) time, O(n) space
    
#     def dfs(self, root):
#         if root is None: return

#         self.dfs(root.left)
#         self.dfs(root.right)
        
#         left_res = self.zigzag(root, True)
#         right_res = self.zigzag(root, False)
#         self.max_length = max(self.max_length, left_res - 1, right_res - 1)        
        
#     def zigzag(self, node, is_left):
#         if id(node) in self.memo: 
#             if is_left and self.memo[id(node)][0]:
#                 return self.memo[id(node)][0]                    
#             elif is_left is False and self.memo[id(node)][1]:
#                 return self.memo[id(node)][1]
#         if node is None:
#             return 0

#         res = 0
#         if is_left:
#             res += self.zigzag(node.left, False)
#         else:
#             res += self.zigzag(node.right, True)

#         length = res + 1            
#         memoized_res = self.memo.get(id(node), [None, None])
#         memoized_res[(0 if is_left else 1)] = length
#         self.memo[id(node)] = memoized_res
#         return res + 1        

# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def longestZigZagRec(self, root, goLeft, steps):
        if root == None:
            return steps - 1
        if goLeft:
            return max(
            self.longestZigZagRec(root.left, False, steps + 1),
            self.longestZigZagRec(root.right, True, 1)
            )
        
        else:
            return max(
            self.longestZigZagRec(root.right, True, steps + 1),
            self.longestZigZagRec(root.left, False, 1)
            )
            
    def longestZigZag(self, root: TreeNode) -> int:
        if root == None:
            return 0
        
        return max(
            self.longestZigZagRec(root, True, 0),
            self.longestZigZagRec(root, False, 0)
        ) 
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def longestZigZag(self, root: TreeNode) -> int:
        self.max_zigzag = 0
        def dfs(node, is_left, streak):
            if not node:
                return
            self.max_zigzag = max(self.max_zigzag, streak)
            
            if is_left:
                dfs(node.right, not is_left, streak + 1)
                dfs(node.right, is_left, 0)
            else:
                dfs(node.left, not is_left, streak + 1)
                dfs(node.left, is_left, 0)
        
        dfs(root, True, 0)
        dfs(root, False, 0)
        return self.max_zigzag
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def longestZigZag(self, root: TreeNode) -> int:
        if root is None:
            return 0
        self.helper(root)
        l = [root]
        m = 0
        while len(l) != 0:
            node = l.pop()
            if node.left != None:
                l.append(node.left)
            if node.right != None:
                l.append(node.right)
            if max(node.val) > m:
                m = max(node.val)
        return m
    def helper(self,root):
        if root is None:
            return 0
        self.helper(root.left)
        self.helper(root.right)
        if root.left == None and root.right == None:
            root.val = (0,0)
        elif root.left != None and root.right == None:
            root.val = (root.left.val[1] + 1, 0)
        elif root.right != None and root.left == None:
            root.val = (0,root.right.val[0] + 1)
        else:
            root.val = (root.left.val[1] + 1, root.right.val[0] + 1)
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def longestZigZag(self, root: TreeNode) -> int:
        
        def Search(root,left,height):
            if root is None:
                return height
            if left:
                return max(Search(root.left,1,0),Search(root.right,0,height+1))
            else:
                return max(Search(root.right,0,0),Search(root.left,1,height+1))
        return max(Search(root.left,1,0),Search(root.right,0,0))
        
        
        
        

# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def longestZigZag(self, root: TreeNode) -> int:
        longest = 0
        if not root:
            return longest
        #direction - right = True, left = False
        def helper(root, level, direction):
            nonlocal longest 
            if level > longest:
                longest = level
            if direction:
                if root.left:
                    helper(root.left, level+1, not direction)
                if root.right:
                    helper(root.right,1,direction)
            else:
                if root.right:
                    helper(root.right, level+1, not direction)
                if root.left:
                    helper(root.left, 1, direction)
            
        if root.right:
            helper(root.right,1,True)
        if root.left:
            helper(root.left,1,False)
        if not root.left and not root.right:
            return 0
        return longest

# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def longestZigZag(self, root: TreeNode) -> int:
        self.res = 0
        
        
        def dfs(node):
            # dfs(node) returns l, r
            # l means the longest zigzag path from node with the first edge going left
            # r means the longest zigzag path from node with the first edge going right
            if not node:
                return -1, -1
            _, r = dfs(node.left) # Note the the first returned value is useless
            l, _ = dfs(node.right)
            self.res = max(self.res, r + 1, l + 1)
            return r + 1, l + 1
        
        dfs(root)
        return self.res
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def longestZigZag(self, root: TreeNode) -> int:
        
        def check(root):
            l = 0   #   u4ee5u5f53u524du8282u70b9u5411u5de6
            r = 0   #   u4ee5u5f53u524du8282u70b9u5411u53f3
            m = 0   #   u5f53u524du5b50u6811u7684u6700u5927u503c(u8d77u70b9u4e0du4e00u5b9au662fu5f53u524du8282u70b9)
            if root.left != None:
                r1 = check(root.left)
                l = r1[1] + 1
                m = max(m, r1[2])
            if root.right!= None:
                r2 = check(root.right)
                r = r2[0] + 1
                m = max(m, r2[2])
            return (l, r, max(l, r, m))
        
        if root == None:
            return 0
        r = check(root)
        return r[2]
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def longestZigZag(self, root: TreeNode) -> int:
        
        memo={}

        ans=[0]


        def cur(node:TreeNode):
            if node!=None:
                memo[node]=[0,0]
            if node.left!=None:
                cur(node.left)
                memo[node][0]=memo[node.left][1]+1
                ans[0]=max(ans[0],memo[node][0])
            if node.right!=None:
                cur(node.right)
                memo[node][1]=memo[node.right][0]+1
                ans[0] = max(ans[0], memo[node][1])


        cur(root)
      
        return ans[0]

# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def longestZigZag(self, root: TreeNode) -> int:
        self.zigzag(root)
        return self.find_longest(root)
    
    def zigzag(self, node: TreeNode) -> int:
        if node is None:
            return
        self.zigzag(node.left)
        self.zigzag(node.right)
        
        if node.left is not None:
            node.left_depth = node.left.right_depth + 1
        else:
            node.left_depth = 0
            
        if node.right is not None:
            node.right_depth = node.right.left_depth + 1
        else:
            node.right_depth = 0
            
    def find_longest(self, node: TreeNode) -> int:
        if node is None:
            return 0
        left_max = self.find_longest(node.left)
        right_max = self.find_longest(node.right)
        return max(left_max, right_max, node.left_depth, node.right_depth)
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def longestZigZag(self, root: TreeNode) -> int:
        self.ans = 0
        
        def aux(root, isleft, memo):
            if not root:
                return 0
            if root in memo:
                if isleft:
                    return memo[root][1]
                else:
                    return memo[root][0]
            memo[root] = [0, 0]
            memo[root][0] = aux(root.left, 1, memo) + 1
            memo[root][1] = aux(root.right, 0, memo) + 1
            self.ans = max(self.ans, memo[root][1], memo[root][0])
            if isleft:
                return memo[root][1]
            else:
                return memo[root][0]
            
        if not root:
            return 0
        memo = {}
        aux(root, 0, memo)
        return self.ans - 1
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    
    def __init__(self):
        self.visited = set()
        
    def zigzagPath(self,node,direction):
        
        length = 0
        while node:
                
            if direction ==1:
                node = node.right
                if node and node.left:
                    self.visited.add((node,"left"))
            elif direction ==0:
                node = node.left
                if node and node.right:
                    self.visited.add((node,"right"))
            direction = 1 - direction
            if node!=None:
                length+=1
        return length
        
    
    def longestZigZag(self, root: TreeNode) -> int:
        max_length = 0
        S = [root]
        while S:
            node = S.pop(0)
            if node.right: 
                if (node,"right") not in self.visited:
                    max_length = max(self.zigzagPath(node,1),max_length)
                self.visited.add((node,"right"))
                S.append(node.right)
            if node.left: 
                if (node,"left") not in self.visited:
                    max_length = max(self.zigzagPath(node,0),max_length)
                self.visited.add((node,"left"))
                S.append(node.left)
        return max_length
        
        
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def longestZigZag(self, root: TreeNode) -> int:
        
        self.mx=0
        def backtrack(root,curr,temp):
            
            if not root:
                return
            self.mx=max(self.mx,temp)
            if curr=="left":
                backtrack(root.right,"right",temp+1)
                backtrack(root.left,"left",1)
            else:
                backtrack(root.left,"left",temp+1)
                backtrack(root.right,"right",1)
            
        backtrack(root,"left",0)
        backtrack(root,"right",0)
        return self.mx
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def longestZigZag(self, root: TreeNode) -> int:
        self.res = 0
        self.helper(root, True)
        self.helper(root, False)
        return self.res - 1
    
    def helper(self, root, isLeft):
        if not root:
            return 0
        
        left = self.helper(root.left, True)
        right = self.helper(root.right, False)
        self.res = max(self.res, left+1, right+1)
        
        return (right+1) if isLeft else (left+1)
        

class Solution:
    def longestZigZag(self, root: TreeNode) -> int:
        self.maxpath = 0
        l, r = self.visit(root)
        return self.maxpath
    
    
    def visit(self, root):
        l = 0
        r = 0
        if root.left:
            ll,lr = self.visit(root.left)
            l = lr + 1
        if root.right:
            rl,rr = self.visit(root.right)
            r = rl + 1
        if max(l,r) > self.maxpath:
            self.maxpath = max(l,r)
        return l, r
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def longestZigZag(self, root: TreeNode) -> int:
        if not root :
            return 0
        
        return max(self.length(root.left, 'l', 1), self.length(root.right,'r',1))
    def length(self, node, d, depth):
        if not node:
            return depth -1
        print((d, depth, node.val))
        
        if d == 'l':
            return max(self.length(node.right, 'r', depth+1), self.length(node.left, 'l',1))
        else:
            return max(self.length(node.left, 'l', depth+1), self.length(node.right,'r',1))

# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def longestZigZag(self, root: TreeNode) -> int:
        
        memo={}

        ans=[0]


        def cur(node:TreeNode):
            if node in list(memo.keys()):
                return 
            
            if node!=None:
                memo[node]=[0,0]
            if node.left!=None:
                cur(node.left)
                memo[node][0]=memo[node.left][1]+1
                ans[0]=max(ans[0],memo[node][0])
            if node.right!=None:
                cur(node.right)
                memo[node][1]=memo[node.right][0]+1
                ans[0] = max(ans[0], memo[node][1])


        cur(root)
      
        return ans[0]

# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def longestZigZag(self, root: TreeNode) -> int:
        
        self.max=0
        def dfs(node):
            if not node:
                return -1,-1
        
            l_dir_left,l_dir_right=dfs(node.left)
            r_dir_left,r_dir_right=dfs(node.right)
            self.max=max(self.max,l_dir_left,l_dir_right+1,r_dir_left+1,r_dir_right)
            return (l_dir_right+1,r_dir_left+1)
            
        dfs(root)
        return self.max
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
from typing import NamedTuple

class Result(NamedTuple):
    left_depth: int
    right_depth: int
    max_depth: int


class Solution:
    def longestZigZag(self, root: TreeNode) -> int:
        res = self.zigzag(root)
        return res.max_depth
    
    def zigzag(self, node: TreeNode) -> int:
        if node is None:
            return Result(-1, -1, 0)
        left_res = self.zigzag(node.left)
        right_res = self.zigzag(node.right)
        
        left_depth = left_res.right_depth + 1
        right_depth = right_res.left_depth + 1
        max_depth = max(left_depth, right_depth, left_res.max_depth, right_res.max_depth)
        return Result(left_depth, right_depth, max_depth)
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def longestZigZag(self, root: TreeNode) -> int:
        self.ret = 0
        self.dfs(root, 0, 0)
        self.dfs(root, 1, 0)
        return self.ret - 1
    
    def dfs(self, root, prevright, length):
        if root is None:
            self.ret = max(self.ret, length)
            return
        
        if prevright:
            self.dfs(root.left, 1 - prevright, length + 1)
            self.dfs(root.right, prevright, 1)
        else:
            self.dfs(root.right, 1 - prevright, length + 1)
            self.dfs(root.left, prevright, 1)
        
            
        return 
class Solution:
    def longestZigZag(self, root: TreeNode) -> int:
        check = 0 # left = 1, right = 2
        count = 0
        self.res = 0

        def dfs(node, count, check):
            if node:
                if check == 1: # from left                    
                    dfs(node.left, 0, 1)
                    dfs(node.right, count+1, 2)
                elif check == 2: # from right
                    dfs(node.left, count+1, 1)
                    dfs(node.right, 0, 2)
                elif check == 0: # from root
                    dfs(node.left, count, 1)
                    dfs(node.right, count, 2)
            self.res = max(self.res, count)
        dfs(root, count, check)
        return self.res
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
_max = 0
class Solution:
    def preorder(self,root, lr, cur_len):
        nonlocal _max
        
        if(root==None):
            return
        
        _max = max(_max,cur_len)
        
        if(lr=='l'):
            self.preorder(root.left,'l',1)
            self.preorder(root.right,'r',cur_len+1)
        elif(lr=='r'):
            self.preorder(root.left,'l',cur_len+1)
            self.preorder(root.right,'r',1)
        else:
            self.preorder(root.left,'l',cur_len+1)
            self.preorder(root.right,'r',cur_len+1)

            
        
    def longestZigZag(self, root: TreeNode) -> int:
        nonlocal _max
        _max = 0
        self.preorder(root,None,0)
        
        return _max

# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def longestZigZag(self, root: TreeNode) -> int:
        dp = [0]
        def f(n,ind):
            if not n:
                dp[ind] = (0,0)
            else:
                dp.extend([0,0])
                temp = len(dp)
                f(n.left,temp-2)
                f(n.right,temp-1)
                dp[ind] = (dp[temp-1][1]+1,dp[temp-2][0]+1)
        f(root,0)
        m = -1
        # print(dp)
        for i in dp:
            m = max(i[0],i[1],m)
        return m-1


                           

# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def longestZigZag(self, root: TreeNode) -> int:
        
        
        seenSoFar = 0
        def longestZigZagUtil(root):
            nonlocal seenSoFar
            if not root:
                return 0, 0
            
            Ll, Lr = longestZigZagUtil(root.left)
            Rl, Rr = longestZigZagUtil(root.right)
            
            curL, curR = 0, 0
            if root.left:
                curL = 1 + Lr
                seenSoFar = max(seenSoFar, Ll)
            if root.right:
                curR = 1 + Rl
                seenSoFar = max(seenSoFar, Rr)
            
            return curL, curR
        
        l, r = longestZigZagUtil(root)
        return max(l, r, seenSoFar)
                

# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def longestZigZag(self, root: TreeNode) -> int:
        res = 0
        def helper(root, direction):
            nonlocal res
            if not root:
                return 0
            left = helper(root.left, 'left')
            right = helper(root.right,'right')
            res = max(res, left+1, right+1)
            return right +1 if direction =='left' else left+1
        if not root:
            return 0
        helper(root,'left')
        helper(root, 'right')
        return res -1 

#         res = 0
#         def helper(root, direction):
#             nonlocal res
#             if not root:
#                 return 0
#             left = helper(root.left, 'left')
#             right = helper(root.right, 'right')
#             res = max(res, left+1, right+1)
#             return right+1 if direction =='left' else left+1
#         if not root:
#             return 0
#         helper(root, 'left')
#         helper(root, 'right')
#         return res-1

# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right

def solve(root,res,ind):
    if root==None:
        return 0
    l=solve(root.left,res,0)
    r=solve(root.right,res,1)
    if ind==0:
        temp=1+r
    elif ind==1:
        temp=1+l
    
    ans=1+max(l,r)
    res[0]=max(res[0],ans)
    return temp

class Solution:
    def longestZigZag(self, root: TreeNode) -> int:
         if root==None:
            return 0
         if root.left==None and root.right==None:
                return 0
         
         res1=[0]
         res2=[0]
         c1=solve(root,res1,0)
         c2=solve(root,res2,1)
         # print("aa",res1,res2,c1,c2)
         
         return max(res1[0],res2[0])-1

# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def longestZigZag(self, root: TreeNode) -> int:
        longest = [0]
        def dfs(node,d,c):
            if not node: 
                longest[0] = max(longest[0],c)
                return
            
            if d == 'r':
                dfs(node.left,'l',c+1)
            else:
                dfs(node.left,'l',0)
            if d == 'l':
                dfs(node.right,'r',c+1)
            else:
                dfs(node.right,'r',0)
            
        dfs(root,'',0)
        return longest[0]
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def longestZigZag(self, root: TreeNode) -> int:
        self.ans = 0
        def helper(node, direction, l):
            if not node: return
            self.ans = max(self.ans, l)
            helper((node.left, node.right)[direction], 1 - direction, l + 1)
            helper((node.left, node.right)[1-direction], direction, 1)
        helper(root, 0, 0)
        helper(root, 1, 0)
        return self.ans

# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def longestZigZag(self, root: TreeNode) -> int:
        def traverse(node,d,curr):
            print((d,curr))
            if not node:
                res[0] = max(res[0],curr-1)
                return
            if d==0:
                traverse(node.left,0,1)
                traverse(node.right,1,curr+1)
            else:
                traverse(node.left,0,curr+1)
                traverse(node.right,1,1)
                
        if not root:
            return 0
        res = [-float('inf')]
        traverse(root.left,0,1)
        traverse(root.right,1,1)
        return res[0]
    
        

class Solution:
    def longestZigZag(self, root: TreeNode) -> int:
        def solve(root):
            if root:
                solve(root.left)
                solve(root.right)
                if root.left:d[root][0] = d[root.left][1] + 1
                if root.right:d[root][1] = d[root.right][0] + 1
                self.ans = max(self.ans,max(d[root]))
            return self.ans
        d = defaultdict(lambda:[0,0])
        self.ans = 0 
        solve(root)
        return self.ans
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def longestZigZag(self, root: TreeNode) -> int:
        if not root:
            return 0
        
        if not root.left and  not root.right:
            return 0
        
        def dfs(root, flag, count):
            if not root:
                self.res = max(self.res, count-1)
                return
                
            if flag == 1:
                dfs(root.left,-1,1+count)
                
                dfs(root.right,1,1)
            else:
                dfs(root.right,1,1+count)
                dfs(root.left,-1,1)
        
            
        
        self.res = 0
        dfs(root, -1, 0)
        return self.res
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def longestZigZag(self, root: TreeNode) -> int:
        res = collections.namedtuple('res', 'left right total')
        
        def dfs(root):
            if not root:
                return res(-1, -1, -1)
        
            left = dfs(root.left)
            right = dfs(root.right)

            return res(left.right + 1, right.left + 1, max(left.right + 1, right.left + 1, left.total, right.total))
        
        return dfs(root).total

# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def node_is_on_the_right(self, node):
        if not node:
            return 0
        continuation = 1 + self.node_is_on_the_left(node.left)
        starts_here = self.node_is_on_the_right(node.right)
        self.m = max(self.m, starts_here)
        return continuation
    
    def node_is_on_the_left(self, node):
        if not node:
            return 0
        continuation = 1 + self.node_is_on_the_right(node.right)
        starts_here = self.node_is_on_the_left(node.left)
        self.m = max(self.m, starts_here)
        return continuation
    
    def longestZigZag(self, root: TreeNode) -> int:
        self.m = 0
        
        x = self.node_is_on_the_right(root) - 1
        self.m = max(self.m, x)
        
        return self.m
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def longestZigZag(self, root: TreeNode) -> int:
        self.mx = 0
        out = self.traverse(root, 0)
        out = self.traverse(root, 1)
        return self.mx - 1
        
    def traverse(self, node, ctype):
        if node is None:
            return 0, 0
        
        ll = lr = rl = rr = 0
        
        if node.left:
            ll, lr = self.traverse(node.left, 0)
        
        if node.right:
            rl, rr = self.traverse(node.right, 1)
        
        best = max(lr, rl) + 1
        self.mx = max(self.mx, best)
        
        return 1 + lr, 1 + rl
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def longestZigZag(self, root: TreeNode) -> int:
        if not root:
            return 0

        res = 0
        def dfs(root, direction, step):
            nonlocal res
            if not root:
                return

            if direction == "l":
                dfs(root.left, 'r', step + 1)
                dfs(root.right, 'l', 1)
            else:
                dfs(root.right, 'l', step + 1)
                dfs(root.left, 'r', 1)
            res = max(res, step)

        dfs(root, 'l', 0)
        dfs(root, 'r', 0)
        return res
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right

from collections import defaultdict

class Solution:
    def longestZigZag(self, root: TreeNode) -> int:
        
        levels = [root]
        ret = 0
        dp_r = defaultdict(int)
        dp_l = defaultdict(int)
        while levels:
            
            nxt = []
            
            for p in levels:
                if p.left:
                    nxt.append(p.left)
                    dp_l[p.left] = dp_r[p] + 1 
                    ret = max(ret, dp_l[p.left])
                    
                if p.right:
                    nxt.append(p.right)
                    dp_r[p.right] = dp_l[p] + 1 
                    ret = max(ret, dp_r[p.right])
            
            levels = nxt
            
            
        return ret
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def longestZigZag(self, root: TreeNode) -> int:
        if root == None:
            return 0
        
        max_zigzag = 0
        zigzag = 0
        
        stack = [(root,None,0)]
        
        while(stack):
            node = stack[-1][0]
            prev_dir = stack[-1][1]
            max_up_to_node = stack[-1][2]
            del stack[-1]
            
            #print(prev_dir, max_up_to_node, node.left, node.right)
            
            if max_up_to_node > max_zigzag:
                max_zigzag = max_up_to_node
            
            if prev_dir == None:
                if node.right != None:
                    stack.append((node.right, 'R', max_up_to_node + 1))
                if node.left != None:
                    stack.append((node.left, 'L', max_up_to_node + 1))
                    
            else:
                if prev_dir == 'R':
                    if node.right != None:
                        stack.append((node.right, 'R', 1))
                    if node.left != None:
                        stack.append((node.left, 'L', max_up_to_node + 1))
                        
                if prev_dir == 'L':
                    if node.right != None:
                        stack.append((node.right, 'R', max_up_to_node + 1))
                    if node.left != None:
                        stack.append((node.left, 'L', 1))
                        
        return max_zigzag

# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def longestZigZag(self, root: TreeNode) -> int:
        
        def dfs(root):
            if root is None:
                return 0, [0, 0] # long zig zag, left, right
            if root.left is None and root.right is None:
                return 0, [0, 0]
            
            left_zigzag, [_, right] = dfs(root.left)
            right_zigzag, [left, _] = dfs(root.right)
            
            zigzag = max(left_zigzag, right_zigzag)
            
            if root.right:
                right_zigzag = 1 + left
                zigzag = max(zigzag, right_zigzag)
            else:
                right_zigzag = 0
            
            if root.left:
                left_zigzag = 1 + right
                zigzag = max(zigzag, left_zigzag)
            else:
                left_zigzag = 0
                
            return zigzag, [left_zigzag, right_zigzag]
        
        return dfs(root)[0]
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def longestZigZag(self, root: TreeNode) -> int:
        self.max_ = 0
        def preorder(root,count,dir_):
            self.max_ = max(self.max_,count)
            
            if root.left:
                if dir_ == 1 or dir_==-1:
                    preorder(root.left,count+1,0)
                else:
                    preorder(root.left,1,0)
                    
            if root.right:
                if dir_ == 0 or dir_==-1:
                    preorder(root.right,count+1,1)
                else:
                    preorder(root.right,1,1)
                    
        preorder(root,0,-1)
        return self.max_
            

# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
from typing import NamedTuple

class Result(NamedTuple):
    left_depth: int
    right_depth: int
    max_depth: int


class Solution:
    def longestZigZag(self, root: TreeNode) -> int:
        return self.zigzag(root).max_depth
    
    def zigzag(self, node: TreeNode) -> int:
        if node is None:
            return Result(-1, -1, 0)
        left_res = self.zigzag(node.left)
        right_res = self.zigzag(node.right)
        
        left_depth = left_res.right_depth + 1
        right_depth = right_res.left_depth + 1
        max_depth = max(left_depth, right_depth, left_res.max_depth, right_res.max_depth)
        return Result(left_depth, right_depth, max_depth)
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def longestZigZag(self, root: TreeNode) -> int:
        def zigzag(node, direction, mz):
            if not node:
                return -1
            if direction == "l":
                zigzag(node.left, "l", mz)
                c = zigzag(node.left, "r", mz)+1
            else:
                zigzag(node.right, "r", mz)
                c = zigzag(node.right, "l", mz)+1
            mz[0] = max(mz[0], c)
            return c

        maxzigzag = [0]
        zigzag(root, "l", maxzigzag)
        zigzag(root, "r", maxzigzag)

        return maxzigzag[0]

# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def longestZigZag(self, root: TreeNode) -> int:
        def traverse(node,d,curr):
            if not node:
                res[0] = max(res[0],curr-1)
                return
            if d==0:
                traverse(node.left,0,1)
                traverse(node.right,1,curr+1)
            else:
                traverse(node.left,0,curr+1)
                traverse(node.right,1,1)
                
        if not root:
            return 0
        res = [-float('inf')]
        traverse(root.left,0,1)
        traverse(root.right,1,1)
        return res[0]
    
        

# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def longestZigZag(self, root: TreeNode) -> int:
        Max=[0]
        vl=set()
        vr=set()
        def path(node,l,d):
            if node == None:
                return l
            
        
            if(d==1):    #start at right
                vr.add(node)
                return path(node.left,l+1,0) 
            else:
                vl.add(node)
                return path(node.right,l+1,1)
            
        #visited=set()
        def dfs(node):
            if(node==None):
                return 
            if(node not in vl):
                Max[0]=max(Max[0],path(node,-1,0))
            if(node not in vr):
                Max[0]=max(Max[0],path(node,-1,1))
                    
            dfs(node.left)
            dfs(node.right)
        dfs(root)
            
        return Max[0]
        
        
            

# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def longestZigZag(self, root: TreeNode) -> int:
        
        self.res = 0
        def dfs(node):
            if not node:
                return -1, -1
            else:
                left = dfs(node.left)
                right = dfs(node.right)
                self.res = max(self.res, 1 + max(left[1], right[0]))
                return 1 + left[1], 1 + right[0]
        dfs(root)
        return self.res
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def __init__(self):
        self.memo = {}
    
    def h(self, node: TreeNode, left: bool) -> int:
        if not node:
            return 0
        if (node, left) not in self.memo:
            ret = 0
            if left and node.left is not None:
                ret = 1+self.h(node.left, False)
            elif not left and node.right is not None:
                ret = 1+self.h(node.right, True)
            self.memo[(node,left)] = ret
        return self.memo[(node,left)]
    def longestZigZag(self, root: TreeNode) -> int:
        if not root:
            return 0
        ret = [0]
        if root.left is not None:
            ret.extend([
                1+self.h(root.left, False),
                self.longestZigZag(root.left)
            ])
        if root.right is not None:
            ret.extend([
                1+self.h(root.right, True),
                self.longestZigZag(root.right)
            ])
        return max(ret)
                

# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def __init__(self):
        self.lmemo = {}
        self.rmemo = {}
        
    def longestZigZag(self, root: TreeNode) -> int:
        def lzz(root,mx):
            if root is not None:
                lzz(root.left,mx)
                lzz(root.right,mx)
                mx[0] = max(mx[0],llzz(root),rlzz(root))
            return mx
        def llzz(root):
            if root not in self.lmemo:
                self.lmemo[root] = 1+rlzz(root.left)
            return self.lmemo[root]
        def rlzz(root):
            if root not in self.rmemo:
                self.rmemo[root] = 1+llzz(root.right)
            return self.rmemo[root]
        self.lmemo[None] = self.rmemo[None] = 0
        return lzz(root,[float('-inf')])[0]-1
#https://leetcode.com/problems/longest-zigzag-path-in-a-binary-tree/discuss/534620/4-python-solutions
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
#        
class Solution:
    def longestZigZag(self, root: TreeNode) -> int:
        self.ans = 0  # nonlocal variable to store answer
		
        def recurse(root):
			# if null return -1 because length is defined 
			# as the number of nodes visited - 1. 
            if not root: return (-1,-1) 
			
			# l1 is max path len if we go left from current node and r1 if we go right						
            l1,r1 = recurse(root.left)
            l2,r2 = recurse(root.right)
			# Notice that if we go left from current node then we have no other choice but
			# to go right from node.left to make the path zigzag. 
            
			# That is why  r1 + 1 is the max path len  if we go left from current node. 
			# Same logic for l2 + 1
            print(root.val,r1,l2)
            self.ans = max(self.ans, max(r1 + 1, l2 + 1))#r1 + 1, l2 + 1
            print(self.ans)
            
            return (r1 + 1, l2 + 1)
			
        recurse(root)
        return self.ans
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    
    # u81eau5df1u5199u7684
    def longestZigZag(self, root: TreeNode) -> int:
        
        # return longest zigzig path of current node from right / left
        def dfs(node):
            if not node.left and not node.right:
                return (0, 0)
            left = right = 0
            if node.left:
                left = dfs(node.left)[1] + 1
            if node.right:
                right = dfs(node.right)[0] + 1
            # res.append(max(left, right))
            # res[0] = max(res[0], max(left, right))
            self.res = max(self.res, max(left, right))
            return (left, right)
        
        # u8fd9u6837u53cdu800cu4f1au8ba9u65f6u95f4u53d8u957f
        # res = [0]
        # res = []
        self.res = 0
        dfs(root)
        # return max(res) if res else 0
        # return res[0]
        return self.res
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def longestZigZag(self, root: TreeNode) -> int:
        self.sol = 0
        left = self.DFS(root,0)
        right = self.DFS(root,1)
        print((left,right,self.sol))
        
        return self.sol-1
        
    def DFS(self,node,direction):
        if not node: return 0
        left = self.DFS(node.left,0)
        right = self.DFS(node.right,1)
        self.sol = max(left+1,right+1,self.sol)
        print((self.sol))
        if direction==0:
            return right+1
        else:
            return left+1

# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def longestZigZag(self, root: TreeNode) -> int:
        self.sol = 0
        left = self.DFS(root,0)
        right = self.DFS(root,1)
        # print(left,right,self.sol)
        
        return self.sol-1
        
    def DFS(self,node,direction):
        if not node: return 0
        left = self.DFS(node.left,0)
        right = self.DFS(node.right,1)
        self.sol = max(left+1,right+1,self.sol)
        print((self.sol))
        if direction==0:
            return right+1
        else:
            return left+1

# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def longestZigZag(self, root: TreeNode) -> int:
        return self.dfs(root)[-1]
    
    def dfs(self, root):
        if not root:
            return [-1, -1, -1] # left, right, total
        
        left, right = self.dfs(root.left), self.dfs(root.right)
        
        return [left[1] + 1, right[0] + 1, max(left[1] + 1, right[0] + 1, left[-1], right[-1])]
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def longestZigZag(self, root: TreeNode) -> int:
        def helper(root):
            if not root:
                return (-1,-1,-1)
            if not root.left and not root.right:
                return (0,0,0)
            else:
                l1,l2,l3 = helper(root.left)
                r1,r2,r3 = helper(root.right)
                
                return (l3+1, max(l3+1, r1+1, l2,r2), r1+1)
        return helper(root)[1]
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def longestZigZag(self, root: TreeNode) -> int:
        if root is None:
            return 0
        self.sol = 0
        
        def recursive_long_zigzag(node):
            if node.left is None and node.right is None:
                return 0,0
            max_left = 0
            max_right = 0
            if node.left:
                _, left_s_right = recursive_long_zigzag(node.left)
                max_left = left_s_right + 1
                self.sol = max(self.sol, max_left)
            if node.right:
                right_s_left, _ = recursive_long_zigzag(node.right)
                max_right = right_s_left + 1
                self.sol = max(self.sol, max_right)
            return max_left, max_right
        
        recursive_long_zigzag(root)
        return self.sol
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def __init__(self):
        self.max_cnt=0
    
    def longestZigZag(self, root: TreeNode) -> int:
        self.dfs(root,True,0)
        self.dfs(root,False,0)
        return self.max_cnt
        
    def dfs(self, root, isLeft, cnt):
        if root is None:
            return
        self.max_cnt=max(self.max_cnt,cnt)
        if isLeft:
            self.dfs(root.left,False,cnt+1)
            self.dfs(root.right,True,1)
        else:
            self.dfs(root.right,True,cnt+1)
            self.dfs(root.left,False,1)
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def longestZigZag(self, root: TreeNode) -> int:
        
        def dfs(root):
            if root is None:
                return 0, [0, 0] # long zig zag, left, right

            left_zigzag, [_, right] = dfs(root.left)
            right_zigzag, [left, _] = dfs(root.right)
            
            zigzag = max(left_zigzag, right_zigzag)
            
            if root.right:
                right_zigzag = 1 + left
                zigzag = max(zigzag, right_zigzag)
            else:
                right_zigzag = 0
            
            if root.left:
                left_zigzag = 1 + right
                zigzag = max(zigzag, left_zigzag)
            else:
                left_zigzag = 0
                
            return zigzag, [left_zigzag, right_zigzag]
        
        return dfs(root)[0]
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def __init__(self):
        self.max_cnt=0
        
    def longestZigZag(self, root: TreeNode) -> int:
        self.helper(root,True,0)
        self.helper(root,False,0)
        return self.max_cnt

    def helper(self, root,isLeft,cnt):
        if root is None:
            return
        if isLeft:
            self.helper(root.left,False,cnt+1)
            self.helper(root.right,True,1)
        else:
            self.helper(root.right,True,cnt+1)
            self.helper(root.left,False,1)
            
        self.max_cnt=max(self.max_cnt,cnt)

# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def longestZigZag(self, root: TreeNode) -> int:
        
        ans = 0
        nodes = [root]
        self.dp = {}
        while(len(nodes)!=0):
            tmp = []
            for node in nodes:
                if(node.left is not None):
                    tmp.append(node.left)
                if(node.right is not None):
                    tmp.append(node.right)
                ans = max(ans, self.helper(node,1))
                ans = max(ans, self.helper(node,0))
            nodes = tmp
        return ans-1
        
    def helper(self,node,status):
        if(node is None):
            return 0
        if((node,status) in self.dp):
            return self.dp[(node,status)]
        if(status == 1):
            ans = self.helper(node.right,0)
        else:
            ans = self.helper(node.left, 1)
        self.dp[(node,status)]  = 1 + ans
        return 1+ans
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def longestZigZag(self, root: TreeNode) -> int:
        
        memo = dict()
        
        def zig_zag_len(node, direction):
            if not node:
                return -1
            if (node, direction) in memo:
                return memo[node, direction]
            if direction == 'L':
                memo[node, direction] = 1 + zig_zag_len(node.right, 'R')
            elif direction == 'R':
                memo[node, direction] = 1 + zig_zag_len(node.left, 'L')
            else:
                memo[node, direction] = max(1 + zig_zag_len(node.right, 'R'),
                                            1 + zig_zag_len(node.left, 'L'),
                                            zig_zag_len(node.right, 'N'),
                                            zig_zag_len(node.left, 'N'))
            return memo[node, direction]
        
        return zig_zag_len(root, 'N')
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def longestZigZag(self, root: TreeNode) -> int:
        def zigzag(node, mz):
            if not node:
                return -1, -1
            l1, r1 = zigzag(node.left, mz)
            l2, r2 = zigzag(node.right, mz)
            mz[0] = max(mz[0], r1+1, l2+1)
            return r1+1, l2+1

        maxzigzag = [0]
        zigzag(root, maxzigzag)
        zigzag(root, maxzigzag)

        return maxzigzag[0]

# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def run(self, node, side, count, memo):
        if not node:
            return count
        
        if node in memo:
            if memo[node][side] > -1:
                return memo[node][side]
        
        memo[node] = [-1, -1, -1]
        
        if side == 0:
            result = self.run(node.right, 1, count+1, memo)
        elif side == 1:
            result = self.run(node.left, 0, count+1, memo)
        else:
            using = max(self.run(node.right, 1, 1, memo), self.run(node.left, 0, 1, memo))
            notUsing = max(self.run(node.right, -1, 0, memo), self.run(node.left, -1, 0, memo))
            result = max(using, notUsing)
        
        memo[node][side] = result
        
        return result
        
    
    def longestZigZag(self, root: TreeNode) -> int:
        if not root:
            return 0
        
        return self.run(root, -1, 0, {})-1
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def longestZigZag(self, root: TreeNode) -> int:
        nodes = self.bfs(root)
        self.dp = {node:[False, False] for node in nodes}
        ans = 0
        for x in range(len(nodes)):
            ans = max(ans, self.recur(0, nodes[x], -1), self.recur(1, nodes[x], -1))
        return ans
        
    
    def bfs(self, node):
        q = [node]
        arr = []
        while q:
            r = q.pop()
            arr.append(r)
            if r.left:
                q.append(r.left)
            if r.right:
                q.append(r.right)
        return arr

    
    def recur(self, p, node, c):
        if not node:
            return c
        if self.dp[node][p] != False:
            return self.dp[node][p]

        if p == 0:
            self.dp[node][p] = self.recur(p^1, node.left, c+1)
            return self.dp[node][p]
        else:
            self.dp[node][p] = self.recur(p^1, node.right, c+1)
            return self.dp[node][p]

