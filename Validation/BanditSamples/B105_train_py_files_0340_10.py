class Solution:
     def simplifyPath(self, path):
         """
         :type path: str
         :rtype: str
         """
         
         tokens = path.split('/')
         stack = list()
         for token in tokens:
             if token != '':
                 stack.append(token)
         
         res = ''
         back = 0
         while stack:
             top = stack.pop()
             if top == '.':
                 continue
             elif top == '..':
                 back = back + 1
             else:
                 if back == 0:
                     res = '/'+top+res
                 else:
                     back = back - 1
                     
         if res == '':
             return '/'
         return res 
                         

