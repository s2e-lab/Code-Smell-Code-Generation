class Solution:
    def largestTimeFromDigits(self, arr: List[int]) -> str:
        
        perm = list(itertools.permutations(arr,4))
        output = []
        
        for a,b,c,d in perm:
            temp = "{}{}:{}{}".format(a,b,c,d)
            try:
                datetime.datetime.strptime(temp, "%H:%M")
            except:
                continue
                
            output.append(temp)
                
        if not output: return ""
        
        return max(output)
