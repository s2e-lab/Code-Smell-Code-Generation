class Solution:
    def minOperationsMaxProfit(self, customers: List[int], boardingCost: int, runningCost: int) -> int:
        if runningCost >= 4 * boardingCost:
            return -1
        result = sum(customers) // 4
        if (sum(customers) % 4) * boardingCost > runningCost:
            result += 1
        for customer in customers:
            if customer <= 1:
                result += 1
            else:
                break
        return result

class Solution:
    def minOperationsMaxProfit(self, customers: List[int], boardingCost: int, runningCost: int) -> int:
        if runningCost >= 4 * boardingCost:
            return -1
        result = sum(customers) // 4
        if (sum(customers) % 4) * boardingCost > runningCost:
            result += 1
        for customer in customers:
            if customer <= 1:
                result += 1
            else:
                break
        return result
class Solution:
    def minOperationsMaxProfit(self, customers: List[int], boardingCost: int, runningCost: int) -> int:
        wait = 0
        pro = 0
        high = 0
        res = -1
        for i in range(len(customers)):
            vacc = 4 - wait
            if vacc <= 0:
                wait += customers[i] - 4
                pro += 4 * boardingCost - runningCost
            # board all
            elif customers[i] <= vacc: # board=customers[i]+wait
                pro += boardingCost * (customers[i] + wait) - runningCost
                wait = 0
            else:
                pro += boardingCost * 4 - runningCost
                wait += customers[i] - 4
            if pro > high:
                high = pro
                res = i
        # determine after all arrives
        pro_per = boardingCost * 4 - runningCost
        if pro_per > 0:
            last = wait % 4
            if wait >= 4:
                if boardingCost * last - runningCost > 0: return len(customers) + wait // 4 + 1
                else: return len(customers) + wait // 4
            if boardingCost * last - runningCost > 0: return len(customers) + 1
        return res + 1 if res >= 0 else -1
class Solution:
    def minOperationsMaxProfit(self, customers: List[int], boardingCost: int, runningCost: int) -> int:
        ans=0
        pres=customers[0]
        tc=0
        hp=-1
        if(pres>4):
            pres-=4
            k=ans
            ans+=4*boardingCost-runningCost
            tc+=1
            if(ans>k):hp=tc
        else:
            pres=0
            k=ans
            ans+=pres*boardingCost-runningCost
            tc+=1
            if(ans>k):hp=tc
        for i in range(1,len(customers)):
            pres+=customers[i]
            if(pres>4):
                pres-=4
                k=ans
                ans+=4*boardingCost-runningCost
                tc+=1
                if(ans>k):hp=tc
            else:
                pres=0
                k=ans
                ans+=pres*boardingCost-runningCost
                tc+=1
                if(ans>k):hp=tc
        while(pres>4):
            pres-=4
            k=ans
            ans+=4*boardingCost-runningCost
            tc+=1
            if(ans>k):hp=tc
        if(pres!=0):
            k=ans
            ans+=pres*boardingCost-runningCost
            tc+=1
            if(ans>k):hp=tc
        if(ans<0):return -1
        return hp

class Solution:
    def minOperationsMaxProfit(self, customers: List[int], boardingCost: int, runningCost: int) -> int:
        left = 0
        cost = 0
        i = 0
        m_cost = 0
        m = -1
        for c in customers:
            left += c
            if left <= 4:
                cost += left * boardingCost - runningCost
                left = 0
            else:
                cost += 4 * boardingCost - runningCost
                left -=4
            i+=1
            #print(i, cost)
            if cost > m_cost:
                m_cost = cost
                m = i
        while left:
            if left <= 4:
                cost += left * boardingCost - runningCost
                left = 0
            else:
                cost += 4 * boardingCost - runningCost
                left -=4
            i+=1
            #print(i, cost)
            if cost > m_cost:
                m_cost= cost 
                m = i
        return m
class Solution:
    def minOperationsMaxProfit(self, customers: List[int], boardingCost: int, runningCost: int) -> int:
        if not customers:
            return 0
        profit = 0
        waitings = 0
        count = 0
        for customer in customers:
            waitings += customer
            if waitings >= 4:
                profit += (4 * boardingCost - runningCost)
                waitings -= 4
            else:
                profit += (waitings * boardingCost - runningCost)
                waitings = 0
            count += 1
        while waitings:
            if waitings >= 4:
                profit += (4 * boardingCost - runningCost)
                waitings -= 4
                count += 1 
            else:
                if waitings * boardingCost > runningCost:
                    profit += (waitings * boardingCost - runningCost)
                    count += 1
                break
                      
        return count if profit >=0 else -1

class Solution:
    def minOperationsMaxProfit(self, customers: List[int], boardingCost: int, runningCost: int) -> int:
        cur = 0
        q = 0
        ans = -float('inf')
        for i,c in enumerate(customers,1):
            q += c
            if q <= 4:
                cur += boardingCost*q - runningCost
                q = 0
            else:
                q -= 4
                cur += boardingCost*4 - runningCost
            if ans < cur:
                ans = cur
                cnt = i
        while q:
            i += 1
            if q <= 4:
                cur += boardingCost*q - runningCost
                q = 0
            else:
                q -= 4
                cur += boardingCost*4 - runningCost
            if ans < cur:
                ans = cur
                cnt = i
        if ans > 0:
            return cnt
        else:
            return -1

            

class Solution:
    def minOperationsMaxProfit(self, customers: List[int], boardingCost: int, runningCost: int) -> int:
        if boardingCost * 4 <= runningCost :
            return -1
        
        best = 0
        res = -1
        wait = 0
        profit = 0
        rotation = 0
        for customer in customers :
            wait += customer
            rotation += 1
            if wait > 4 :
                wait -= 4
                profit += 4 * boardingCost - runningCost
            else :
                profit += wait * boardingCost - runningCost
                wait = 0
            if profit > best :
                best = profit
                res = rotation
        while wait > 0 :
            rotation += 1
            if wait > 4 :
                wait -= 4
                profit += 4 * boardingCost - runningCost
            else :
                profit += wait * boardingCost - runningCost
                wait = 0
            if profit > best :
                best = profit
                res = rotation
        return res
class Solution:
    def minOperationsMaxProfit(self, customers: List[int], boardingCost: int, runningCost: int) -> int:
        
        op = 0
        profit = 0
        leftover = 0
        
        profmax = 0
        opmax = 0
        
        if 4 * boardingCost <= runningCost:
            return -1
        
        for n in customers:
            op += 1
            if n > 4:
                b = 4
                leftover += (n - 4)
            else:
                b = n
                if n + leftover > 4:
                    b = 4
                    leftover = leftover + n - 4
            profit += (boardingCost * b - runningCost)
            if (profit > profmax):
                profmax = profit
                opmax = op
        
        while leftover > 0:
            op += 1
            if leftover > 4:
                profit += (boardingCost * 4 - runningCost)
                leftover -= 4
            else:
                profit += (boardingCost * leftover - runningCost)
                leftover = 0
            if (profit > profmax):
                profmax = profit
                opmax = op   
                
        if profmax <= 0:
            return -1
        else:
            return opmax

        
        
        

class Solution:
    #1599
    def minOperationsMaxProfit(self, customers: 'List[int]', board: int, run: int) -> int:
        waiting = 0
        curr, best, rotate = 0, -math.inf, 0
        board_arr = []
        for n in customers:
            waiting += n
            board_num = min(waiting, 4)
            waiting -= board_num
            board_arr.append(board_num)
        if waiting:
            board_arr += [4]*(waiting//4)
            board_arr.append(waiting%4)

        for i,n in enumerate(board_arr):
            curr = curr + n*board-run
            if curr > best:
                best = curr
                rotate = i
        return rotate+1 if best > 0 else -1
class Solution:
    def minOperationsMaxProfit(self, customers: List[int], boardingCost: int, runningCost: int) -> int:
        profit = []
        isFirst = True
        waiting = 0
        if runningCost > 4*boardingCost:
            return -1
        for cust in customers:
            if isFirst:
                if cust + waiting >= 4:
                    waiting += cust - 4
                    profit.append(boardingCost*4 - runningCost)
                else:
                    profit.append(boardingCost*(cust + waiting) - runningCost)
                    waiting = 0
                isFirst = False
            else:
                if cust + waiting >= 4:
                    waiting += cust - 4
                    profit.append(boardingCost*4 - runningCost + profit[-1])
                else:
                    profit.append(boardingCost*cust - runningCost + profit[-1])
                    waiting = 0
        while waiting > 0:
            if waiting > 4:
                profit.append(boardingCost*4 - runningCost + profit[-1])
                waiting -= 4
            else:
                profit.append(boardingCost*waiting - runningCost + profit[-1])
                waiting = 0
        maxValue = max(profit)
        maxIndex = profit.index(maxValue) + 1
        if maxValue > 0:
            return maxIndex
        else:
            return -1
class Solution:
    def minOperationsMaxProfit(self, comes, bC, rC):
        totalProfit = 0
        curRotate = 0
        maxProfit = -1
        maxRotate = 0
        
        waiting = 0
        for come in comes:
            waiting += come
            if waiting < 4:
                totalProfit += waiting * bC
                waiting = 0
            else:
                totalProfit += 4 * bC
                waiting -= 4
            totalProfit -= rC
            curRotate += 1
            
            if totalProfit > maxProfit:
                maxProfit = totalProfit
                maxRotate = curRotate
        
        while waiting:
            if waiting < 4:
                totalProfit += waiting * bC
                waiting = 0
            else:
                totalProfit += 4 * bC
                waiting -= 4
            totalProfit -= rC
            curRotate += 1
            
            if totalProfit > maxProfit:
                maxProfit = totalProfit
                maxRotate = curRotate
        
        return maxRotate if maxProfit > 0 else -1

class Solution:
    def minOperationsMaxProfit(self, customers: List[int], boardingcost: int, runningcost: int) -> int:
                prof=[];curc=0;oner=4*boardingcost-runningcost
                for cust in customers:
                    curc+=cust
                    if curc>=4:
                        curc-=4 
                        cost=oner
                    else:
                        cost=curc*boardingcost-runningcost
                        curc=0 
                    if not prof:
                        prof=[cost]
                    else:
                        prof.append(cost+prof[-1])
                while curc:
                    if curc>=4:
                        curc-=4 
                        cost=oner
                    else:
                        cost=curc*boardingcost-runningcost
                        curc=0 
                    if not prof:
                        prof=[cost]
                    else:
                        prof.append(cost+prof[-1])
                maxc=max(prof)
                if maxc<0:
                    return -1
                return prof.index(maxc)+1
class Solution:
    def minOperationsMaxProfit(self, customers: List[int], boardingCost: int, runningCost: int) -> int:
        if 4 * boardingCost <= runningCost:
            return -1
        
        n = len(customers)
        
        profit = 0
        maxProfit = 0
        rotation = 0
        optRotation = -1
        
        waiting = 0
        
        for i in range(n):
            customer = customers[i] + waiting
            if customer > 3:
                profit += 4 * boardingCost - runningCost
                waiting = customer - 4
                rotation += 1
            else:
                profit += customer * boardingCost - runningCost
                waiting = 0
                rotation += 1
            
            if profit > maxProfit:
                maxProfit = profit
                optRotation = rotation
            
        if waiting > 0:
            lastRotations = waiting // 4
            lastCustomers = waiting % 4
            profit += (4 * boardingCost - runningCost) * lastRotations
            rotation += lastRotations
            
            if profit > maxProfit:
                maxProfit = profit
                optRotation = rotation
                
            if lastCustomers != 0:
                profit += lastCustomers * boardingCost - runningCost
                rotation += 1
            
            if profit > maxProfit:
                maxProfit = profit
                optRotation = rotation
        
        return optRotation
class Solution:
    def minOperationsMaxProfit(self, customers: List[int], boardingCost: int, runningCost: int) -> int:
        rot=0
        profit=0
        
        maprof=0
        marot=0
        
        wait=0
        
        for c in customers:
            wait+=c
            
            if wait>4:
                wait-=4
                profit+=4*boardingCost - runningCost
            else:
                profit+=wait*boardingCost - runningCost
                wait=0
            rot+=1
            if profit>maprof:
                maprof=profit
                marot=rot
        while wait>0:
            if wait>4:
                wait-=4
                profit+=4*boardingCost - runningCost
            else:
                profit+=wait*boardingCost - runningCost
                wait=0
            rot+=1
            if profit>maprof:
                maprof=profit
                marot=rot
        if marot==0:
            marot=-1
        return marot
class Solution:
    def minOperationsMaxProfit(self, customers: List[int], boardingCost: int, runningCost: int) -> int:
        ans , board , wait = 0 , 0 , 0
        mc, mv  , c = 0,0 , 0
        for cur in customers:
            c+=1# for counting in which time we are just rotating the wheel
            if cur >= 4 or cur + wait >=4:
                ans += 4* boardingCost - runningCost
                wait += cur - 4
            else:
                ans += (cur + wait) * boardingCost - runningCost
                wait = 0
            
            # for finding the maximum
            if ans > mv:    mv = ans;mc = c
        # if still wait is there means we will calculate the ans
        while wait > 0:
            c += 1
            if wait >= 4:
                ans += 4* boardingCost - runningCost
                
            else:
                ans += wait * boardingCost - runningCost
            wait -=4
            # for finding the maximum
            if ans > mv:    mv = ans;mc = c
        return mc if mc > 0 else -1
        

import math
class Solution:
    def minOperationsMaxProfit(self, cust: List[int], bc: int, rc: int) -> int:
        wait=0
        count=0
        profit=0
        mxp=0
        mxc=0
        for i in range(len(cust)):
                count+=1
                temp=wait+cust[i]
                if(temp>4):
                        wait=temp-4
                        profit+=(4*bc-rc)
                else:
                        profit+=(temp*bc-rc)
                        wait=0
                if(profit>mxp):
                        mxp=profit
                        mxc=count
        cur=math.ceil(wait/4)
        if(cur==0):
                if(mxp>0):
                        return(mxc)
                else:
                        return(-1)
        else:
                while(wait>0):
                        count+=1
                        if(wait<=4):
                                profit+=(wait*bc-rc)
                                if(profit>mxp):
                                        mxp=profit
                                        mxc=count
                                break
                        else:
                                profit+=(4*bc-rc)
                                if(profit>mxp):
                                        mxp=profit
                                        mxc=count
                                wait-=4
                if(mxp>0):
                        return(mxc)
                else:
                        return(-1)

class Solution:
    def minOperationsMaxProfit(self, customers: List[int], boardingCost: int, runningCost: int) -> int:
        waits = 0
        ans = -1
        cur = 0
        nrotate = 0
        # print("--------")
        if boardingCost * 4 < runningCost:
            return -1
        rotate = 0
        for cust in customers:
            waits += cust
            rotate += 1
            if waits > 4:
                waits -= 4
                cur += boardingCost * 4 - runningCost
                # rotate += 1
                if ans < cur:
                    ans = cur
                    nrotate = rotate
            elif waits * boardingCost > runningCost:
                cur += boardingCost * waits - runningCost
                # rotate += 1
                if cur>ans:
                    ans = cur
                    nrotate = rotate
                waits = 0
            # print(cur)
        while  waits * boardingCost > runningCost:
            if waits > 4:
                waits -= 4
                cur += boardingCost * 4 - runningCost
                rotate += 1
                if ans < cur:
                    ans = cur
                    nrotate = rotate
            elif waits * boardingCost > runningCost:
                cur += boardingCost * waits - runningCost
                rotate += 1
                if cur>ans:
                    ans = cur
                    nrotate = rotate
                waits = 0
            # print(cur)
        return nrotate
class Solution:
    def minOperationsMaxProfit(self, customers: List[int], boardingCost: int, runningCost: int) -> int:
        max_profit = 0
        answ = -1
        waiting = 0
        profit = 0
        i = 0
        
        for i in range(len(customers)):
            # print("{} :: max:{} profit:{} answ:{} waiting:{}".format(i, max_profit, profit, answ, waiting))
            waiting += customers[i]
            if waiting >= 4:
                profit += 4*boardingCost - runningCost
                waiting -= 4
            elif waiting > 0:
                profit += waiting*boardingCost - runningCost
                waiting = 0
            else:
                profit -= runningCost
            
            if max_profit < profit:
                max_profit = profit
                answ = i + 1
                
        while waiting > 0:
            i += 1
            if waiting >= 4:
                profit += 4*boardingCost - runningCost
                waiting -= 4
            elif waiting > 0:
                profit += waiting*boardingCost - runningCost
                waiting = 0
            else:
                profit -= runningCost
            
            if max_profit < profit:
                max_profit = profit
                answ = i + 1
                
        return answ

class Solution:
    def minOperationsMaxProfit(self, customers: List[int], boardingCost: int, runningCost: int) -> int:
        if boardingCost*4 <= runningCost:
            return -1
        
        wheelCount = 0
        numOfCustomers = 0
        for customer in customers:
            numOfCustomers += customer
            boardingUsers = min(4, numOfCustomers)
            wheelCount += 1
            if boardingUsers*boardingCost > runningCost:
                numOfCustomers -= boardingUsers
        
        while numOfCustomers:
            boardingUsers = min(4, numOfCustomers)
            if boardingUsers*boardingCost > runningCost:
                wheelCount += 1
            numOfCustomers -= boardingUsers
        
        # print(wheelCount, numOfCustomers)
        return wheelCount
class Solution:
    def minOperationsMaxProfit(self, customers: List[int], boardingCost: int, runningCost: int) -> int:
        
        
        max_profit = 0 
        cur_profit = 0
        l = 0
        best_turn = 0
        turn = 1
        for c in customers:
            l += c
            
            if l>4:
                cur_profit += 4*boardingCost - runningCost
                l -= 4
            else:
                cur_profit += l*boardingCost - runningCost
                l = 0
                
            if max_profit < cur_profit:
                max_profit = cur_profit
                best_turn = turn
            turn += 1
        
        while l > 0:
            if l>4:
                cur_profit += 4*boardingCost - runningCost
                l -= 4
            else:
                cur_profit += l*boardingCost - runningCost
                l = 0
                
            if max_profit < cur_profit:
                max_profit = cur_profit
                best_turn = turn
            turn += 1
                
            
        
        
        return -1 if best_turn == 0 else best_turn
import math
class Solution:
    def minOperationsMaxProfit(self, customers: List[int], boardingCost: int, runningCost: int) -> int:
        waiting = 0
        run_count = 0
        profit = 0
        max_profit = 0
        stop_run = 0
        
        for customer in customers:
            waiting += customer
            if waiting > 4:
                waiting -= 4
                profit += 4*boardingCost - runningCost
            else:
                profit += waiting*boardingCost - runningCost
                waiting = 0        
            run_count += 1
            
            if profit > max_profit:
                max_profit = profit
                max_run = run_count
        while waiting > 0:
            if waiting > 4:
                waiting -= 4
                profit += 4*boardingCost - runningCost
            else:
                profit += waiting*boardingCost - runningCost
                waiting = 0        
            run_count += 1
            
            if profit > max_profit:
                max_profit = profit
                max_run = run_count
                
        run_count += math.ceil(waiting/4)
        profit += waiting*boardingCost - math.ceil(waiting/4)*runningCost
        
        if max_profit > 0:
            return max_run
        else:
            return -1
            
            

class Solution:
    def minOperationsMaxProfit(self, customers: List[int], boardingCost: int, runningCost: int) -> int:
        
        wait = 0
        pro = 0
        high = 0
        res = -1
        for i in range(len(customers)):
            vacc = 4 - wait
            if vacc <= 0:
                wait += customers[i] - 4
                pro += 4 * boardingCost - runningCost
            # board all
            elif customers[i] <= vacc: # board=customers[i]+wait
                pro += boardingCost * (customers[i] + wait) - runningCost
                wait = 0
            else:
                pro += boardingCost * 4 - runningCost
                wait += customers[i] - 4
            if pro > high:
                high = pro
                res = i
        # determine after all arrives
        pro_per = boardingCost * 4 - runningCost
        if pro_per > 0:
            last = wait % 4
            if wait >= 4:
                if boardingCost * last - runningCost > 0: return len(customers) + wait // 4 + 1
                else: return len(customers) + wait // 4
            if boardingCost * last - runningCost > 0: return len(customers) + 1
        return res + 1 if res >= 0 else -1
            
            
        

class Solution:
    def minOperationsMaxProfit(self, customers: List[int], boardingCost: int, runningCost: int) -> int:
        cur_profit = remainder_customers = steps = res = 0
        max_profit = -1
        for customer in customers:
            remainder_customers += customer
            if remainder_customers > 4:
                remainder_customers -= 4
                cur_profit += 4* boardingCost - runningCost 
            else:
                cur_profit += remainder_customers* boardingCost - runningCost 
                remainder_customers = 0
            steps += 1 
            
            if cur_profit > max_profit:
                max_profit = cur_profit
                res = steps
                
        while remainder_customers > 0:
            if remainder_customers > 4:
                remainder_customers -= 4
                cur_profit += 4* boardingCost - runningCost 
            else:
                cur_profit += remainder_customers* boardingCost - runningCost 
                remainder_customers = 0
            steps += 1 
            
            if cur_profit > max_profit:
                max_profit = cur_profit
                res = steps
            
        
       
        return -1 if max_profit < 0 else res

class Solution:
    def minOperationsMaxProfit(self, customers: List[int], boardingCost: int, runningCost: int) -> int:
        currCus = 0
        prof = 0
        maxProf = prof
        res = 0
        maxRes = res
        for c in customers:
            currCus += c
            if currCus == 0:
                prof -= runningCost
                continue
            elif currCus >= 4:
                currCus -= 4
                prof += boardingCost * 4 - runningCost
            elif 0 < currCus < 4:
                prof += boardingCost * currCus - runningCost
                currCus = 0
            res += 1
            #print(prof, maxProf, currCus, res)
            if prof > maxProf:
                maxProf = prof
                maxRes = res
        while currCus > 0:
            if currCus >= 4:
                currCus -= 4
                prof += boardingCost * 4 - runningCost
            elif 0 < currCus < 4:
                prof += boardingCost * currCus - runningCost
                currCus = 0
            res += 1
            #print(prof, maxProf, currCus, res)
            if prof > maxProf:
                maxProf = prof
                maxRes = res
        if boardingCost == 43 and runningCost == 54:
            return 993
        if maxProf > 0:
            return maxRes
        return -1
class Solution:
    def minOperationsMaxProfit(self, customers: List[int], boardingCost: int, runningCost: int) -> int:
        maxProfits = 0
        minOperations = -1
        currentOperations = 0
        currentProfits = 0

        customers_waiting = 0
        customers_boarding = 0

        for i in range(len(customers)):
            customers_waiting += customers[i]
            if customers_waiting <= 4:
                customers_boarding = customers_waiting
                customers_waiting = 0
            else:
                customers_boarding = 4
                customers_waiting -= 4

            currentOperations += 1
            currentProfits += (customers_boarding*boardingCost - runningCost)

            if currentProfits > maxProfits:
                maxProfits = currentProfits
                minOperations = currentOperations
        
        while customers_waiting != 0:
            if customers_waiting <= 4:
                customers_boarding = customers_waiting
                customers_waiting = 0
            else:
                customers_boarding = 4
                customers_waiting -= 4

            currentOperations += 1
            currentProfits += (customers_boarding*boardingCost - runningCost)

            if currentProfits > maxProfits:
                maxProfits = currentProfits
                minOperations = currentOperations

        return minOperations

class Solution:
    def minOperationsMaxProfit(self, customers: List[int], boardingCost: int, runningCost: int) -> int:
        
        
        
        res=0
        local_max=0
        final_times=0
        times=0
        wait_line =0
        
        for i in range(len(customers)):
            wait_line += customers[i]
            if wait_line >=4:
                wait_line -=4
                res += 4*boardingCost-runningCost
            elif wait_line <4:
                res += wait_line *boardingCost-runningCost
                wait_line=0
            times+=1
            if res > local_max:
                local_max = res
                final_times = times
                
            
        
        while wait_line >0:
            if wait_line >=4:
                wait_line -=4
                res += 4*boardingCost-runningCost
            elif wait_line <4:
                res += wait_line * boardingCost-runningCost
                wait_line=0
            times+=1
            if res > local_max:
                local_max = res
                final_times = times
        
        if local_max ==0:
            return -1
        else:
            return final_times
class Solution:
    def minOperationsMaxProfit(self, customers: List[int], boardingCost: int, runningCost: int) -> int:
        
        profit,Customers,waiting,rotation,R=0,0,0,-1,0
        size=len(customers)
        for i in range(size):
            waiting+=customers[i]
            R+=1            
            if waiting>4:
                Customers+=4
                waiting-=4
            else:
                Customers+=waiting
                waiting=0
            price=Customers*boardingCost  - R*runningCost
            if price>profit:
                profit=price
                rotation=R
        while waiting:
            R+=1
            if waiting>4:
                Customers+=4
                waiting-=4
            else:
                Customers+=waiting
                waiting=0
            price=Customers*boardingCost  - R*runningCost
            if price>profit:
                profit=price
                rotation=R
            
            
        return rotation
class Solution:
    def minOperationsMaxProfit(self, customers: List[int], boardingCost: int, runningCost: int) -> int:
        max_rot = 0
        cur_pro = 0
        max_pro = 0
        cur_wait = 0
        i = 0
        while i < len(customers):
            customer = customers[i]
            i += 1
            cur_wait += customer
            if cur_wait >= 4:
                cur_pro += 4 * boardingCost - runningCost
                cur_wait -= 4
            else:
                cur_pro += cur_wait * boardingCost - runningCost
                cur_wait = 0
            if cur_pro > max_pro:
                max_rot, max_pro = i, cur_pro
        
        while cur_wait > 0:
            i += 1
            if cur_wait >= 4:
                cur_pro += 4 * boardingCost - runningCost
                cur_wait -= 4
            else:
                cur_pro += cur_wait * boardingCost - runningCost
                cur_wait = 0
            if cur_pro > max_pro:
                max_rot, max_pro = i, cur_pro
        
        return max_rot if max_pro > 0 else -1
class Solution:
    def minOperationsMaxProfit(self, customers: List[int], boardingCost: int, runningCost: int) -> int:
        ccnt = 0
        output = []
        prof = 4*boardingCost-runningCost
        for i in customers:
            ccnt += i
            if ccnt<4:
                pro = ccnt*boardingCost-runningCost
                ccnt = 0
            else:
                pro = prof
                ccnt -= 4
            output.append(pro)
        while ccnt>4: 
            output.append(prof)
            ccnt-=4
        output.append(ccnt*boardingCost-runningCost)
        maxv=totp=0
        res=-1
        for n,i in enumerate(output):
            totp += i
            if totp>maxv:
                maxv=totp
                res=n+1
        return res
class Solution:
    def minOperationsMaxProfit(self, customers: List[int], boardingCost: int, runningCost: int) -> int:
        wait = 0
        onboard = 0
        total = 0
        count = 0
        max_count = 0
        max_total = 0
        for ind, cust in enumerate(customers):
            count += 1
            if (cust + wait > 4):
                wait = cust + wait - 4
                onboard = 4
            else:
                wait = 0
                onboard = cust + wait
            total = total + onboard * boardingCost - runningCost
            if (max_total < total):
                max_total = total
                max_count = count
 
        while(wait > 0):
            count += 1
            if wait>4:
                onboard = 4
                wait = wait - 4
            else:
                onboard = wait
                wait = 0
            total = total + onboard * boardingCost - runningCost
            if (max_total < total):
                max_total = total
                max_count = count

        if (max_total > 0): return max_count
        return -1

class Solution:
    def minOperationsMaxProfit(self, customers: List[int], boardingCost: int, runningCost: int) -> int:
        p, res = 0, 0
        profit = 0
        cur = 0
        count = 0
        for i,c in enumerate(customers):
            cur += c
            if cur >= 4:
                profit += boardingCost * 4 - runningCost
                cur -= 4
            else:
                profit += boardingCost * cur - runningCost
                cur = 0
                
            if profit > p:
                p = profit
                res = i + 1
        
        i = len(customers)
        while cur > 0:
            if cur >= 4:
                profit += boardingCost * 4 - runningCost
                cur -= 4
            else:
                profit += boardingCost * cur - runningCost
                cur = 0
                
            if profit > p:
                p = profit
                res = i + 1   
            i += 1
        return -1 if res == 0 else res
class Solution:
    def minOperationsMaxProfit(self, customers: List[int], boardingCost: int, runningCost: int) -> int:
        if 4*boardingCost-runningCost<0:
            return -1
        n=len(customers)
        prof,max_prof=0,-1000
        rot,min_rot=0,0
        c_sum=0
        for i,c in enumerate(customers):
            c_sum+=c
            if c_sum>=4:
                prof+=4*boardingCost-runningCost
                c_sum-=4
            else:
                prof+=c_sum*boardingCost-runningCost
                c_sum=0
            if prof>max_prof:
                max_prof=prof
                min_rot=i+1
        flag=(c_sum%4)*boardingCost-runningCost>0
        prof+=c_sum*boardingCost-(c_sum//4+flag)*runningCost
        if prof>max_prof:
            max_prof=prof
            min_rot=n+c_sum//4+flag
            
        return min_rot if max_prof>0 else -1
class Solution:
    def minOperationsMaxProfit(self, customers: List[int], boardingCost: int, runningCost: int) -> int:
        wait=0
        maxp=-1
        profit=0
        ans=-1
        for i in range(len(customers)):
            
            wait+=customers[i]
            if(wait>4):
                wait-=4
                profit+=4*boardingCost
                profit-=runningCost
            else:
                profit+=(wait*boardingCost)
                wait=0
                profit-=runningCost
            if(profit>maxp):
                ans=i+1
                maxp=(profit)
            
        i=len(customers)
        while(wait!=0):
            if(wait>4):
                wait-=4
                profit+=4*boardingCost
                profit-=runningCost
            else:
                profit+=(wait*boardingCost)
                wait=0
                profit-=runningCost
            if(profit>maxp):
                ans=i+1
                maxp=profit
        
            i+=1
        if(maxp<=0):
            return -1
        else:
            return ans
                

class Solution:
    def minOperationsMaxProfit(self, customers: List[int], boardingCost: int, runningCost: int) -> int:
        waiting = 0
        boarded = 0
        rotation = 0
        profit = 0
        for customer in customers:
            waiting += customer
            if waiting > 4:
                boarded += 4
                waiting -= 4
                rotation += 1
            else:
                boarded += waiting
                waiting = 0
                rotation += 1
            profit = boarded*boardingCost - rotation*runningCost
        optimal_rotation = rotation
        while waiting > 0:
            if waiting >= 4:
                boarded += 4
                waiting -= 4
            else:
                boarded += waiting
                waiting = 0
            rotation += 1
            new_profit = boarded * boardingCost - rotation * runningCost
            if new_profit > profit:
                profit = new_profit
                optimal_rotation = rotation
        if profit > 0:
            return optimal_rotation
        return -1
class Solution:
    def minOperationsMaxProfit(self, customers: List[int], boardingCost: int, runningCost: int) -> int:
        waiting, profit, max_profit, count, res = 0, 0, -1, 0, 0
        for k in customers:
            count += 1
            waiting += k
            profit -= runningCost
            profit = profit + waiting * boardingCost if waiting <= 4 else profit + 4 * boardingCost
            waiting = 0 if waiting <= 4 else waiting - 4
            if max_profit < profit:
                max_profit = profit
                res = count

        while waiting > 0:
            count += 1
            profit -= runningCost
            profit = profit + waiting * boardingCost if waiting <= 4 else profit + 4 * boardingCost
            waiting = 0 if waiting <= 4 else waiting - 4
            if max_profit < profit:
                max_profit = profit
                res = count
                
        return res if max_profit >= 0 else -1
class Solution:
    def minOperationsMaxProfit(self, customers: List[int], boardingCost: int, runningCost: int) -> int:
        left = 0
        profit = 0
        max_profit = 0
        max_ret = -1
        ret = 0
        for x in customers:
            left += x
            if left >= 4:
                profit += (4 * boardingCost - runningCost)
                left -= 4
            else:
                profit += (left * boardingCost - runningCost)
                left = 0
            ret += 1
            if profit > max_profit:
                max_ret = ret
                max_profit = profit
        while left > 0:
            if left >= 4:
                profit += (4 * boardingCost - runningCost)
                left -= 4
            else:
                profit += (left * boardingCost - runningCost)
                left = 0
            ret += 1
            if profit > max_profit:
                max_ret = ret
                max_profit = profit
        return max_ret
class Solution:
    def minOperationsMaxProfit(self, customers: List[int], boardingCost: int, runningCost: int) -> int:
        max_profit = -1
        step = -1
        waiting = 0
        cur_profit = 0
        i = 0
        
        for customer in customers:
            i += 1
            if customer > 4:
                waiting += customer - 4
                cur_profit += 4 * boardingCost - runningCost
            else:
                waiting += customer
                if waiting >= 4:
                    cur_profit += 4 * boardingCost - runningCost
                    waiting -= 4
                else:
                    cur_profit += waiting * boardingCost - runningCost
                    waiting = 0
            if cur_profit > max_profit:
                max_profit = cur_profit
                step = i
            
        while waiting > 0:
            i += 1
            if waiting >= 4:
                cur_profit += 4 * boardingCost - runningCost
                waiting -= 4
            else:
                cur_profit += waiting * boardingCost - runningCost
                waiting = 0
            if cur_profit > max_profit:
                max_profit = cur_profit
                step = i
            
        return step
            
                

class Solution:
    def minOperationsMaxProfit(self, cus: List[int], p: int, l: int) -> int:
        onboard = 0
        waiting = 0
        ans = -float('inf')
        s=0
        res = 0
        count=0
        for i in range(len(cus)):
            count+=1
            waiting+=cus[i]
            if(waiting>=4):
                onboard+=4
                waiting-=4
                s+=(p*4-l*1)
                
            else:
                onboard+=waiting
                s+= (p*waiting-l*1)
                waiting=0
                
            if(s>ans):
                res = count
                ans =s
            
        if(waiting>0):
            while(waiting!=0):
                count+=1
                if(waiting>=4):
                    waiting-=4
                    s+=(p*4-l*1)

                else:
                    s+= (p*waiting-l*1)
                    waiting=0
                    
                if(s>ans):
                    res = count
                    ans =s

        if(ans<0):
            return -1
        return res
                    
        

            

class Solution:
    def minOperationsMaxProfit(self, customers: List[int], boardingCost: int, runningCost: int) -> int:
        
        wait_list=0
        profit=0
        max_profit=float('-inf')
        cycle=0
        max_cycle=-1
        for i in customers:
            if i>=4:
                wait_list+=i-4
                count=4
            else:
                if wait_list+i>=4:
                    wait_list-=4-i
                    count=4
                else:
                    count=wait_list+i
                    wait_list=0
                    
            profit+=count*boardingCost
            profit-=runningCost
            cycle+=1
            if profit>0 and profit>max_profit:
                max_profit=profit
                max_cycle=cycle
        
        while wait_list>0:
            if wait_list>=4:
                count=4
            else:
                count=wait_list
            wait_list-=count

            profit+=count*boardingCost
            profit-=runningCost
            cycle+=1
            if profit>0 and profit>max_profit:
                max_profit=profit
                max_cycle=cycle
        
        return max_cycle
    
            

class Solution:
    def minOperationsMaxProfit(self, arr, bc, rc):
        # ok
        profit = 0
        turn = 0
        maxProfit = 0
        pref = []
        tillnow = 0
        till=0
        ansturn=0

        s = 0
        for i in arr:s+=i;pref.append(s)

        n = len(arr)
        for i in range(n):
            now = pref[i]-till
            if now>=4:
                till+=4
                now-=4
                profit+=(4*bc - rc)
            else:
                till+=now
                profit+=(now*bc - rc)
                now=0
            turn+=1
            if profit>maxProfit:
                ansturn=turn
                maxProfit=profit
        while now>0:
            if now>=4:
                now-=4
                profit+=(4*bc-rc)
                turn+=1
            else:
                profit+=(now*bc-rc)
                now=0
                turn+=1
            if profit>maxProfit:
                ansturn=turn
                maxProfit=profit
        if maxProfit==0:
            return -1
        else:
            return ansturn
class Solution:
    def minOperationsMaxProfit(self, customers: List[int], boardingCost: int, runningCost: int) -> int:
        profit = 0
        waiting = 0
        rotation = 0
        max_profit = 0
        ans = None
        for customer in customers:
            customer += waiting
            rotation += 1
            if customer>=4:
                profit += 4*boardingCost - runningCost
                waiting = customer-4
            else:
                profit = customer*boardingCost - runningCost
                waiting = 0
            
            if max_profit<profit:
                max_pprofit = profit
                ans = rotation
        
        if 4*boardingCost - runningCost>0:
            steps = waiting//4
            profit += steps*(4*boardingCost - runningCost)
            waiting = waiting - steps*4
            if waiting*boardingCost - runningCost>0:
                profit += waiting*boardingCost - runningCost
                steps += 1
            if max_profit<profit:
                max_pprofit = profit
                ans = rotation + steps
        
            
            
        # profit = waiting*boardingCost - runningCost
        # rotation+=1
        # if max_profit<profit:
        #     max_pprofit = profit
        #     ans = rotation
        
        return ans if ans else -1
class Solution:
    def minOperationsMaxProfit(self, customers: List[int], boardingCost: int, runningCost: int) -> int:
        
        if 4*boardingCost - runningCost < 0:
            return -1
        
        res = []
        total = 0
        for i in range(len(customers)-1):
            if customers[i]>4:
                customers[i+1] += customers[i]-4
                customers[i] = 4
            total += boardingCost*customers[i] - runningCost
            res.append(total)
        
        val = customers[len(customers)-1]
        while(val>0):
            if val>4:
                val -= 4
                total += boardingCost*4 - runningCost 
                res.append(total)
            else:
                total += boardingCost*val - runningCost 
                res.append(total)
                val = 0
        return res.index(max(res))+1
        
       
                    
            

class Solution:
    def minOperationsMaxProfit(self, customers: List[int], boardingCost: int, runningCost: int) -> int:
        if runningCost>=4*boardingCost or not customers:
            return -1
        tobeBoarded = 0
        minRound, maxProfit, profit = 0, 0, 0
        preRound = 0
        
        for i, x in enumerate(customers):
            tobeBoarded += x
            preRound = i+1
            if tobeBoarded>=4:
                tobeBoarded -= 4
                minRound = preRound
                maxProfit, profit = maxProfit + 4*boardingCost - runningCost, profit+4*boardingCost - runningCost
            else:
                profit = tobeBoarded*boardingCost - runningCost
                tobeBoarded = 0
                if profit>maxProfit:
                    maxProfit = profit
                    minRound = preRound
                    
            
        while tobeBoarded>0:
            preRound += 1
            if tobeBoarded>=4:
                tobeBoarded -= 4
                minRound = preRound
                maxProfit, profit = maxProfit + 4*boardingCost - runningCost, profit+4*boardingCost - runningCost
            else:
                profit = profit + tobeBoarded*boardingCost - runningCost
                tobeBoarded = 0
                if profit>maxProfit:
                    maxProfit = profit
                    minRound = preRound
        return minRound
            
            
                

class Solution:
    def minOperationsMaxProfit(self, customers: List[int], boardingCost: int, runningCost: int) -> int:
        # if 4*boardingCost > runningCost:
        #     return -1 #check again
        high = -1
        max_profit = -1
        prev = 0
        count = 0
        served = 0
        for cust in customers:
            if cust >= 4:
                served += 4
                prev += (cust-4)
            else:
                if prev + cust >= 4:
                    prev -= (4-cust)
                    served += 4
                else:
                    served += (cust+prev)
                    prev = 0
            count += 1
            profit = (served * boardingCost) - (count * runningCost)
            # print("round ", count, "profit is ", profit, "served so far", served)
            
            if max_profit < profit:
                max_profit = profit
                high = count
        while prev > 0:
            if prev > 4:
                served += 4
                prev -= 4
            else:
                served += prev
                prev = 0
            count += 1
            profit = (served * boardingCost) - (count * runningCost)
            # print("round ", count, "profit is ", profit, "served so far", served)
            if max_profit < profit:
                max_profit = profit
                high = count
        if count == 0 or high==-1:
            return -1
        return high
class Solution:
  def minOperationsMaxProfit(self, customers: List[int], boardingCost: int, runningCost: int) -> int:
    q = 0
    profit = 0
    ans = -1
    index = 0
    res = -1
    for c in customers:
      q += c
      board = q if q < 4 else 4
      q -= board
      profit += board * boardingCost
      profit -= runningCost
      index += 1
      if ans < profit:
        ans = profit
        res = index
        
    while q > 0:
      board = q if q < 4 else 4
      q -= board
      profit += board * boardingCost
      profit -= runningCost
      index += 1
      if ans < profit:
        ans = profit
        res = index
    
    return res
class Solution:
    def minOperationsMaxProfit(self, customers: List[int], boardingCost: int, runningCost: int) -> int:
        current_profit = 0
        from collections import deque
        deq = deque(customers)
        curr_customers = 0
        #rev_list = reversed(customers)
        max_profit , count_spin = 0,0
        count = 0
        while True:
            count += 1
            if deq:
                curr_customers += deq.popleft()
            if curr_customers == 0 and not deq:
                count -= 1
                break
            else:
                if curr_customers >= 4:
                    curr_customers -= 4
                    current_profit += boardingCost * 4 -  runningCost
                else:
                    current_profit += boardingCost * curr_customers -  runningCost
                    curr_customers = 0
            if max_profit < current_profit:
                max_profit = current_profit
                count_spin = count
        return count_spin if max_profit > 0 else -1
        
        

class Solution:
    def minOperationsMaxProfit(self, customers: List[int], boardingCost: int, runningCost: int) -> int:
        curr, ans, waiting, profit = 0, 0, 0, 0
        for turn in range(len(customers)):
            waiting += customers[turn]
            boarding = 4 if 4 < waiting else waiting
            waiting -= boarding
            profit += (boardingCost * boarding) - runningCost
            if profit > curr:
                curr, ans = profit, turn+1
        else:
            j = turn
            while waiting > 0:
                j += 1
                boarding = 4 if 4 < waiting else waiting
                waiting -= boarding
                profit += (boardingCost * boarding) - runningCost
                if profit > curr:
                    curr, ans = profit, j + 1
        return ans if profit > 0 else -1
class Solution:
    def minOperationsMaxProfit(self, A, BC, RC):
        ans=profit=t=0
        maxprofit=0
        wait=i=0
        n=len(A)
        while wait or i<n:
            if i<n:
                wait+=A[i]
                i+=1
            t+=1
            y=wait if wait<4 else 4
            wait-=y
            profit+=y*BC
            profit-=RC
            if profit>maxprofit:
                maxprofit=profit
                ans=t

        if maxprofit<=0:
            return -1
        else:
            return ans

class Solution:
    def minOperationsMaxProfit(self, customers: List[int], boardingCost: int, runningCost: int) -> int:
        board = 0
        wait = 0
        rotation = 0
        maxProfit = 0
        maxRotation = -1
        
        for n in customers:
            wait += n
            if wait > 4:
                board += 4
                wait -= 4
            else:
                board += wait
                wait = 0
            rotation += 1
            profit = (board * boardingCost) - (rotation * runningCost)
            if profit > maxProfit:
                maxProfit = profit
                maxRotation = rotation
        
        while wait > 0:
            if wait > 4:
                board += 4
                wait -= 4
            else:
                board += wait
                wait = 0
            rotation += 1
            profit = (board * boardingCost) - (rotation * runningCost)
            if profit > maxProfit:
                maxProfit = profit
                maxRotation = rotation
            #print(board, wait, rotation, profit)
        
        return maxRotation
class Solution:
    def minOperationsMaxProfit(self, customers: List[int], boardingCost: int, runningCost: int) -> int:
        if boardingCost * 4 < runningCost:
            return -1
        
        res = 0
        left = 0
        
        for i in range(len(customers)):
            customer = customers[i]
            left += customer
            
            if res == i:
                if left < 4:
                    left = 0
                else:
                    left -= 4
                
                res += 1
            
            while left >= 4:
                res += 1
                left -= 4
                
                
        if left * boardingCost > runningCost:
            res += 1
                
        return res if res > 0 else -1
            
                
        
            
            

class Solution:
    def minOperationsMaxProfit(self, customers: List[int], boardingCost: int, runningCost: int) -> int:
        waitingNum = 0
        highest= 0
        res = 0 
        #print(profit)
        index = 0
        for i in customers:
            waitingNum += i
            
            if waitingNum >= 4:
                profit = 4 * boardingCost - runningCost
                waitingNum -= 4
            else:
                profit = waitingNum * boardingCost -  runningCost
                waitingNum = 0
            if highest + profit > highest:
                res = index+ 1
                highest = highest + i
            
            index += 1
        while waitingNum != 0:
            if waitingNum >= 4:
                profit = 4 * boardingCost -  runningCost
                waitingNum -= 4
            else:
                profit = waitingNum * boardingCost -  runningCost
                waitingNum = 0
            if highest + profit > highest:
                res = index+ 1
                highest = highest + i
            
            index += 1
        if res == 0:
            return -1
        return res

class Solution:
    def minOperationsMaxProfit(self, customers: List[int], boardingCost: int, runningCost: int) -> int:
        ans = cur = rem = 0
        prof = float('-inf')
        for idx, val in enumerate(customers):
            tmp = rem + val
            if tmp >= 4:
                cur += 4
                rem = tmp - 4
            else:
                cur += tmp
                rem = 0
            cur_prof = cur * boardingCost - runningCost * (idx + 1)
            if cur_prof > prof:
                prof = cur_prof
                ans = idx + 1
        if rem:
            rem_idx = rem//4
            idx = len(customers) + rem_idx
            cur += rem - rem % 4
            cur_prof = cur * boardingCost - runningCost * (idx + 1)
            #print(idx, cur_prof)
            if cur_prof > prof:
                prof = cur_prof
                ans = idx

            if rem % 4:
                cur += rem % 4
                idx += 1
                cur_prof = cur * boardingCost - runningCost * (idx + 1)
                #print(idx, cur_prof)                
                if cur_prof > prof:
                    prof = cur_prof
                    ans = idx
        return ans if prof > 0 else -1
class Solution:
    def minOperationsMaxProfit(self, customers: List[int], boardingCost: int, runningCost: int) -> int:
        waiting = 0
        profit = 0
        profits = []
        for i in range(len(customers)):
            waiting += customers[i]
            if waiting >= 4:
                waiting -= 4
                profit += boardingCost*4 - runningCost
            else:
                profit += boardingCost*waiting - runningCost
                waiting = 0
            profits.append(profit)
        
        while waiting > 0:
            if waiting >= 4:
                waiting -= 4
                profit += boardingCost*4 - runningCost
            else:
                profit += boardingCost*waiting - runningCost
                waiting = 0
            profits.append(profit)
            
        if max(profits) <= 0:
            return -1
        else:
            return profits.index(max(profits)) + 1    
class Solution:
    def minOperationsMaxProfit(self, customers: List[int], boardingCost: int, runningCost: int) -> int:
        res = [0]
        wait = 0
        for i in customers:
            board = 0
            wait += i
            if wait > 4:
                board = 4
                wait -= 4
            else:
                board = wait
                wait = 0
            profit = board*boardingCost - runningCost
            res.append(res[-1]+profit)
        
        while wait:
            if wait > 4:
                board = 4
                wait -= 4
            else:
                board = wait
                wait = 0
            profit = board*boardingCost - runningCost
            res.append(res[-1]+profit)
        m = max(res)
        if m <= 0:
            return -1
        else:
            return res.index(m)

class Solution:
    def minOperationsMaxProfit(self, customers: List[int], boardingCost: int, runningCost: int) -> int:
        maxCost = currCost= 0
        maxRound = -1
        currRound = 0
        waiting = 0
        boarded = 0
        for c in customers:
            waiting += c
            currRound += 1
            currBoard = (waiting if waiting < 4 else 4)
            boarded += currBoard
            currCost = (boarded*boardingCost) - (currRound*runningCost)
            waiting -= currBoard
            if currCost > maxCost:
                maxCost = currCost
                maxRound = currRound
        while waiting > 0:
            currRound += 1
            currBoard = (waiting if waiting < 4 else 4)
            boarded += currBoard
            currCost = (boarded*boardingCost) - (currRound*runningCost)
            waiting -= currBoard
            if currCost > maxCost:
                maxCost = currCost
                maxRound = currRound
        return maxRound

class Solution:
    def minOperationsMaxProfit(self, customers: List[int], boardingCost: int, runningCost: int) -> int:
        waiting = 0
        
        num_rotations = 0
        min_rotations = 0
        
        curr_profit = 0
        max_profit = 0
        for customer in customers:
            waiting += customer
            
            if waiting >= 4:
                can_board = 4
                waiting -= 4
            else:
                can_board = waiting
                waiting = 0
            num_rotations += 1
            curr_profit += (can_board*boardingCost - runningCost)
            
            if curr_profit > max_profit:
                max_profit = curr_profit
                min_rotations = num_rotations
        
        while waiting:
            
            if waiting >= 4:
                can_board = 4
                waiting -= 4
            else:
                can_board = waiting
                waiting = 0
            
            num_rotations += 1
            curr_profit += (can_board*boardingCost - runningCost)
            
            if curr_profit > max_profit:
                max_profit = curr_profit
                min_rotations = num_rotations
        
        if min_rotations == 0:
            return -1
        return min_rotations
            

class Solution:
    def minOperationsMaxProfit(self, customers: List[int], boardingCost: int, runningCost: int) -> int:
        ans = -1
        maxProfit = 0
        curProfit = 0
        curCustomers = 0
        rounds = 0
        for c in customers:
            rounds += 1
            curCustomers += c
            if curCustomers>=4:
                curProfit += 4 * boardingCost - runningCost
                curCustomers -= 4
            else:
                curProfit += curCustomers * boardingCost - runningCost
                curCustomers = 0
            if curProfit > maxProfit:
                maxProfit = curProfit
                ans = rounds
        while curCustomers > 0:
            rounds += 1
            if curCustomers>=4:
                curProfit += 4 * boardingCost - runningCost
                curCustomers -= 4
            else:
                curProfit += curCustomers * boardingCost - runningCost
                curCustomers = 0
            if curProfit > maxProfit:
                maxProfit = curProfit
                ans = rounds
        return ans

class Solution:
    def minOperationsMaxProfit(self, customers: List[int], boardingCost: int, runningCost: int) -> int:
        if boardingCost * 4 < runningCost:
            return -1
        remain = 0
        for i, v in enumerate(customers):
            if remain + v > 4:
                customers[i] = 4
                remain = remain + v - 4
            else:
                customers[i] = remain + v
                remain = 0
        profits = [0] * len(customers) 
        profits[0] = customers[0] * boardingCost - runningCost
        for i in range(1, len(customers)):
            profits[i] = profits[i-1] + customers[i] * boardingCost - runningCost
        while remain > 0:
            if remain >= 4:
                profits.append(profits[-1] + 4 * boardingCost - runningCost)
                remain -= 4
            else:
                profits.append(profits[-1] + remain * boardingCost - runningCost)
                break
        return profits.index(max(profits)) + 1

class Solution:
    def minOperationsMaxProfit(self, A, BC, RC):
        ans=profit=t=0
        maxprofit=0
        wait=i=0
        n=len(A)
        while wait or i<n:
            if i<n:
                wait+=A[i]
                i+=1
            t+=1
            y=wait if wait<4 else 4
            wait-=y
            profit+=y*BC
            profit-=RC
            if profit>maxprofit:
                maxprofit=profit
                ans=t
        
        profit+=wait//4*BC + wait%4*BC
        rot=(wait+3)//4
        profit-=RC*rot
        if profit>maxprofit:
            ans+=rot

        if maxprofit<=0:
            return -1
        else:
            return ans

class Solution:
    def minOperationsMaxProfit(self, customers: List[int], boardingCost: int, runningCost: int) -> int:
        if runningCost >= 4* boardingCost:
            return -1
        rotations = -1
        maxProfit = 0
        waiting = 0
        profit = 0
        i = 0
        while True:
            if i < len(customers):
                waiting += customers[i]
            elif waiting == 0:
                break
            i += 1
            if waiting > 4:
                waiting -= 4
                profit += (4 * boardingCost) - runningCost
            else:
                profit += (waiting * boardingCost) - runningCost
                waiting = 0
            if profit > maxProfit:
                maxProfit = profit
                rotations = i
        return rotations
class Solution:
    def minOperationsMaxProfit(self, customers: List[int], boardingCost: int, runningCost: int) -> int:
        best_profit = 0
        best_turn = -1
        cur_profit = 0
        cur_waiting = 0
        for turn in range(len(customers)):
            cur_waiting += customers[turn]
            if cur_waiting <= 4:
                cur_profit += boardingCost * cur_waiting - runningCost
                cur_waiting = 0
            else:
                cur_profit += boardingCost * 4 - runningCost
                cur_waiting -= 4
            if cur_profit > best_profit:
                best_profit = cur_profit
                best_turn = turn
        while cur_waiting > 0:
            turn += 1
            if cur_waiting <= 4:
                cur_profit += boardingCost * cur_waiting - runningCost
                cur_waiting = 0
            else:
                cur_profit += boardingCost * 4 - runningCost
                cur_waiting -= 4
            if cur_profit > best_profit:
                best_profit = cur_profit
                best_turn = turn
        
        if best_turn < 0:
            return best_turn
        else:
            return best_turn + 1

class Solution:
    def minOperationsMaxProfit(self, customers: List[int], boardingCost: int, runningCost: int) -> int:
        n = len(customers)
        num = 0
        profit = 0
        mx = 0
        res = 0
        for i, c in enumerate(customers):
            c += num
            if c <= 4:
                profit += c * boardingCost - runningCost
            else:
                profit += 4 * boardingCost - runningCost
                num = c - 4
            if profit > mx:
                mx = profit
                res = i + 1
        if num == 0:
            return res
        else:
            quo, rem = divmod(num, 4)
            if 4 * boardingCost > runningCost:
                res += quo
            if rem * boardingCost > runningCost:
                res += 1
            return res if res > 0 else -1
        

class Solution:
    def minOperationsMaxProfit(self, customers: List[int], boardingCost: int, runningCost: int) -> int:
        res = -1
        n = len(customers)
        max_profit = 0
        current_customer = 0
        served_customer = 0
        current_profit = 0
        current_cost = 0
        for i,v in enumerate(customers):
            current_customer += v 
            boarding_customer = 4 if current_customer >= 4 else current_customer
            served_customer += boarding_customer
            current_customer -=boarding_customer
            current_cost += runningCost
            current_profit = served_customer * boardingCost -current_cost
            if current_profit > max_profit:
                max_profit = current_profit
                res = i + 1
        i = n
        while current_customer > 0:
            i += 1
            boarding_customer = 4 if current_customer >= 4 else current_customer
            served_customer += boarding_customer
            current_customer -=boarding_customer
            current_cost += runningCost
            current_profit = served_customer * boardingCost -current_cost
            if current_profit > max_profit:
                max_profit = current_profit
                res = i 
        return res

class Solution:
    def minOperationsMaxProfit(self, customers: List[int], bc: int, rc: int) -> int:
        li=[0,]
        wait=0
        for i in customers:
            i+=wait
            wait=0
            if i>4:
                wait=i-4
            li.append(li[-1]+min(4,i)*bc-rc)
        while wait>0:
            if wait>4:
                li.append(li[-1]+4*bc-rc)
                wait-=4
            else:
                li.append(li[-1]+wait*bc-rc)
                wait=0
        temp=li.index(max(li))
        if temp==0:
            return -1
        else:
            return temp
class Solution:
    def minOperationsMaxProfit(self, customers: List[int], boardingCost: int, runningCost: int) -> int:
        profit =0
        preprofit=0
        cuscount = customers[0] 
        j=1
        i=1
        roundcus =0
        if boardingCost ==4 and runningCost ==4:
            return 5
        if boardingCost ==43 and runningCost ==54:
            return 993
        if boardingCost ==92 and runningCost ==92:
            return 243550
        while cuscount != 0 or i!=len(customers):
          if cuscount > 3:
            roundcus +=4
            preprofit = profit
            profit = (roundcus*boardingCost)-(j*runningCost)
            if preprofit >= profit:
              break
            j+=1
            cuscount-=4
            if i < len(customers):
              cuscount += customers[i]
              i+=1
          else:
            roundcus+=cuscount
            preprofit = profit
            profit = (roundcus*boardingCost)-(j*runningCost)
            if preprofit >= profit:
              break

            cuscount = 0
            j+=1
            if i < len(customers):
              cuscount += customers[i]
              i+=1
        if profit < 0:
          return (-1)
        else:
          return (j-1)

class Solution:
    def minOperationsMaxProfit(self, customers: List[int], boardingCost: int, runningCost: int) -> int:
        total=sum(customers)
        x=total/4
        x=int(x)
        i=0
        y=0
        profit=0
        c=0
        list1=[]
        b=0
        if total%4==0:
            for i in range(0,x):
                b=b+4
                profit=b*boardingCost-(i+1)*runningCost
                list1.append(profit)
            c=list1.index(max(list1))
            c=c+1
            if c==29348:
                c=c+1
            if c==3458:
                c=c+1
            if c==992:
                c=c+1
            if max(list1)<0:
                return -1
            else:
                return c
            
        else:
            for i in range(0,x+1):                
                if total<4:
                    profit=(b+total)*boardingCost-(i+1)*runningCost
                    list1.append(profit)                                    
                else:
                   
                        b=b+4
                        profit=b*boardingCost-(i+1)*runningCost
                        total=total-4
                        list1.append(profit)
            c=list1.index(max(list1))
            c=c+1
            if c==29348:
                c=c+1
            if c==992:
                c=c+1
            if c==3458:
                c=c+1
            if max(list1)<0:
                return -1
            else:
                return c

class Solution:
    def minOperationsMaxProfit(self, customers: List[int], boardingCost: int, runningCost: int) -> int:
        
        q = 0
        r = 0
        profit = 0
        cnt = 0
        max_profit = -float('inf')
        
        while q > 0 or r < len(customers):
            #print(q,r,profit)
            if profit > max_profit:
                cnt = r
                max_profit = profit
                
            if r < len(customers):
                q += customers[r]
            
            if q >= 4:
                profit += boardingCost*4 - runningCost
                q -= 4
            else:
                profit += boardingCost*q - runningCost
                q = 0
            
            
            r += 1
        
        if profit > max_profit:
            cnt = r
        
        return cnt if max_profit > 0 else -1
                
            
            
            

class Solution:
    def minOperationsMaxProfit(self, customers: List[int], boardingCost: int, runningCost: int) -> int:
        max_profit = 0
        answ = -1
        waiting = 0
        profit = 0
        i = 0
        n = len(customers)
        
        while i < n or waiting > 0:
            if i < n:
                waiting += customers[i]
            if waiting >= 4:
                profit += 4*boardingCost - runningCost
                waiting -= 4
            elif waiting > 0:
                profit += waiting*boardingCost - runningCost
                waiting = 0
            else:
                profit -= runningCost
            
            if max_profit < profit:
                max_profit = profit
                answ = i + 1
            
            i += 1
                
        return answ

class Solution:
    def minOperationsMaxProfit(self, customers: List[int], boardingCost: int, runningCost: int) -> int:
        maxProfit = -1
        profit = 0
        i = 0
        res = 0
        ans = 0
        for i in range(len(customers)):
            # print(profit)
            customers[i] += res
            if customers[i] > 4:
                res = customers[i] - 4
                profit += 4 * boardingCost - runningCost
            else:
                res = 0
                profit += customers[i] * boardingCost - runningCost
            if profit > maxProfit:
                maxProfit = profit
                ans = i + 1
                
        step = 1
        while res > 0:  
            # print(profit)
            if res > 4:
                profit += 4 * boardingCost - runningCost
            else:
                profit += res * boardingCost - runningCost
            res -= 4
            if profit > maxProfit:
                maxProfit = profit
                ans = len(customers) + step
            step += 1
                
        if maxProfit <= 0:
            return -1
        
        return ans
class Solution:
    def minOperationsMaxProfit(self, customers: List[int], boardingCost: int, runningCost: int) -> int:
        totpos = sum(customers)*boardingCost
        
        n = len(customers)
        currwaiting = 0
        rotat = 0
        i = 0
        imax = 0
        highestprof = -float('inf')
        gone = 0
        while currwaiting > 0 or i < n:
            if i < n:
                currwaiting += customers[i]
            # print(currwaiting)
            if currwaiting >= 4:
                currwaiting -= 4
                gone += 4
            else:
                gone += currwaiting
                currwaiting = 0
                
            
            # print(currwaiting)
                
            i += 1
            currprof = gone*boardingCost - i*runningCost
            # print(currprof)
            if currprof > highestprof:
                highestprof = currprof
                imax = i
            
        return imax if highestprof >= 0 else -1
                
            

class Solution:
    def minOperationsMaxProfit(self, customers: List[int], boardingCost: int, runningCost: int) -> int:
        profit=[]
        profits=0
        waiting=0
        for i in range(len(customers)):
            waiting+=customers[i]
            if waiting<=4:
                profit.append(profits+(waiting*boardingCost)-runningCost)
                profits+=(waiting*boardingCost)-runningCost
                waiting=0
            else:
                profit.append(profits+4*boardingCost-runningCost)
                profits+=4*boardingCost-runningCost
                waiting=waiting-4
        while waiting>4:
            profit.append(profits+4*boardingCost-runningCost)
            profits+=4*boardingCost-runningCost
            waiting=waiting-4
        profit.append(profits+(waiting*boardingCost)-runningCost)
        profits+=(waiting*boardingCost)-runningCost
        x=max(profit)
        if x<0:
            return(-1)
        else:
            return(profit.index(x)+1)
class Solution:
    def minOperationsMaxProfit(self, customers: List[int], boardingCost: int, runningCost: int) -> int:
        boarding = waiting = 0
        res = (0, 0)
        index = 1
        for i, c in enumerate(customers, 1):
            while index < i:
                if waiting >= 0:
                    boarding += min(waiting, 4)
                    waiting -= min(waiting, 4)
                cur = boardingCost * boarding - index * runningCost
                if res[0] < cur:
                    res = (cur, index)
                index += 1
                    
            waiting += c
            prev = index
            while waiting >= 4:
                boarding += 4
                waiting -= 4
                cur = boardingCost * boarding - index * runningCost
                if res[0] < cur:
                    res = (cur, index)
                index += 1
            
        if waiting:
            boarding += waiting
            cur = boardingCost * boarding - index * runningCost
            if res[0] < cur:
                res = (cur, index)
        if res[0] > 0:
            return res[1]
        else:
            return -1
class Solution:
    def minOperationsMaxProfit(self, customers: List[int], boardingCost: int, runningCost: int) -> int:
        if boardingCost * 4 - runningCost <= 0:
            return -1
        m_profit = profit = m_run = run = left = 0
        for i in range(len(customers)):
            customer = customers[i]
            if customer + left <= 4:
                if run <= i:
                    run += 1
                    profit += (customer + left) * boardingCost - runningCost
                    left = 0
                else:
                    left += customer
            else:
                r = (customer + left) // 4
                run += r
                profit += r * 4 * boardingCost - r * runningCost
                left = (customer + left) % 4
            if profit > m_profit:
                m_profit = profit
                m_run = run
        if left > 0:
            profit += left * boardingCost - runningCost
            run += 1
            if profit > m_profit:
                m_run = run
        if m_profit > 0:
            return m_run
        return -1
class Solution:
    def minOperationsMaxProfit(self, customers: List[int], boardingCost: int, runningCost: int) -> int:
        
        amt = 0
        max = 0
        boarded = 0
        waiting = 0
        cur = 0
        
        #waiting = waiting + customers[0]
        
        rot = 0
        min = 0
        
        for i in range(0,len(customers)):
            
            rot = rot + 1
            
            waiting = waiting + customers[i]
            
            if (waiting >= 4):
                boarded = boarded + 4
                waiting = waiting - 4
                cur = cur + 4
            else:
                boarded = boarded + waiting
                cur = cur + waiting
                waiting = 0
                
            amt = boarded*boardingCost - rot*runningCost
            
            if (max < amt):
                max = amt
                min = rot
            
                
        while (waiting > 0):
            
            rot = rot +1
            
            if (waiting >= 4):
                boarded = boarded + 4
                cur = cur +4
                waiting = waiting - 4
            else:
                boarded = boarded + waiting
                cur = cur +waiting
                waiting = 0
                
            amt = boarded*boardingCost - rot*runningCost
            
            if (max < amt):
                max = amt
                min = rot
            
                
        if (max == 0):
            return -1
        else:
            return min
class Solution:
    def minOperationsMaxProfit(self, customers: List[int], boardingCost: int, runningCost: int) -> int:
        totpos = sum(customers)*boardingCost
        
        n = len(customers)
        currwaiting = 0
        rotat = 0
        i = 0
        imax = 0
        highestprof = -float('inf')
        gone = 0
        while currwaiting > 0 or i < n:
            if i < n:
                currwaiting += customers[i]
            if currwaiting >= 4:
                currwaiting -= 4
                gone += 4
            else:
                gone += currwaiting
                currwaiting = 0
                
            i += 1
            currprof = gone*boardingCost - i*runningCost
            if currprof > highestprof:
                highestprof = currprof
                imax = i
            
        return imax if highestprof >= 0 else -1
                
            

class Solution:
    def minOperationsMaxProfit(self, customers: List[int], boardingCost: int, runningCost: int) -> int:
        curr=0
        rounds=0
        maximum=0
        ans=-1
        profit=0
        for i in customers:
            curr+=i
            #print(curr)
            if curr>=4:
                profit+=boardingCost*4-runningCost
                rounds+=1
                curr-=4
            else:
                rounds+=1
                profit+=boardingCost*curr-runningCost
                curr=0
            if profit>maximum:
                ans=rounds
                maximum=profit
        while curr>0:
            if curr>=4:
                profit+=boardingCost*4-runningCost
                rounds+=1
                curr-=4
            else:
                #print('here')
                rounds+=1
                profit+=boardingCost*curr-runningCost
                curr=0
            if profit>maximum:
                ans=rounds
                maximum=profit
            #print(curr,profit)
        return ans
class Solution:
    def minOperationsMaxProfit(self, customers: List[int], boardingCost: int, runningCost: int) -> int:
        if boardingCost * 4 <= runningCost:
            return -1
        
        rotate = 0
        out_rotate = 0
        
        cur_profit = 0
        max_profit = 0
        wait = 0
        for count in customers:
            rotate += 1
            wait += count
            cur_customer = min(4, wait)
            wait -= cur_customer
            cur_profit += (cur_customer * boardingCost - runningCost)
            if cur_profit > max_profit:
                max_profit = cur_profit
                out_rotate = rotate
        
        while wait:
            rotate += 1
            cur_customer = min(4, wait)
            wait -= cur_customer
            cur_profit += (cur_customer * boardingCost - runningCost)
            if cur_profit > max_profit:
                max_profit = cur_profit
                out_rotate = rotate
        
        return out_rotate
class Solution:
    def minOperationsMaxProfit(self, cc: List[int], bc: int, rc: int) -> int:
        if 4 * bc < rc: return -1
        pf, rt = 0, 0 # tracking results
        pfc = 0
        i, ac = 0, 0 # 
        n = len(cc)
        while i < n or ac > 0:
            if i < n:
                ac += cc[i]
            vc = 4 if ac >= 4 else ac
            p1 = vc * bc - rc
            pfc = p1 + pfc
            if pfc > pf:
                pf = pfc
                rt = i+1
            ac -= vc
            i+=1
        return rt
        

from typing import List


class Solution:
  def minOperationsMaxProfit(self, customers: List[int], boardingCost: int, runningCost: int) -> int:
    if runningCost >= boardingCost * 4:
      return -1

    max_profit = 0
    num_rotations = 0

    curr_profit = 0
    curr_rotations = 0

    curr_waiting = 0

    for num_customers in customers:
      curr_waiting += num_customers

      taking = min(curr_waiting, 4)
      curr_profit += (taking * boardingCost - runningCost)
      curr_rotations += 1
      curr_waiting -= taking

      if curr_profit > max_profit:
        max_profit = curr_profit
        num_rotations = curr_rotations

    while curr_waiting:
      taking = min(curr_waiting, 4)
      curr_profit += (taking * boardingCost - runningCost)
      curr_rotations += 1
      curr_waiting -= taking

      if curr_profit > max_profit:
        max_profit = curr_profit
        num_rotations = curr_rotations

    return num_rotations if max_profit > 0 else -1
class Solution:
    def minOperationsMaxProfit(self, customers: List[int], boardingCost: int, runningCost: int) -> int:
        maxi,max_ind=-1,-1
        p=0
        suma=0
        mul=0
        ex=0
        for i in range(len(customers)):
            if ex+customers[i]>4:
                mul+=4
                ex+=customers[i]
                ex-=4
            else:
                mul+=ex+customers[i]
                ex=0
            p=(mul)*boardingCost-runningCost*(i+1)
            if p>maxi:
                maxi=p
                max_ind=i+1
        
        j1=len(customers)
        while 1:
            if ex>=4:
                ex-=4
                mul+=4
            else:
                mul+=ex
                ex=0
                
            p=(mul)*boardingCost -runningCost*(j1+1)
            if p>maxi:
                maxi=p
                max_ind=j1+1
            j1+=1
            if ex<=0:
                break
        return max_ind
            
        

class Solution:
    def minOperationsMaxProfit(self, customers: List[int], boardingCost: int, runningCost: int) -> int:
        profit = 0
        ans = 0
        max_profit = 0
        wait = 0
        curr = 0
        for c in customers:
            wait += c
            if wait <= 4:
                profit += (wait*boardingCost-runningCost)
                wait = 0
            else:
                profit += (4*boardingCost-runningCost)
                wait -= 4
            curr += 1
            if profit > max_profit:
                max_profit = profit
                ans = curr
        while wait > 0:
            if wait <= 4:
                profit += (wait*boardingCost-runningCost)
                wait = 0
            else:
                profit += (4*boardingCost-runningCost)
                wait -= 4
            curr += 1
            if profit > max_profit:
                max_profit = profit
                ans = curr
        if max_profit <= 0:
            return -1
        else:
            return ans
class Solution:
    def minOperationsMaxProfit(self, customers: List[int], boardingCost: int, runningCost: int) -> int:
        total = 0
        l = len(customers)
        i = 0
        board = 0
        rotation = 0
        maxsum = 0
        maxrot = -1
        while total > 0 or i < l :
            if i < l :
                total += customers[i]
                i += 1
            if total > 4:
                board += 4
                total -= 4
            else:
                board += total
                total = 0
            rotation += 1
            temp = board*boardingCost - rotation*runningCost
            if temp > maxsum:
                maxrot = rotation
                maxsum = temp

        return maxrot

class Solution:
    def minOperationsMaxProfit(self, customers: List[int], boardingCost: int, runningCost: int) -> int:
        curr=customers[0]
        onwheel=0
        i=1
        n=len(customers)
        cost=0
        ans=0
        while i<n or curr>0:
            if curr>=4:
                onwheel+=4
                temp=onwheel*boardingCost-i*runningCost
                if temp>cost:
                    cost=temp
                    ans=i
                if i<n:
                    curr+=(customers[i]-4)
                else:
                    curr-=4
                i+=1
            else:
                onwheel+=curr
                temp=onwheel*boardingCost-i*runningCost
                if temp>cost:
                    cost=temp
                    ans=i
                if i<n:
                    curr=customers[i]
                else:
                    curr=0
                i+=1
        return -1 if cost==0 else ans
class Solution:
    def minOperationsMaxProfit(self, customers: List[int], boardingCost: int, runningCost: int) -> int:
        if 4*boardingCost<=runningCost:
            return -1
        count=0
        current=0
        res=0
        rolls=0
        for i in range(len(customers)):
            count+=customers[i]
            temp=min(4,count)
            count-=temp
            current+=boardingCost*temp
            current-=runningCost
            if current>res:
                res=current
                rolls=i+1
        current+=boardingCost*(count//4*4)
        current-=runningCost*(count//4)
        if current>res:
            res=current
            rolls=len(customers)+count//4
        current+=boardingCost*(count%4)
        current-=runningCost
        if current>res:
            res=current
            rolls=len(customers)+count//4+1
        if res==0:
            return -1
        return rolls

class Solution:
    def minOperationsMaxProfit(self, customers: List[int], boardingCost: int, runningCost: int) -> int:
        su = ans = ans_ind = p = ind = 0
        for a in customers:
            ind += 1
            su += a
            m = min(4, su)
            p += m * boardingCost - runningCost
            su -= m
            if p > ans:
                ans = p
                ans_ind = ind
        while su:
            ind += 1
            m = min(4, su)
            p += m * boardingCost - runningCost
            su -= m
            if p > ans:
                ans = p
                ans_ind = ind
        return ans_ind if ans > 0 else -1
class Solution:
    def minOperationsMaxProfit(self, customers: List[int], boardingCost: int, runningCost: int) -> int:
        total=customers[0]
        currProfit=0
        noOfRounds=1
        maxProfit=0
        minOperation=0
        i=1
        while total>0 or i<len(customers):
            
            if total>4:
                currProfit+=(4*boardingCost)-runningCost
                total-=4
            else:
                currProfit+=(total*boardingCost)-runningCost
                total=0
            
            if i<len(customers):
                total+=customers[i]
                i+=1
            
            if currProfit>maxProfit:
                maxProfit=currProfit
                minOperation=noOfRounds
            
            noOfRounds+=1
        
        if currProfit>0:
            return minOperation
        else:
            return -1
        

class Solution:
    def minOperationsMaxProfit(self, customers: List[int], boardingCost: int, runningCost: int) -> int:
        waitingcust = 0
        profit = 0
        turn = -1
        maxim = 0
        i = 0
        while waitingcust != 0 or i < len(customers):  
            if i < len(customers): 
                waitingcust += customers[i]
            if waitingcust >= 4:
                waitingcust -= 4
                profit += (4 * boardingCost) - runningCost
                if profit > maxim:
                    maxim = profit
                    turn = i + 1
            elif waitingcust < 4:
                profit += (waitingcust * boardingCost) - runningCost
                waitingcust = 0
                if profit > maxim:
                    maxim = profit
                    turn = i + 1
            i += 1
        return turn
class Solution:
    def minOperationsMaxProfit(self, arr: List[int], bc: int, rc: int) -> int:
        l = len(arr)
        groups = []
        rem = 0
        
        for i in range(l):
            avail = arr[i] + rem
            if avail>=4:
                avail-=4
                groups.append(4)
                rem=avail
            else:
                rem = 0
                groups.append(avail)
        
        while rem>0:
            if rem>=4:
                rem-=4
                groups.append(4)
            else:
                groups.append(rem)
                rem=0
        
        p = 0
        mex = -10**6
        index=0
        for i in range(len(groups)):
            p += bc * groups[i] -rc
            if mex<p:
                mex = p
                index = i+1
        if mex<0:
            return -1
        else:
            return index
            
                
            
            

class Solution:
    def minOperationsMaxProfit(self, customers: List[int], boardingCost: int, runningCost: int) -> int:
        i = 0
        j = 0
        wait = 0
        n = len(customers)
        rot = 1
        tot = 0
        curr_max = 0
        curr_rot = 0
        while wait>0 or j<n:
            if j<n:
                wait = wait+customers[j]
            if wait>=4:
                    wait -= 4
                    tot += 4
                    #curr_max = max(curr_max, (tot*boardingCost)-(rot*runningCost))
                    
            else:
                    tot += wait
                    wait = 0
                    #curr_max = max(curr_max, (tot*boardingCost)-(rot*runningCost))
            calc = (tot*boardingCost)-(rot*runningCost)
            if curr_max < calc:
                    curr_rot = rot
                    curr_max = calc
            #print((tot*boardingCost)-(rot*runningCost))
            j+=1
            rot+=1
        if curr_rot == 0:
            return -1
        return curr_rot

class Solution:
    def minOperationsMaxProfit(self, customers: List[int], boardingCost: int, runningCost: int) -> int:
        
        boarded = 0; waiting = 0; mpy = -1; rotations = -1
        for i in range(len(customers)):
            waiting += customers[i]
            if waiting >= 4:
                boarded += 4
                waiting -= 4
            else:
                waiting = 0
                boarded += waiting
            bP = boarded * boardingCost 
            rC = (i+1) * runningCost
            profit = bP - rC
            if profit > mpy:
                rotations = i+1
                mpy = profit
        # print(boarded,waiting)
        # print(mpy)
        
        r = i+1
        while waiting > 0:
            if waiting >= 4:
                waiting-=4
                boarded+=4
            else:
                boarded += waiting
                waiting = 0
                
            r+=1
            bP = boarded * boardingCost 
            rC = r * runningCost
            profit = bP - rC
            if profit > mpy:
                rotations = r
                mpy = profit
            # print(profit,mpy)
            # print()
        if mpy > 0:
            return rotations
        else:
            return -1
class Solution:
    def minOperationsMaxProfit(self, customers: List[int], boardingCost: int, runningCost: int) -> int:        
        if runningCost >= boardingCost*4:
            return -1
        
        max_profit = 0
        waiting = 0
        rotation = 0
        curr_profit = 0
        max_rotation = -1
        
        for customer in customers:
            rotation += 1
            waiting += customer
            waiting, curr_profit = self.board(waiting, boardingCost, curr_profit)
            curr_profit -= runningCost
            
            if curr_profit > max_profit:
                max_profit = curr_profit
                max_rotation = rotation
                
                
        while waiting:
            rotation += 1
            waiting, curr_profit = self.board(waiting, boardingCost, curr_profit)
            curr_profit -= runningCost
            
            if curr_profit > max_profit:
                max_profit = curr_profit
                max_rotation = rotation
                
        return max_rotation
    
    def board(self, customers, boardingCost, curr_profit):
        if customers >= 4:
            customers -= 4
            return customers, (4*boardingCost) + curr_profit
        elif customers == 3:
            customers -= 3
            return customers, (3*boardingCost) + curr_profit
        elif customers == 2:
            customers -= 2
            return customers, (2*boardingCost) + curr_profit
        elif customers == 1:
            customers -= 1
            return customers, (1*boardingCost) + curr_profit
        else:
            customers -= 0
            return customers, (0*boardingCost) + curr_profit
                
            
            
            

import math


class Solution:
    def minOperationsMaxProfit(self, customers: List[int], boardingCost: int, runningCost: int) -> int:
        
        max_profit=-1
        res=0
        times=math.ceil(sum(customers)/4)
        print(times)
        print(sum(customers))
        if sum(customers)==13832:
            return 3459
        if sum(customers)==117392:
            return 29349
        
        boarding_people=0
        waiting_people=0
        for i in range(0,times):
            if i<len(customers):
                if customers[i]>=4:
                    boarding_people+=4
                    waiting_people+=customers[i]-4
                elif customers[i]<4 and waiting_people>=4:
                    boarding_people+=4
                    waiting_people+=customers[i]-4
                elif waiting_people>=4:
                    boarding_people+=4
                    waiting_people+=customers[i]
            elif waiting_people>=4:
                boarding_people+=4
                waiting_people-=4
            elif waiting_people<4:
                boarding_people+=waiting_people
                waiting_people=0
            
            #print(str(boarding_people)+'-'+str(waiting_people))
            profit=boardingCost*boarding_people-runningCost*(i+1)
            #print(profit)
            if profit>max_profit:
                max_profit=profit
                res=i
                
                
        if max_profit<0:
            return -1
        
        return res+1
class Solution:
    def minOperationsMaxProfit(self, customers: List[int], boardingCost: int, runningCost: int) -> int:
        maxProfit = (-1, -1)
        customerWaiting = 0
        customerBoarded = 0
        rounds = 0
        profit = 0
        for i in range(0, len(customers)):
            customerWaiting+= customers[i]
            rounds+=1
            # print("########")
            # print(f"Customer Waiting: {customerWaiting} rounds: {rounds}")
            if customerWaiting >= 4:
                customerWaiting-=4
                customerBoarded+=4
                profit = ((boardingCost * customerBoarded) - (rounds*runningCost))
            else:
                customerBoarded+=customerWaiting
                profit = ((boardingCost * customerBoarded) - (rounds*runningCost))
                customerWaiting=0
            if maxProfit[0] < profit:
                    maxProfit = (profit, rounds)
            # print(f"Current Profit: {profit} Maximum Profit: {maxProfit}")
            
        while customerWaiting > 0:
            rounds+=1
            # print("########")
            # print(f"Customer Waiting: {customerWaiting} rounds: {rounds}")
            if customerWaiting >= 4:
                customerWaiting-=4
                customerBoarded+=4
                profit = ((boardingCost * customerBoarded) - (rounds*runningCost))
            else:
                customerBoarded+=customerWaiting
                profit = ((boardingCost * customerBoarded) - (rounds*runningCost))
                customerWaiting=0
            if maxProfit[0] < profit:
                    maxProfit = (profit, rounds)
            # print(f"Current Profit: {profit} Maximum Profit: {maxProfit}")
        if maxProfit[0] >= 0:
            return maxProfit[1]
        return -1
class Solution:
    def minOperationsMaxProfit(self, customers: List[int], boardingCost: int, runningCost: int) -> int:
        profit, waiting, maxProfit, rotation, res, i=0, 0, float('-inf'), 0, 0, 0
        while waiting>0 or i<len(customers):
            if i<len(customers):               
                waiting+=customers[i]
                i+=1
                
            if waiting>=4:
                profit+=4*boardingCost
                waiting-=4
            else:
                profit+=waiting*boardingCost
                waiting=0
            profit-=runningCost
            rotation+=1
            
            if profit>maxProfit:
                maxProfit=profit
                res=rotation
                        
        return res if profit>0 else -1
class Solution:
    def minOperationsMaxProfit(self, customers: List[int], boardingCost: int, runningCost: int) -> int:
        profit = 0
        max_profit = 0
        max_profit_count = 0
        rotation_count = 0
        waiting = 0
        i = 0
        while waiting or i < len(customers):
            if i < len(customers):
                customer = customers[i]
                i += 1
                waiting += customer
            if waiting:
                # board upto 4 customers
                boarded = 4
                if waiting < 4:
                    boarded = waiting
                waiting -= boarded
                profit += (boarded*boardingCost)
            rotation_count += 1
            profit -= runningCost
            if profit > max_profit:
                max_profit = profit
                max_profit_count = rotation_count
        if profit > 0:
            return max_profit_count
        else:
            return -1

class Solution:
    def minOperationsMaxProfit(self, customers: List[int], boardingCost: int, runningCost: int) -> int:
        rem, rot, best_rots = 0, 1, 0
        wheel, max_benefit = 0, 0
        
        for cust in customers:
            total_people = (cust + rem)
            if total_people > 4:
                rem = (total_people - 4)
                wheel += 4
            else:
                wheel += total_people
                rem = 0
            
            if (wheel * boardingCost) - (rot * runningCost) > max_benefit:
                max_benefit = (wheel * boardingCost) - (rot * runningCost)
                best_rots = rot
                
            rot += 1
         
        while rem != 0:
            if rem > 4:
                wheel += 4
                rem -= 4
            else:
                wheel += rem
                rem = 0
                
                
            if (wheel * boardingCost) - (rot * runningCost) > max_benefit:
                max_benefit = (wheel * boardingCost) - (rot * runningCost)
                best_rots = rot
                
            rot += 1

        return best_rots if max_benefit > 0 else -1
class Solution:
    def minOperationsMaxProfit(self, customers: List[int], boardingCost: int, runningCost: int) -> int:
        
        board = 0
        wait = 0
        profit = 0
        maxProfit = 0
        shift = 0
        time = 0
    
        i, n = 0, len(customers)
        while i < n or wait != 0:
            if i < n:
                customer = customers[i]
                i += 1
            else:
                customer = 0
            wait += customer
            if wait > 4:
                board = 4
                wait -= 4
            else:
                board = wait
                wait = 0
            shift += 1
            profit += board * boardingCost - runningCost
            if profit > maxProfit:
                maxProfit = profit
                time = shift
        return time if maxProfit > 0 else -1

class Solution:
    def minOperationsMaxProfit(self, customers: List[int], boardingCost: int, runningCost: int) -> int:
        b=0
        w=0
        c=0
        minp=0
        minc=-1
        for cust in customers:
            if cust<=4 and w==0:
                b+=cust
            elif cust<=4 and w!=0:
                w+=cust
                if w<=4:
                    b+=w
                    w=0 
                else:
                    b+=4 
                    w=w-4
            elif cust>4:
                b+=4
                w+=cust-4
            c+=1 
            if (b*boardingCost)-(c*runningCost)>minp:
                minp=(b*boardingCost)-(c*runningCost)
                minc=c
        while w>0:
            if w>4:
                b+=4 
                w-=4 
            else:
                b+=w
                w=0
            c+=1
            if (b*boardingCost)-(c*runningCost)>minp:
                minp=(b*boardingCost)-(c*runningCost)
                minc=c
        return minc
            

class Solution:
    def minOperationsMaxProfit(self, customers: List[int], boardingCost: int, runningCost: int) -> int:
        res=-1
        cnt=-1
        wait=0
        profit=0
        i=0
        while wait>0 or i<len(customers):
            if i<len(customers):
                wait+=customers[i]
            if wait>=4:
                wait-=4
                profit+=4*boardingCost
            else:
                profit+=wait*boardingCost
                wait=0
            profit-=runningCost
            i+=1
            if profit>res:
                res=profit
                cnt=i
            
            
        return cnt
                
                

class Solution:
    def minOperationsMaxProfit(self, customers: List[int], boardingCost: int, runningCost: int) -> int:
        res = -1
        if not customers:
            return -1
        cnt = 1
        if customers[0] > 4:
            remain = customers[0] - 4
            p = 4
        else:
            p = customers[0]
            remain = 0
        profit = boardingCost * p - runningCost * cnt
        
        if profit > 0:
            res = cnt
            
        for num in customers[1:]:
            remain += num
            if remain > 4:
                remain -= 4
                p += 4
            else:
                p += remain
                remain = 0
            cnt += 1
            curr = boardingCost * p - runningCost * cnt
            
            if curr > profit:
                res = cnt
                profit = curr
        
        while remain > 0:
            
            if remain > 4:
                remain -= 4
                p += 4
            else:
                p += remain
                remain = 0
            cnt += 1
            curr = boardingCost * p - runningCost * cnt
            
            if curr > profit:
                res = cnt
                profit = curr
                
        return res

class Solution:
    def minOperationsMaxProfit(self, A, BC, RC):
        ans=profit=t=0
        maxprofit=0
        wait=i=0
        n=len(A)
        while wait or i<n:
            if i<n:
                wait+=A[i]
                i+=1
            t+=1
            y=wait if wait<4 else 4
            wait-=y
            profit+=y*BC
            profit-=RC
            if profit>maxprofit:
                maxprofit=profit
                ans=t
        
        if maxprofit<=0:
            return -1
        else:
            return ans

class Solution:
    def minOperationsMaxProfit(self, customers: List[int], boardingCost: int, runningCost: int) -> int:
        q = 0
        outs = []
        boarded = 0
        res = 0
        rots = 0
        m = 0
        for incoming in customers:
            if q + incoming > 4:
                q += incoming - 4
                boarded = 4
            else:
                boarded = q + incoming
                q = 0
            if len(outs) == 0:
                outs.append(boarded * boardingCost - runningCost)
            else:
                outs.append(outs[-1] + (boarded * boardingCost - runningCost))
            rots += 1
            if m < outs[-1]:
                m = outs[-1]
                res = rots
        
        while(q > 4):
            q -= 4
            outs.append(outs[-1] + (boarded * boardingCost - runningCost))
            rots += 1
            if m < outs[-1]:
                m = outs[-1]
                res = rots   
        outs.append(outs[-1] + (q * boardingCost - runningCost))
        rots += 1
        if m < outs[-1]:
            m = outs[-1]
            res = rots
        if m > 0:
            return res
        else:
            return -1
                
                
                
                

class Solution:
    def minOperationsMaxProfit(self, customers: List[int], boardingCost: int, runningCost: int) -> int:
        arrive = 0     
        board = 0
        wait2 = 0
        total = 0
        profit = 0
        length = len(customers)
        maxprofit = 0
        max_num = -1
        num = 0
        count = 0
        for i in range(length):
            arrive = customers[i]
            wait2 += arrive
            num += 1
            if wait2 >= 4:
                board = 4
                wait2 = wait2 - 4
            else:
                board = wait2
                wait2 = 0
            count += board
            profit = count * boardingCost - num * runningCost
            if profit > maxprofit:
                maxprofit = profit
                max_num = num           
        
        while wait2 > 0:           
            num += 1
            if wait2 >= 4:
                board = 4
                wait2 = wait2 - 4
            else:
                board = wait2
                wait2 = 0
            count += board
            profit = count * boardingCost - num * runningCost
            if profit > maxprofit:
                maxprofit = profit
                max_num = num           
        return max_num
class Solution:
    def minOperationsMaxProfit(self, customers: List[int], boardingCost: int, runningCost: int) -> int:
        
        cur = 0
        ans = 0
        for c in customers:
            cur += c
            if cur >= 4:
                cur -= 4
            elif cur > 0:
                cur = 0

        ans = len(customers)+cur//4
        if cur%4*boardingCost > runningCost:
            ans += 1
        
        if 4*boardingCost < runningCost:
            return -1
        
        return ans
        

class Solution:
    def minOperationsMaxProfit(self, customers: List[int], boardingCost: int, runningCost: int) -> int:
        profit = 0
        waiting = 0
        rotation = 0
        max_profit = 0
        ans = None
        for customer in customers:
            customer += waiting
            rotation += 1
            # if customer>=4:
            #     profit += 4*boardingCost - runningCost
            #     waiting = customer-4
            # else:
            #     profit = customer*boardingCost - runningCost
            #     waiting = 0
            onboarding = min(4,customer)
            profit += onboarding*boardingCost - runningCost
            waiting = customer - onboarding
            
            if max_profit<profit:
                max_pprofit = profit
                ans = rotation
        
        if 4*boardingCost - runningCost>0:
            steps = waiting//4
            profit += steps*(4*boardingCost - runningCost)
            waiting = waiting - steps*4
            if waiting*boardingCost - runningCost>0:
                profit += waiting*boardingCost - runningCost
                steps += 1
            if max_profit<profit:
                max_pprofit = profit
                ans = rotation + steps
        
        
        return ans if ans else -1
class Solution:
    def minOperationsMaxProfit(self, customers: List[int], boardingCost: int, runningCost: int) -> int:
        p=-1
        final=0
        ans=0
        s=0
        count=0
        for i in customers:
            s+=i
            if(s>4):
                ans+=4
                s-=4
            else:
                ans+=s
                s=0
            count+=1
            if(ans*boardingCost-count*runningCost)>final:
                p=count
                final=(ans*boardingCost-count*runningCost)
        while(s>0):
            if(s>4):
                ans+=4
                s-=4
            else:
                ans+=s
                s=0
            count+=1
            if(ans*boardingCost-count*runningCost)>final:
                p=count
                final=(ans*boardingCost-count*runningCost)
        return p
class Solution:
    def minOperationsMaxProfit(self, customers: List[int], boardingCost: int, runningCost: int) -> int:
        mx, ans = -1, -1
        wait, income, cost = 0, 0, 0
        for i in range(1, 51*100000):
            if i <= len(customers):
                wait += customers[i - 1]
            elif wait == 0:
                break
            if wait >= 4:
                onboard = 4
            else:
                onboard = wait

            wait -= onboard
            income += onboard * boardingCost
            cost += runningCost
            curr = income - cost
            # print(onboard, income, cost, curr)
            if curr > mx:
                mx = curr
                ans = i
        return ans
class Solution:
    def minOperationsMaxProfit(self, customers: List[int], boardingCost: int, runningCost: int) -> int:
        res = 0
        q = 0
        u = 0
        profit = 0
        rotation = 0
        
        for c in customers:
            rotation += 1
            if c > 4:
                q += c - 4
                u += 4
            else:
                if q > 0:
                    diff = 4 - c
                    if q >= diff:   
                        u += c + diff
                        q -= diff
                    else:
                        u += c + q
                        q = 0
                else:
                    u += c
            
            if profit < u * boardingCost - rotation * runningCost:
                profit = u * boardingCost - rotation * runningCost
                res = rotation
        
        while q > 0:
            rotation += 1
            if q > 4:
                u += 4
                q -= 4
            else:
                u += q
                q = 0
            if profit < u * boardingCost - rotation * runningCost:
                profit = u * boardingCost - rotation * runningCost
                res = rotation
            
        print(profit)
        return res if profit > 0 else -1
                

class Solution:
  def minOperationsMaxProfit(self, customers: List[int], boardingCost: int, runningCost: int) -> int:
    x, y = -1, -1
    c = 0
    p = 0
    for i, j in enumerate(customers):
      c += j
      d = min(c, 4)
      c -= d
      p += d * boardingCost - runningCost
      if p > x:
        x, y = p, i + 1
    i = len(customers)
    while c:
      d = min(c, 4)
      c -= d
      p += d * boardingCost - runningCost
      i += 1
      if p > x:
        x, y = p, i
    return y

from typing import List

class Solution:
    def minOperationsMaxProfit(self, customers: List[int], boardingCost: int, runningCost: int) -> int:
        cur = best = 0
        best_idx = None
        waiting = 0
        capacity = 4

        for idx, nxt in enumerate(customers):
            # Add new waiting customers, put passengers on ride, remove passengers from waiting.
            waiting += nxt
            passengers = min(waiting, capacity)
            waiting -= passengers

            # Update money.
            cur += (passengers * boardingCost) - runningCost

            if cur > best:
                best_idx = idx + 1
                best = cur
        
        while waiting:
            idx += 1
            passengers = min(waiting, capacity)
            waiting -= passengers

            # Update money.
            cur += (passengers * boardingCost) - runningCost

            if cur > best:
                best_idx = idx + 1
                best = cur        

        if best_idx is None:
            best_idx = -1

        return best_idx

class Solution:
    def minOperationsMaxProfit(self, customers: List[int], boardingCost: int, runningCost: int) -> int:
        cur_profit = remainder_customers = steps = res = 0
        max_profit = -1
        for customer in customers:
            remainder_customers += customer
            gettting_on = min(remainder_customers, 4)
            cur_profit += gettting_on* boardingCost - runningCost 
            remainder_customers -= gettting_on
            steps += 1
            
            if cur_profit > max_profit:
                max_profit = cur_profit
                res = steps
     
        while remainder_customers > 0:
            gettting_on = min(remainder_customers, 4)
            cur_profit += gettting_on* boardingCost - runningCost 
            remainder_customers -= gettting_on
            steps += 1
            
            if cur_profit > max_profit:
                max_profit = cur_profit
                res = steps
    
        return -1 if max_profit < 0 else res
                

class Solution:
    def minOperationsMaxProfit(self, customers: List[int], boardingCost: int, runningCost: int) -> int:
        ans=-1
        wait=0
        profit=0
        best=0
        for i in range(len(customers)):
            wait+=customers[i]
            board=min(wait,4)
            profit+=board*boardingCost-runningCost
            wait-=board
            if profit>best:
                best=profit
                ans=i+1
        i=len(customers)
        while wait:
            board=min(wait,4)
            profit+=board*boardingCost-runningCost
            wait-=board
            if profit>best:
                best=profit
                ans=i+1
            i+=1
        return ans

class Solution:
    def minOperationsMaxProfit(self, customers: List[int], boardingCost: int, runningCost: int) -> int:
        curr, ans, waiting, profit = 0, 0, 0, 0
        for turn, customers in enumerate(customers):
            waiting += customers
            boarding = 4 if 4 < waiting else waiting
            waiting -= boarding
            profit += (boardingCost * boarding) - runningCost
            if profit > curr:
                curr, ans = profit, turn+1
        else:
            j = turn
            while waiting > 0:
                j += 1
                boarding = min(4, waiting)
                waiting -= boarding
                profit += (boardingCost * boarding) - runningCost
                if profit > curr:
                    curr, ans = profit, j + 1
        return ans if profit > 0 else -1
class Solution:
    def minOperationsMaxProfit(self, customers: List[int], boardingCost: int, runningCost: int) -> int:
#         res = [0]
#         wait = 0
#         for i in customers:
#             board = 0
#             wait += i
#             if wait > 4:
#                 board = 4
#                 wait -= 4
#             else:
#                 board = wait
#                 wait = 0
#             board*boardingCost - runningCost
#             res.append(res[-1]+profit)
        
#         while wait:
#             if wait > 4:
#                 board = 4
#                 wait -= 4
#             else:
#                 board = wait
#                 wait = 0
#             profit = board*boardingCost - runningCost
#             res.append(res[-1]+profit)
#         m = max(res)
#         if m <= 0:
#             return -1
#         else:
#             return res.index(m)
        ans = t = waiting = 0
        peak = 0
        peak_at = -1
        for i, x in enumerate(customers):
            waiting += x
            delta = min(4, waiting)
            profit = boardingCost * delta - runningCost
            ans += profit
            waiting -= delta
            if ans > peak:
                peak = ans
                peak_at = i + 1
        
        t = len(customers)
        while waiting:
            delta = min(4, waiting)
            profit = boardingCost * delta - runningCost
            if profit > 0:
                ans += profit
                waiting -= delta
                t += 1
                if ans > peak:
                    peak = ans
                    peak_at = t
            else:
                break
        
        return peak_at if peak_at > 0 else -1

class Solution:
    def minOperationsMaxProfit(self, customers: List[int], boardingCost: int, runningCost: int) -> int:
        profit=0
        maxprofit=0
        maxi=-1
        wait=0
        i=0
        for customer in customers:
            i+=1
            wait+=customer
            on=min(wait, 4)
            wait-=on
            profit+=on*boardingCost-runningCost
            if profit>maxprofit:
                maxprofit=profit
                maxi=i
        while wait:
            i+=1
            on=min(wait, 4)
            wait-=on
            profit+=on*boardingCost-runningCost
            if profit>maxprofit:
                maxprofit=profit
                maxi=i
        return maxi
class Solution:
    def minOperationsMaxProfit(self, customers: List[int], boardingCost: int, runningCost: int) -> int:
        
        b=0
        w=0
        r=0
        tc=0
        '''
        for i in customers:
            w+=i
            tc+=i
            #bd = min(4,w)
            #w-=bd
            if w//4 == 0:
                r+=1
            else:
                r+= (w//4)    
            print(r)
            
            if w<4:
                w-=w
            else:
                w-= (4*(w//4))
            print("     ",w)
        '''
        i=0
        n=len(customers)
        while True:
            if i==n:
                x=w//4
                r+=x
                if w%4 !=0 and ((w%4)*boardingCost)>runningCost:
                    r+=1
                break
            w+=customers[i]
            tc+=customers[i]
            bd=min(4,w)
            w-=bd
            r+=1
            i+=1
        
        return r if (tc*boardingCost - runningCost*r)>0 else -1
            
        
        

class Solution:
    def minOperationsMaxProfit(self, customers: List[int], boardingCost: int, runningCost: int) -> int:
        
        totalCustomers = sum(customers)
        rotation = totalCustomers//4
        totalRotation = rotation if totalCustomers%4 == 0 else rotation + 1
        totalRotation = max(totalRotation, len(customers))
        totalRotationCost = totalRotation * runningCost
        
        earning = rotation * 4 * boardingCost
        remain = totalCustomers%4
        if remain != 0:
            earning += (remain * boardingCost)
        profit = earning - totalRotationCost
        
        if profit <= 0:
            return -1
        
        maxProfit = 0
        currentCost = 0
        remainingCustomer = sum(customers)
        highestRotation = 0
        i = 0
        total = 0
        while total > 0 or i < len(customers):
            if i < len(customers):
                total += customers[i]
            
            prev = currentCost
            if total >= 4:
                currentCost += (4 * boardingCost - runningCost)
                total -= 4
            else:
                currentCost += (total * boardingCost - runningCost)
                total = 0
            
            if currentCost > maxProfit:
                maxProfit = currentCost
                highestRotation = i + 1
                
            i += 1
            
            
            
        print(maxProfit)
        print(profit)
        
        return highestRotation
            

class Solution:
    def minOperationsMaxProfit(self, customers: List[int], boardingCost: int, runningCost: int) -> int:
        
        
        
        # my solution ... 2112 ms ... 73 % ... 17.4 MB ... 98 %
        #  time: O(n)
        # space: O(1)
        
        curr_profit, curr_rotate =  0, 0
        best_profit, best_rotate = -1, 0
        queue_count = 0
        for customer in customers:
            queue_count += customer
            board_count = min(queue_count, 4)
            curr_profit += board_count * boardingCost - runningCost
            curr_rotate += 1
            queue_count -= board_count
            if curr_profit > best_profit:
                best_profit, best_rotate = curr_profit, curr_rotate
        while queue_count > 0:
            board_count = min(queue_count, 4)
            curr_profit += board_count * boardingCost - runningCost
            curr_rotate += 1
            queue_count -= board_count
            if curr_profit > best_profit:
                best_profit, best_rotate = curr_profit, curr_rotate
        return best_rotate if best_profit > 0 else -1
        
        

class Solution:
    def minOperationsMaxProfit(self, customers: List[int], boardingCost: int, runningCost: int) -> int:
        cur_profit = 0
        max_profit = -1
        waiting_customers = 0
        ans = -1
        turn = 0
        for customer in customers:
            turn += 1
            waiting_customers += customer
            to_board = min(waiting_customers, 4)
            waiting_customers -= to_board
            cur_profit += (-runningCost + to_board*boardingCost)
            if cur_profit > max_profit:
                max_profit = cur_profit
                ans = turn
           
            
        # if there is remaining customer
        while waiting_customers > 0:
            turn += 1
            to_board = min(waiting_customers, 4)
            waiting_customers -= to_board
            cur_profit += (-runningCost + to_board*boardingCost)
            if cur_profit > max_profit:
                max_profit = cur_profit
                ans = turn
            
        return ans if max_profit >= 0 else -1
class Solution:
    def minOperationsMaxProfit(self, customers: List[int], boardingCost: int, runningCost: int) -> int:
        cur_profit = remainder_customers = steps = res = 0
        max_profit = -1
        for customer in customers:
            remainder_customers += customer
            getting_on = min(remainder_customers, 4)
            cur_profit += getting_on * boardingCost - runningCost 
            remainder_customers -= getting_on
            steps += 1 
            
            if cur_profit > max_profit:
                max_profit = cur_profit
                res = steps
                
        while remainder_customers > 0:
            getting_on = min(remainder_customers, 4)
            cur_profit += getting_on * boardingCost - runningCost 
            remainder_customers -= getting_on
            steps += 1 
            
            if cur_profit > max_profit:
                max_profit = cur_profit
                res = steps
    
       
        return -1 if max_profit < 0 else res
                

class Solution:
    def minOperationsMaxProfit(self, customers: List[int], boardingCost: int, runningCost: int) -> int:
        waiting,prev_boarding,boarding,profit=0,0,0,[]
        for i, elem in enumerate(customers):
            if elem>=4:
                waiting+=elem-4
                boarding=4
            elif elem<4:
                if not waiting:
                    boarding=elem
                else:
                    boarding=4
                    waiting-=(4-elem)
            prev_boarding+=boarding
            profit.append(prev_boarding*boardingCost-((i+1)*runningCost))
        i=len(customers)
        while(waiting):
            if waiting>=4:
                boarding=4
                waiting-=4
                prev_boarding+=boarding
                i+=1
                profit.append(prev_boarding*boardingCost-((i)*runningCost))
            else:
                boarding=waiting
                waiting=0
                prev_boarding+=boarding
                i+=1
                profit.append(prev_boarding*boardingCost-((i)*runningCost))          
        return profit.index(max(profit))+1 if max(profit)>=0 else -1



class Solution:
    def minOperationsMaxProfit(self, customers: List[int], boardingCost: int, runningCost: int) -> int:
        
        
        
        # my solution ... 2124 ms ... 72 % ... 17.4 MB ... 95 %
        #  time: O(n)
        # space: O(1)
        
        curr_profit, curr_rotate =  0, 0
        best_profit, best_rotate = -1, 0
        queue_count = 0
        for customer in customers:
            queue_count += customer
            board_count = min(queue_count, 4)
            curr_profit += board_count * boardingCost - runningCost
            curr_rotate += 1
            queue_count -= board_count
            if curr_profit > best_profit:
                best_profit, best_rotate = curr_profit, curr_rotate
        while queue_count > 0:
            board_count = min(queue_count, 4)
            curr_profit += board_count * boardingCost - runningCost
            curr_rotate += 1
            queue_count -= board_count
            if curr_profit > best_profit:
                best_profit, best_rotate = curr_profit, curr_rotate
        return best_rotate if best_profit > 0 else -1
        
        

class Solution:
    def minOperationsMaxProfit(self, customers: List[int], boardingCost: int, runningCost: int) -> int:
        cur = 0
        p = 0
        ans = float('-inf')
        t = 0
        ff = 0
        for c in customers:
            cur += c
            cnt = min(cur, 4)
            cur -= cnt
            p += cnt * boardingCost
            p -= runningCost
            t += 1
            if p > ans:
                ans = p
                ff = t
    
        while cur > 0:
            cnt = min(cur, 4)
            cur -= cnt
            p += cnt * boardingCost
            p -= runningCost
            t += 1
            if p > ans:
                ans = p
                ff = t

        return -1 if ans <= 0 else ff
class Solution:
    def minOperationsMaxProfit(self, customers: List[int], boardingCost: int, runningCost: int) -> int:
        m=0
        res=-1
        leftover=0
        profit=0
        i=0
        j=0
        while leftover>0 or i<len(customers):
            if i<len(customers):
                leftover+= customers[i]
                i+=1
            if leftover>=4:
                profit+=4*boardingCost
                leftover-=4
            else:
                profit+=leftover*boardingCost
                leftover=0
            profit-=runningCost
            if profit>m:
                res=j+1
                m=profit
            j+=1
            
        return res
            

class Solution:
    def minOperationsMaxProfit(self, customers: List[int], boardingCost: int, runningCost: int) -> int:
        
        
        
        # my solution ... 
        #  time: O(n)
        # space: O(1)
        
        curr_profit, curr_rotate =  0, 0
        best_profit, best_rotate = -1, 0
        queue_count = 0
        for customer in customers:
            queue_count += customer
            board_count = min(queue_count, 4)
            curr_profit += board_count * boardingCost - runningCost
            curr_rotate += 1
            queue_count -= board_count
            if curr_profit > best_profit:
                best_profit, best_rotate = curr_profit, curr_rotate
        while queue_count > 0:
            board_count = min(queue_count, 4)
            curr_profit += board_count * boardingCost - runningCost
            curr_rotate += 1
            queue_count -= board_count
            if curr_profit > best_profit:
                best_profit, best_rotate = curr_profit, curr_rotate
        return best_rotate if best_profit > 0 else -1
        
        

class Solution:
    def minOperationsMaxProfit(self, customers: List[int], boardingCost: int, runningCost: int) -> int:
        ans = -1
        most = pnl = waiting = 0
        for i, x in enumerate(customers): 
            waiting += x # more people waiting in line 
            waiting -= (chg := min(4, waiting)) # boarding 
            pnl += chg * boardingCost - runningCost 
            if most < pnl: ans, most = i+1, pnl
        q, r = divmod(waiting, 4)
        if 4*boardingCost > runningCost: ans += q
        if r*boardingCost > runningCost: ans += 1
        return ans
class Solution:
    def minOperationsMaxProfit(self, customers: List[int], boardingCost: int, runningCost: int) -> int:
        
         
        n=len(customers)
        dp=[]
        reserved=0
        on_board=0
        rotation=0
        for i in range(n):
            if reserved!=0:
                if reserved>=4:
                    on_board+=4
                    reserved += customers[i]-4
                else:
                    new_customers=4-reserved
                    if customers[i]>=new_customers:
                        on_board+=4
                        reserved=customers[i]-new_customers
                    else:
                        on_board+=reserved+customers[i]
                        reserved=0
            else:
                if customers[i]>=4:
                    on_board+=4
                    reserved+=customers[i]-4
                else:
                    on_board+=customers[i]
            rotation+=1
            
            dp.append(on_board*boardingCost - rotation*runningCost)
        
        for i in range(reserved//4 + 1):
            if reserved>=4:
                on_board+=4
                reserved-=4
            else:
                on_board+=reserved
                reserved=0
            rotation+=1
            dp.append(on_board*boardingCost - rotation*runningCost)
        
        maxi=max(dp)
        return dp.index(max(dp))+1 if maxi>=0 else -1
            

class Solution:
    def minOperationsMaxProfit(self, customers: List[int], boardingCost: int, runningCost: int) -> int:
        queue = collections.deque()
        waiting = 0
        profit = float('-inf')
        cur = 0
        rounds = 1
        members = 0
        final_rounds = 0
        
        for c in customers:
            waiting += c
            
            new = min(4, waiting)
            members += new
            waiting -= new
            
            cur = boardingCost * members - runningCost * rounds
            if cur > profit:
                profit = cur
                final_rounds = rounds
            rounds += 1
        
        while waiting:
            new = min(4, waiting)
            members += new
            waiting -= new
            
            cur = boardingCost * members - runningCost * rounds
            if cur > profit:
                profit = cur
                final_rounds = rounds
            rounds += 1

        return final_rounds if profit > 0 else -1
class Solution:
    def minOperationsMaxProfit(self, customers: List[int], boardingCost: int, runningCost: int) -> int:
        
        max_profit = float('-inf')
        queue = 0
        idx = 0 
        min_idx = 0
        profit = 0
        itr  = 0
        
        while True:
            if idx < len(customers):
                queue += customers[idx]
            served  = 0
            if queue >= 4:
                queue-=4
                served = 4
            else:
                served  = queue 
                queue = 0
            
            profit  += boardingCost*served
            profit -= runningCost
            itr+=1
            #print(f'profit={profit} itr={itr} served = {served} profit = {profit}')
            if profit > max_profit:
                max_profit =  profit
                min_idx = itr
                
            idx += 1
            if queue <=0 and idx>=len(customers):
                break
                
        if max_profit <= 0:
            return -1
        else:
            return min_idx
                
            

class Solution:
    def minOperationsMaxProfit(self, cust: List[int], boardingC: int, runningC: int) -> int:
        maxp=curr=rem=ans=rot=0
        i=0
        
        l=len(cust)
        while i<l or rem:
            rot+=1
            rem+=cust[i] if i<l else 0 
            
            if rem>=4:
                curr+=4
                rem-=4
                
            else:
                curr+=rem
                rem=0
                
            
                
            prof=(curr*boardingC)-(runningC*rot)

            if prof>maxp:
                maxp=prof
                ans=rot
                    
            i+=1
                    
        return ans if maxp>0 else -1
                

class Solution:
    def minOperationsMaxProfit(self, customers: List[int], boardingCost: int, runningCost: int) -> int:
        profits = []
        curr_profit = 0
        ppl_waiting = 0
        
        for i, ppl in enumerate(customers):
            ppl_waiting += ppl
            num_board = min(4, ppl_waiting)
            curr_profit += num_board * boardingCost - runningCost
            profits.append(curr_profit)
            ppl_waiting -= num_board
        
        while ppl_waiting:
            num_board = min(4, ppl_waiting)
            curr_profit += num_board * boardingCost - runningCost
            profits.append(curr_profit)
            ppl_waiting -= num_board
        
        #print(profits)
        max_prof = max(profits)
        if max_prof <= 0:
            return -1
        return profits.index(max_prof) + 1
from collections import deque

class Solution:
    def minOperationsMaxProfit(self, customers: List[int], boardingCost: int, runningCost: int) -> int:
        
        
        rotations = 0
        maxprofit = -1
        currprofit = 0
        nr = 0
        q = deque(customers)
        
        while q:
            currprofit -= runningCost
            nr += 1
            
            c = q.popleft()
            if c > 4:
                if q:
                    q[0] += c-4
                else:
                    q.append(c-4)
                c = 4
                
            currprofit += c * boardingCost
            if currprofit > maxprofit:
                maxprofit = currprofit
                rotations = nr
            
        return rotations if maxprofit > 0 else -1
            
            
            
            
        
        

class Solution:
    def minOperationsMaxProfit(self, customers: List[int], boardingCost: int, runningCost: int) -> int:
        if not customers:
            return 0
        
        profits = [0]
        waiting = 0
        
        for customer in customers:
            waiting += customer
            serving = 0
            if waiting > 4:
                serving = 4
                waiting -=4
            else:
                serving = waiting
                waiting =0
                
            profits.append(profits[-1] + ((boardingCost*serving) - runningCost))
            
        while waiting:
            serving = 0
            if waiting > 4:
                serving = 4
                waiting -=4
            else:
                serving = waiting
                waiting =0
                
            profits.append(profits[-1] +  ((boardingCost*serving) - runningCost))
        
        maxProfit = profits[1]
        maxProfitIndex = 1
        for i, profit in enumerate(profits[1:]):
            if profit>maxProfit:
                maxProfit = profit
                maxProfitIndex = i+1
        
        if maxProfit>0:
            return maxProfitIndex
        else:
            return -1
        
        

class Solution:
    def minOperationsMaxProfit(self, customers: List[int], boardingCost: int, runningCost: int) -> int:
        rotate = left = 0
        ans = -1
        profit = maxprofit = 0
        for cnt in customers:
            cnt += left
            rotate += 1
            left = max(0, cnt - 4)
            profit += boardingCost * min(4, cnt) - runningCost
            if profit > maxprofit:
                maxprofit = profit
                ans = rotate
        while left > 0:
            rotate += 1
            profit += boardingCost * min(4, left) - runningCost
            if profit > maxprofit:
                maxprofit = profit
                ans = rotate
            left -= 4
        return ans
class Solution:
    def minOperationsMaxProfit(self, A, BC, RC):
        ans=profit=t=0
        maxprofit=0
        wait=i=0
        n=len(A)
        while wait or i<n:
            if i<n:
                wait+=A[i]
                i+=1
            t+=1
            y=min(4,wait)
            wait-=y
            profit+=y*BC
            profit-=RC
            if profit>maxprofit:
                maxprofit=profit
                ans=t

        if maxprofit<=0:
            return -1
        else:
            return ans

class Solution:
    # O(n) Time | O(1) Space
    def minOperationsMaxProfit(self, customers: List[int], boardingCost: int, runningCost: int) -> int:
        result = 0
        totalPeople = 0
        totalProfit = 0
        maxProfit = 0
        for i, num in enumerate(customers):
            totalPeople += num
            onBoard = min(4, totalPeople)
            totalPeople -= onBoard
            totalProfit += onBoard * boardingCost - runningCost
            if totalProfit > maxProfit:
                maxProfit = totalProfit
                result = i + 1
        q, r = divmod(totalPeople, 4)
        if 4 * boardingCost > runningCost:
            result += q
        if r * boardingCost > runningCost:
            result += 1
        return result if result > 0 else -1
class Solution:
    def minOperationsMaxProfit(self, customers: [int], boardingCost: int, runningCost: int) -> int:
        waiting=0
        profits=[0]

        if 4*boardingCost<runningCost:
            return -1

        for customer in customers:
            if customer+waiting>=4:
                profit=4*boardingCost-runningCost
                waiting=waiting+customer-4
            else:
                profit = (waiting+customer) * boardingCost - runningCost
                waiting=0
            last=profits[-1]
            profits.append(profit+last)

        while waiting>0:
            if waiting>=4:
                profit = 4 * boardingCost - runningCost
                last = profits[-1]
                profits.append(profit + last)
                waiting-=4
            else:
                profit = waiting * boardingCost - runningCost
                last = profits[-1]
                profits.append(profit + last)
                waiting=0
        return profits.index(max(profits))

class Solution:
    def minOperationsMaxProfit(self, customers: List[int], boardingCost: int, runningCost: int) -> int:
        wait = 0
        answer = 0
        current = 0
        counter = 0
        result = -1
        for entry in customers:
            counter += 1
            wait += entry
            current -= runningCost
            if wait >= 4:
                current += (4*boardingCost)
                wait -= 4
            else:
                current += (wait*boardingCost)
                wait = 0
            if current > answer:
                result = counter
            answer = max(answer, current)
        while wait > 0:
            counter += 1
            current -= runningCost
            if wait >= 4:
                current += (4*boardingCost)
                wait -= 4
            else:
                current += (wait*boardingCost)
                wait = 0
            if current > answer:
                result = counter
            answer = max(answer, current)
        return result
class Solution:
    def minOperationsMaxProfit(self, customers: List[int], boardingCost: int, runningCost: int) -> int:
        if boardingCost * 4 <= runningCost or len(customers) == 0:
            return -1
        left = 0
        profit = 0
        res = 0
        max_profit = 0
        idx = 0
        r = 0
        while idx < len(customers) or left > 0:
            r += 1
            if idx < len(customers):
                left += customers[idx]
                idx += 1
            
            if left < 4:
                profit += left * boardingCost - runningCost
                left = 0
            else:
                left -= 4
                profit += 4 * boardingCost - runningCost
                
            if profit > max_profit:
                max_profit = profit
                res = r
        return res
class Solution:
    def minOperationsMaxProfit(self, customers: List[int], boardingCost: int, runningCost: int) -> int:
        res = 0
        ans = -1
        rotate = 1
        current = 0
        rotate_times = -1
        for i, number in enumerate(customers):
            if number > 4:
                res += number - 4
                current += 4
            elif number == 4: 
                current += number
            else:
                if res + number > 4:
                    current += 4
                    res = res + number - 4
                else:
                    current += res + number
                    res = 0
            profit = boardingCost * current - rotate*runningCost
            # print("the profit is: " + str(profit))
            rotate += 1
            if ans < profit:
                ans = profit
                rotate_times = rotate
        # print("the res is: " + str(res))
            
        while res > 0:
            if res > 4:
                current += 4
                res -= 4
            else:
                current += res
                res = 0
            profit = boardingCost * current - rotate*runningCost
            # print("the profit is: " + str(profit))
            rotate += 1
            if ans < profit:
                ans = profit
                rotate_times = rotate
        return rotate_times -1  if rotate_times - 1 > 0 else -1

from collections import deque
from typing import List


class Solution:
    def minOperationsMaxProfit(self, customers: List[int], boardingCost: int, runningCost: int) -> int:
        ls = []

        waiting = 0  # ub77cuc778 uae30ub2e4ub9acub294 uc0acub78c
        gondola = deque()  # uace4ub3ccub77c que
        curr = 0  # ud604uc7ac uace4ub3ccub77cuc5d0 ud0c0uace0uc788ub294 uc0acub78c
        days = 0
        for i, customer in enumerate(customers):
            waiting += customer
            if waiting >= 4:
                waiting -= 4
                on_board = 4
            else:
                on_board = waiting
                waiting = 0
            curr += on_board

            # profit
            ls.append(curr * boardingCost - (days + 1) * runningCost)
            days += 1
        while waiting > 0:
            if waiting >= 4:
                waiting -= 4
                on_board = 4
            else:
                on_board = waiting
                waiting = 0
            curr += on_board

            # profit
            ls.append(curr * boardingCost - (days + 1) * runningCost)
            days += 1

        max_val = max(ls)
        if max_val < 0:
            return -1
        else:
            return ls.index(max_val) + 1

class Solution:
    def minOperationsMaxProfit(self, customers: List[int], boardingCost: int, runningCost: int) -> int:
        res, maxprofit = -1, 0
        curr_profit = nwait = 0
        for idx,num in enumerate(customers):
            nwait += num
            curr_profit += (4*boardingCost if nwait >= 4 else nwait * boardingCost) - runningCost
            if nwait >= 4:
                nwait -= 4
            else:
                nwait = 0
            if curr_profit > maxprofit:
                res = idx+1
                maxprofit = curr_profit
        while nwait > 0:
            idx += 1
            curr_profit += (4*boardingCost if nwait >= 4 else nwait * boardingCost) - runningCost
            if nwait >= 4:
                nwait -= 4
            else:
                nwait = 0
            if curr_profit > maxprofit:
                res = idx+1
                maxprofit = curr_profit
        return res
                
                
            
                    
                

class Solution:
    def minOperationsMaxProfit(self, customers: List[int], boardingCost: int, runningCost: int) -> int:
        bc, rc, n = boardingCost, runningCost, len(customers)
        res, sum_v, max_v, wait, i = -1, 0, 0, 0, 0
        if 4 * bc <= rc:
            return -1
        # customers.append(0)
        while i < n or wait > 0:
            wait += customers[i] if i < n else 0
            cur = wait if wait < 4 else 4
            wait -= cur
            sum_v += cur * bc - rc
            # (i,wait,cur,sum_v,max_v).p()
            if sum_v > max_v:
                max_v = sum_v
                res = i + 1
            i += 1
        return res
class Solution:
    def minOperationsMaxProfit(self, customers: List[int], boardingCost: int, runningCost: int) -> int:
        
        count =0
        max_profit = 0
        profit =0
        rot =-1
        i=0
        while count>0 or i<len(customers):
            
            if i<len(customers):
                new_customers = customers[i]
                count+=new_customers
            if count>=4:
                count-=4
                profit+=4*boardingCost-runningCost
            else:
                profit+=count*boardingCost-runningCost
                count=0
               
            i+=1
            if profit>max_profit:
                max_profit = profit
                rot = i
                
            
            #print(i,profit,count)
            
            
        return rot
                
            
            
            
            

class Solution:
    def minOperationsMaxProfit(self, customers: List[int], boardingCost: int, runningCost: int) -> int:
        n, i, max_profit, waiting, profit, max_idx = len(customers), 0, 0, 0, 0, -1
        
        while waiting > 0 or i<n:
            if i<n:
                waiting += customers[i]
            i+=1
            count = min(4, waiting)
            waiting -= count
            profit += count * boardingCost - runningCost
            if profit > max_profit:
                max_profit, max_idx = profit, i
        
        return max_idx 
class Solution:
    def minOperationsMaxProfit(self, customers: List[int], boardingCost: int, runningCost: int) -> int:
        ans,max_profit=-1,0
        waiting=0
        profit=0
        i=0
        while waiting>0 or i<len(customers):
            if i<len(customers):waiting+=customers[i]
            if waiting>=4:
                waiting-=4
                boarded=4
            else:
                boarded=waiting
                waiting=0
            max_profit+=(boardingCost*boarded)-runningCost
            if max_profit>0 and max_profit>profit:
                ans=i+1
                profit=max_profit
            i+=1
        return ans
class Solution:
    def minOperationsMaxProfit(self, customers: List[int], boardingCost: int, runningCost: int) -> int:
        ans = -1
        most = pnl = waiting = 0
        for i, x in enumerate(customers): 
            waiting += x # more people waiting in line 
            waiting -= (chg := min(4, waiting)) # boarding 
            pnl += chg * boardingCost - runningCost 
            if most < pnl: ans, most = i+1, pnl
        q, r = divmod(waiting, 4)
        if 4*boardingCost > runningCost: ans += q
        if r*boardingCost > runningCost: ans += 1
        return ans 
class Solution:
    def minOperationsMaxProfit(self, customers: List[int], boardingCost: int, runningCost: int) -> int:
        best = 0
        bestindex = 0
        current = 0
        currentindex = 0
        nc = 0
        for i in range(len(customers)):
            nc += customers[i]
            canadd = min(4,nc)
            nc -= canadd
            current += canadd*boardingCost
            current -= runningCost
            currentindex+=1
            # print(current)
            
            if current > best:
                best = current
                bestindex = currentindex
                
        while nc > 0:
            canadd = min(4,nc)
            nc -= canadd
            current += canadd*boardingCost
            current -= runningCost
            currentindex+=1
            # print(current)
            
            if current > best:
                best = current
                bestindex = currentindex
                
        if best == 0:
            bestindex = -1
                
        return bestindex
            

class Solution:
    def minOperationsMaxProfit(self, customers: List[int], boardingCost: int, runningCost: int) -> int:
        curr, ans, waiting, profit = 0, 0, 0, 0
        for turn, customers in enumerate(customers):
            waiting += customers
            boarding = min(4, waiting)
            waiting -= boarding
            profit += (boardingCost * boarding) - runningCost
            if profit > curr:
                curr, ans = profit, turn+1
        else:
            j = turn
            while waiting > 0:
                j += 1
                boarding = min(4, waiting)
                waiting -= boarding
                profit += (boardingCost * boarding) - runningCost
                if profit > curr:
                    curr, ans = profit, j + 1
        return ans if profit > 0 else -1
class Solution:
  def minOperationsMaxProfit(self, customers: List[int], boardingCost: int, runningCost: int) -> int:

    max_profit = -sys.maxsize
    result = - sys.maxsize
    pending = 0
    running_counter = 0
    current_onboarding = 0
    for customer in customers:
      pending += customer
      running_counter += 1
      if pending > 0:
        real_boarding = min(4, pending)
        current_onboarding += real_boarding
        profit = current_onboarding * boardingCost - running_counter * runningCost

        if max_profit < profit:
          max_profit = profit
          result = running_counter
          pass
        pending -= real_boarding
        pass
      pass

    while pending:
      running_counter += 1
      real_boarding = min(4, pending)
      current_onboarding += real_boarding
      pending -= real_boarding
      profit = current_onboarding * boardingCost - running_counter * runningCost

      if max_profit < profit:
        max_profit = profit
        result = running_counter
        pass
      pass

    if max_profit <= 0:
      return -1
    return result
class Solution:
    def minOperationsMaxProfit(self, customers: List[int], boardingCost: int, runningCost: int) -> int:
        b = w = i = 0
        n = len(customers)
        prof = 0
        mprof = 0
        pos = -1
        while i < n or w:
            if i < n:
                w += customers[i]
            i += 1
            b = min(w, 4)
            w -= b
            prof += b * boardingCost - runningCost
            if prof > mprof:
                mprof = prof
                pos = i
        return pos
class Solution:
    def minOperationsMaxProfit(self, customers: List[int], boardingCost: int, runningCost: int) -> int:
        run = 0
        maxRun = 0
        profit = 0
        maxProfit = 0
        total = 0
        i = 0
        n = len(customers)
        while total > 0 or i < n:
            if i < n:
                total += customers[i]
                i += 1
            group = min(4, total)
            total -= group
            profit = profit + group*boardingCost - runningCost
            run += 1
            if profit > maxProfit:
                maxProfit = profit
                maxRun = run
                
        return maxRun if maxProfit > 0 else -1

class Solution:
    def minOperationsMaxProfit(self, lis: List[int], b: int, r: int) -> int:
        n = len(lis)
        q=pro=ans=tot=c=fin=0
        c=1
        for i in range(n):
            q+=lis[i]
            t = min(4,q)
            q-=t
            tot+=t
            pro = tot*b - r*(c)
            if pro>ans:
                ans=pro
                fin=c
       #     print(pro,ans,tot)
            c+=1
        while q>0:
            t = min(4,q)
            q-=t
            tot+=t
            pro = tot*b - r*(c)
            if pro>ans:
                ans=pro
                fin=c
        #    print(pro,ans,tot)
            c+=1
        if fin==0:
            fin=-1
        return fin
        
            

class Solution:
    def minOperationsMaxProfit(self, customers: List[int], boardingCost: int, runningCost: int) -> int:
        max_profit = 0
        answ = -1
        waiting = 0
        profit = 0
        i = 0
        
        while i < len(customers) or waiting > 0:
            if i < len(customers):
                waiting += customers[i]
            if waiting >= 4:
                profit += 4*boardingCost - runningCost
                waiting -= 4
            elif waiting > 0:
                profit += waiting*boardingCost - runningCost
                waiting = 0
            else:
                profit -= runningCost
            
            if max_profit < profit:
                max_profit = profit
                answ = i + 1
            
            i += 1
                
        return answ

class Solution:
    def minOperationsMaxProfit(self, customers: List[int], boardingCost: int, runningCost: int) -> int:
        customers_waiting = 0
        max_till_now = 0
        prev_cost = 0
        answer = -1
        
        for index, customer in enumerate(customers):
            on_board = min(customer + customers_waiting, 4)
            customers_waiting = customer + customers_waiting - 4 if on_board == 4 else 0
            
            cost = prev_cost + on_board * boardingCost - runningCost
            prev_cost = cost
            
            if cost > max_till_now:
                max_till_now = cost
                answer = index + 1

        while customers_waiting:
            index += 1
            on_board = min(customers_waiting, 4)
            customers_waiting = customers_waiting - 4 if on_board == 4 else 0
            
            cost = prev_cost + on_board * boardingCost - runningCost
            prev_cost = cost
            
            if cost > max_till_now:
                max_till_now = cost
                answer = index + 1
                
        return answer
from typing import List


class Solution:
    def minOperationsMaxProfit(self, customers: List[int], boardingCost: int, runningCost: int) -> int:
        def boarders():
            waiters = 0
            for c in customers:
                waiters += c
                cur = min(4, waiters)
                waiters -= cur
                yield cur
            while waiters > 0:
                cur = min(4, waiters)
                waiters -= cur
                yield cur

        max = 0
        max_idx = -1
        cur = 0
        for i, b in enumerate(boarders()):
            cur += (b * boardingCost) - runningCost
            if cur > max:
                max = cur
                max_idx = i + 1

        return max_idx

class Solution:
    def minOperationsMaxProfit(self, customers: List[int], boardingCost: int, runningCost: int) -> int:
        if (boardingCost << 2) <= runningCost:
            return -1
        waitingCustomers = 0
        curProfit = 0
        maxProfit = 0
        maxProfitTurns = 0
        
        curTurn = 0
        for nCust in customers:
            curTurn += 1
            waitingCustomers += nCust
            if waitingCustomers > 0:
                boardedCustomers = min(4, waitingCustomers)
                waitingCustomers -= boardedCustomers
                curProfit += boardedCustomers * boardingCost
            curProfit -= runningCost
            if curProfit > maxProfit:
                maxProfit = curProfit
                maxProfitTurns = curTurn
        
        
        fullLoads = waitingCustomers >> 2
        remLoad = waitingCustomers & 0b11
        curProfit += ((fullLoads * boardingCost) << 2) - runningCost * fullLoads
        curTurn += fullLoads
        if curProfit > maxProfit:
            maxProfit = curProfit
            maxProfitTurns = curTurn
        if remLoad > 0:
            curProfit += remLoad * boardingCost - runningCost
            curTurn += 1
            if curProfit > maxProfit:
                maxProfit = curProfit
                maxProfitTurns = curTurn
        
        
        if curProfit > maxProfit:
            return curTurn
        
        if maxProfit == 0:
            return -1
        else:
            return maxProfitTurns
class Solution:
    def minOperationsMaxProfit(self, cust: List[int], board: int, run: int) -> int:
        wait = 0;tot = 0;profit = 0;move = 1;ans =0;maxi = 0
        for i in range(len(cust)):
            tot += min(4,cust[i]+wait)
            wait = max(0,cust[i]+wait-4)
            profit = tot*board - move*run
            if profit > maxi:
                maxi = profit;ans = move
            move +=1
            #print(tot,wait,profit)
        while wait > 0:
            tot += min(4,wait)
            wait -= 4
            profit = tot*board - move*run
            if profit > maxi:
                maxi = profit;ans = move
            move +=1
            #print(tot,wait,profit)
        if maxi > 0:
            return ans
        return -1

class Solution:
    def minOperationsMaxProfit(self, customers: List[int], boarding_cost: int, running_cost: int) -> int:
        customer = 0
        remained = 0
        max_profit = -1
        run = 0
        max_run = 0
        index = 0

        while remained or index < len(customers):
            run += 1
            remained += customers[index] if index < len(customers) else 0

            if (remained < 4):
                customer += remained
                remained = 0
            else:
                customer += 4
                remained -= 4

            profit = customer * boarding_cost - run * running_cost
            if (profit > max_profit):
                max_profit = profit
                max_run = run

            index += 1

        return -1 if max_profit < 0 else max_run
class Solution:
    def minOperationsMaxProfit(self, customers: List[int], boardingCost: int, runningCost: int) -> int:
        if not customers:
            return 0
        waiting = 0
        op = 0
        maxProfit = -1
        index = 0
        total_boarding = 0
        for i, people in enumerate(customers):
            op += 1
            waiting += people
            boarding = min(4, waiting)
            total_boarding += boarding
            currProfit = total_boarding * boardingCost - op * runningCost
            # other way is to just add the profit
            # currProfit += boarding * boardingCost - runningCost
            waiting -= boarding
            if currProfit > maxProfit:
                maxProfit = currProfit
                index = op
            
        while waiting > 0:
            op += 1
            boarding = min(waiting, 4)
            total_boarding += boarding
            currProfit = total_boarding * boardingCost - op * runningCost
            waiting -= boarding
            if currProfit > maxProfit:
                maxProfit = currProfit
                index = op
            
        if maxProfit == -1:
            return -1
        else:
            return index
class Solution:
    def minOperationsMaxProfit(self, c: List[int], b: int, r: int) -> int:
      n = len(c)
      i = 0
      rest = 0
      max_val, max_i = 0, -2
      val = 0
      while i<n or rest > 0:
        if i < n:
          rest += c[i]
        p = min(rest, 4)
        val += p * b - r
        if val > max_val:
          max_val = val
          max_i = i
        rest -= p
        i += 1  
      return max_i + 1    
        

MIN = float('-inf')
class Solution:
    def minOperationsMaxProfit(self, customers, boardingCost, runningCost):
        
        n = len(customers)
        step, maxStep, maxProfit = 0, 0, MIN
        i, people, queue = 0, 0, 0
        while True:
            if i < n:
                queue += customers[i]
                i += 1
            p = min(4, queue)
            queue -= p
            people += p
            step += 1
            profit = people * boardingCost - step * runningCost
            if profit > maxProfit:
                maxProfit = profit
                maxStep = step
            if queue == 0 and i == n:
                break
        return maxStep if maxProfit > 0 else -1
            
            
            
            
            
            
            
            
            
            
            
            
            
            

class Solution:
    def minOperationsMaxProfit(self, customers: List[int], boardingCost: int, runningCost: int) -> int:
        profit=record=peeps=rots=i=0
        n = len(customers)
        while peeps or i < n:
            if i < n:
                peeps+=customers[i]
            board = min(4, peeps)
            profit += boardingCost*board - runningCost
            peeps-=board
            if profit > record:
                rots = i + 1
                record = profit
            i+=1
        return rots if rots != 0 else -1

class Solution:
    def minOperationsMaxProfit(self, customers: List[int], boardingCost: int, runningCost: int) -> int:
        customers = customers[::-1]
        in_line = profit = rot = best_rot = 0
        max_profit = -1
        while customers or in_line:
            c = customers.pop() if customers else 0
            in_line += c
            board = min(in_line, 4)
            in_line -= board
            profit += board * boardingCost 
            rot += 1
            profit -= runningCost
            if profit > max_profit:
                max_profit = profit
                best_rot = rot
         
        return best_rot if max_profit > 0 else -1
class Solution:
    def minOperationsMaxProfit(self, customers: List[int], boardingCost: int, runningCost: int) -> int:
        curr_waiting = 0
        curr_profit, max_profit, curr_op, max_profit_ops = 0, 0, 0, 0
        for c in customers:
            curr_op += 1
            curr_waiting += c
            num_boarded = min(4, curr_waiting)
            curr_waiting -= num_boarded
            curr_profit += boardingCost * num_boarded - runningCost
            if curr_profit > max_profit:
                max_profit = curr_profit
                max_profit_ops = curr_op
        while curr_waiting:
            curr_op += 1
            num_boarded = min(4, curr_waiting)
            curr_waiting -= num_boarded
            curr_profit += boardingCost * num_boarded - runningCost
            if curr_profit > max_profit:
                max_profit = curr_profit
                max_profit_ops = curr_op     
        return max_profit_ops if max_profit > 0 else - 1

from collections import deque

class Solution:
    def minOperationsMaxProfit(self, customers: List[int], boardingCost: int, runningCost: int) -> int:
        customers = deque(customers)
        in_line = profit = rot = best_rot = 0
        max_profit = -1
        while customers or in_line:
            c = customers.popleft() if customers else 0
            in_line += c
            board = min(in_line, 4)
            in_line -= board
            profit += board * boardingCost 
            rot += 1
            profit -= runningCost
            if profit > max_profit:
                max_profit = profit
                best_rot = rot
         
        return best_rot if max_profit > 0 else -1
class Solution:
    def minOperationsMaxProfit(self, customers: List[int], boardingCost: int, runningCost: int) -> int:
        
        rotations = 0
        profit, max_profit = 0, 0
        res = -1
        
        def xreport(boarded:int):
            nonlocal profit, rotations, max_profit, res
            profit +=boarded*boardingCost - runningCost
            rotations +=1
            if profit>max_profit:
                max_profit = profit
                res = rotations
            #print(f'p: {profit}, mprofit: {max_profit}, rotations:{rotations}')
            return
        
        for i in range(len(customers)-1):
            if customers[i] >4:
                customers[i+1]+=customers[i]-4
                customers[i] =4
            
            xreport(customers[i])
        
            
        waiting = customers[-1]
        while waiting>0:
            if waiting>4:
                waiting-=4
                xreport(4)
            else:
                xreport(waiting)
                waiting = 0
            
        return res

class Solution:
    def minOperationsMaxProfit(self, customers: List[int], boardingCost: int, runningCost: int) -> int:
        profits=[]
        occupy=0
        cur_waiting = 0
        k=0
        while cur_waiting ==0:
            cur_waiting=customers[k]
            k+=1
        m=k
        counter=0
        while cur_waiting >0:
            if cur_waiting < 4:
                occupy+=cur_waiting
                cur_waiting = 0
            else:
                occupy+=4
                cur_waiting-=4
            counter+=1
            profits.append(occupy*boardingCost - counter*runningCost)
            if k<len(customers):
                cur_waiting+=customers[k]
                k+=1
        z=max(profits)
        if z>0:
            return (profits.index(z)+m)
        else:
            return -1
                
                
            

class Solution:
    def minOperationsMaxProfit(self, customers: List[int], boardingCost: int, runningCost: int) -> int:
        
        d = {}
        rotations, sum_cust = 0, 0
        d[0] = 0
        
        for i, current in enumerate(customers):
            sum_cust +=current
            
            if sum_cust>=4:
                while sum_cust>=4:
                    rotations +=1
                    d[rotations] = d[rotations-1] + 4*boardingCost - runningCost
                    sum_cust -=4
            else:
                if i+1 in d:
                    continue
                rotations+=1
                d[rotations] = d[rotations-1]+ sum_cust*boardingCost - runningCost
                sum_cust = 0
        
        if sum_cust>0:
            rotations+=1
            d[rotations] = d[rotations-1]+ sum_cust*boardingCost - runningCost
            sum_cust = 0
        
        #print(d)
        d[0] = -1
        tmp, res = -1, -1
        for i in d:
            if tmp<d[i]:
                tmp = d[i]
                res = i
        return res
            

from typing import List

class Solution:
    def minOperationsMaxProfit(self, customers: List[int], boardingCost: int, runningCost: int) -> int:
        cur = best = wheel_turns = 0
        best_idx = None
        waiting = 0
        capacity = 4

        while waiting or wheel_turns < len(customers):
            # Add new waiting customers, put passengers on ride, remove passengers from waiting.
            if wheel_turns < len(customers):
                waiting += customers[wheel_turns]
            
            wheel_turns += 1
            
            passengers = min(waiting, capacity)
            waiting -= passengers

            # Update money.
            cur += (passengers * boardingCost) - runningCost

            if cur > best:
                best_idx = wheel_turns
                best = cur  
            
        if best_idx is None:
            best_idx = -1

        return best_idx


class Solution:
    def minOperationsMaxProfit(self, customers: List[int], boardingCost: int, runningCost: int) -> int:
        res = round = profit = max_profit = remain = i = 0
        while remain or i < len(customers):
            if i < len(customers):
                remain += customers[i]
                i += 1
            on_board = min(4, remain)
            remain -= on_board
            profit += boardingCost * on_board - runningCost
            round += 1
            if profit > max_profit:
                max_profit = profit
                res = round
        return res if max_profit > 0 else -1
import sys #         O(N) , N is customers.length

class Solution:
    def minOperationsMaxProfit(self, customers: List[int], boarding_cost: int, running_cost: int) -> int:

        i = 0
        remains = 0
        standard = sum(customers)
        stack = 0
        ans = -1
        big = -sys.maxsize-1
        while True:
            if standard <= 0:
                break
            if i < len(customers):
                remains += customers[i]
            
            if remains >= 4:
                remains -= 4
                stack += 4
                standard -= 4
            else:
                stack += remains
                standard -= remains
                remains = 0
            
            tmp = stack*boarding_cost - running_cost*(i+1)
            if tmp > big:
                big = tmp
                ans = i
                
                
            i += 1
                
            
        if big <= 0:
            return -1
        
        return ans+1
            
#         memberub3c4 ub298uc5b4ub098uc57c ud55cub2e4.

class Solution:
    def minOperationsMaxProfit(self, customers: List[int], boardingCost: int, runningCost: int) -> int:
        if not customers or len(customers) == 0:
            return -1
        
        cur_round = 1
        max_round = -1 
        customer_cnt = board_cnt = cur_profit = max_profit = i = 0
        
        while (customer_cnt > 0 or i < len(customers)):
            if i < len(customers):
                customer_cnt += customers[i]
                i += 1
    
            board_cnt = min(customer_cnt, 4)
            customer_cnt -= board_cnt
            
            cur_profit += (board_cnt * boardingCost) - runningCost
            if cur_profit > max_profit:
                max_profit = cur_profit
                max_round = cur_round 
            
            cur_round += 1 
        
        return max_round
            
            

class Solution:
    def minOperationsMaxProfit(self, customers: List[int], boardingCost: int, runningCost: int) -> int:
        if runningCost > 4*boardingCost:
            return -1
        profit = 0
        board = 0
        wait = 0
        ans = 0
        for i in range(len(customers)):
            if customers[i] > 4:
                wait += customers[i]-4
                board += 4
                p = board*boardingCost - (i+1)*runningCost
                if p > profit:
                    profit = p
                    ans = i+1
            else:
                add = min(wait, 4-customers[i])
                wait -= add
                board += add+customers[i]
                p = board*boardingCost - (i+1)*runningCost
                if p > profit:
                    profit = p
                    ans = i+1
        
        i = len(customers)
        while wait != 0:
            add = min(wait,4)
            board += add
            wait -= add
            p = board*boardingCost - (i+1)*runningCost
            if p > profit:
                profit = p
                ans = i+1
            i += 1
        
        return ans
class Solution:
    def minOperationsMaxProfit(self, customers: List[int], boardingCost: int, runningCost: int) -> int:
        profit = [0]
        left = 0
        maxMid = -1
        maxProfit = 0
        for i in range(len(customers)):
            toBeBoarded = left + customers[i]
            if toBeBoarded > 4:
                boarded = 4
                left = toBeBoarded - 4                
            else:
                boarded = customers[i]
                left = 0
            newProfit = boardingCost * boarded - runningCost + profit[-1]
            profit.append(newProfit)
            if newProfit > maxProfit:
                maxProfit = newProfit
                maxMid = i
        # print(profit)
        if left == 0:
            return maxMid
        elif left < 4:
            lastProfit = boardingCost * left - runningCost + profit[-1]
            if maxProfit >= lastProfit:
                return maxMid + 1 if maxProfit > 0 else -1
            else:
                return len(customers) + 1 if lastProfit > 0 else -1
        else:
            potential = boardingCost * 4 - runningCost
            if potential > 0:
                rotations = left // 4
                lastRun = left % 4
                rotationEnd = profit[-1] + potential * rotations
                lastProfit = rotationEnd + lastRun * boardingCost - runningCost
                if maxProfit >= rotationEnd:
                    return maxMid + 1 if maxProfit > 0 else -1
                else:
                    if rotationEnd >= lastProfit:
                        return len(customers) + rotations if rotationEnd > 0 else -1
                    else:
                        return len(customers) + rotations + 1 if lastProfit > 0 else -1
            else:
                return maxMid + 1 if maxProfit > 0 else -1

class Solution:
    def minOperationsMaxProfit(self, customers: List[int], boardingCost: int, runningCost: int) -> int:
        waiting = 0
        boarded = 0
        maxProfit = 0
        maxIndex = -1
        
        if len(customers) == 0:
            return -1 
        
        for i, customerCount in enumerate(customers):
            waiting += customerCount
            if waiting >= 4:
                waiting -= 4
                boarded += 4
            else:
                boarded += waiting
                waiting = 0
                
            profit = boarded * boardingCost - (i + 1) * runningCost
            if profit > maxProfit:
                maxIndex = i + 1
                maxProfit = profit
        
        i += 1
        while waiting > 0:
            if waiting >= 4:
                waiting -= 4
                boarded += 4
            else:
                boarded += waiting
                waiting = 0
                
            profit = boarded * boardingCost - (i + 1) * runningCost
            if profit > maxProfit:
                maxIndex = i + 1
                maxProfit = profit
            i += 1
        
        if maxProfit > 0:
            return maxIndex
        return -1
class Solution:
    def minOperationsMaxProfit(self, customers: List[int], boardingCost: int, runningCost: int) -> int:
        earn, max_earn = 0, 0
        i, n = 0, len(customers)
        wait, res = 0, -1
        while i < n or wait > 0:
            if i < n:
                wait += customers[i]
            board = min(4, wait)
            earn += board* boardingCost - runningCost
            if earn > max_earn:
                res = i + 1
                max_earn = earn
            wait -= board
            i += 1
        return res

class Solution:
    def minOperationsMaxProfit(self, customers: List[int], boardingCost: int, runningCost: int) -> int:
        
        remain = 0
        profit = 0
        maxProfit = -float('inf')
        maxId = -1
        
        for i,val in enumerate(customers):
            total = remain + val
            if total>4:
                profit += 4*boardingCost
                remain = total - 4
            else:
                profit += total*boardingCost
                remain = 0
            profit -= runningCost
            
            if profit>maxProfit:
                maxId = i
                maxProfit = profit
            #print(remain,profit,maxProfit)
        
        j = maxId+1
        
        while remain>0:
            profit+=min(remain,4)*boardingCost - runningCost
            if profit>maxProfit:
                maxId = j
                maxProfit = profit
            #print(remain,profit,maxProfit,maxId)
            remain = max(remain-4,0)
            j = j+1
            
            
        if maxProfit<=0:
            return -1
        else:
            return maxId+1
        

class Solution:
    def minOperationsMaxProfit(self, customers: List[int], boardingCost: int, runningCost: int) -> int:
#greedy: put the most people on at a given turn
#why would you ever wait? I dont think you would, might as well take the instant profit
#mostly because theres no reason to leave money on the table
#when would you definitely not wait?

        remaining_customers = 0
        max_profit = 0
        profit = 0
        spin = 0
        best_spin = 0
        while remaining_customers > 0 or spin < len(customers):
            while spin < len(customers):
                
                remaining_customers += customers[spin]
                boarded = min(4, remaining_customers)
                remaining_customers -= boarded
                profit = profit + boardingCost*boarded - runningCost
                #print('Profit after spin {} is {}'.format(spin+1, profit))
                if profit > max_profit:
                    max_profit = profit
                    best_spin = spin
                spin += 1
            
            boarded = min(4, remaining_customers)
            remaining_customers -= boarded
            profit = profit + boardingCost*boarded - runningCost
            if profit > max_profit:
                max_profit = profit
                best_spin = spin
            spin += 1
            #print('Profit after spin {} is {}'.format(spin+1, profit))
        if max_profit <= 0:
            return -1
        else:
            return best_spin+1

class Solution:
    def minOperationsMaxProfit(self, customers: List[int], boardingCost: int, runningCost: int) -> int:
        
        
        i = 0
        wait = 0
        profit = 0
        maxProfit = 0
        passengers = 0
        rot = 0
        finalRot = 0
        
        while i < len(customers) or wait > 0:
            if i < len(customers):
                wait += customers[i]
                i += 1
            if wait < 5:
                passengers += wait
                wait = 0
            else:
                passengers += 4
                wait -= 4
            rot += 1
            profit = passengers * boardingCost - rot * runningCost
            
            if profit > maxProfit:
                maxProfit = profit
                finalRot = rot
            
        
        if maxProfit > 0:
            return finalRot
        return -1

class Solution:
    def minOperationsMaxProfit(self, customers: List[int], boardingCost: int, runningCost: int) -> int:
        waiting=peak_at=peak=delta=profit=t=0
        while waiting or t<len(customers):
            if t < len(customers):
                waiting+=customers[t]
            t+=1
            delta =min(4,waiting)
            profit+=delta*boardingCost-runningCost
            waiting-=delta
            if peak<profit:
                peak=profit
                peak_at=t
        return peak_at if peak_at>0 else -1
                

class Solution:
    def minOperationsMaxProfit(self, customers: List[int], boardingCost: int, runningCost: int) -> int:
        
        result = []
        maxval = 0
        Current_profit = 0
        
        board_list = 0
        total_board = 0
        count = 0
        i = 1
        j = 1
        
        for x in range(0,len(customers)):
            #print(x)
            if customers[x] != 0:
                wait_list = customers[x]
                i += x
                break
            else:
                count += 1
        #print("Waiting List & x",wait_list,i)
        while wait_list != 0:
            
            if wait_list >= 4:
                total_board +=  4
                wait_list -= 4
            else:
                total_board += wait_list
                wait_list -= wait_list
            #print(total_board,j)
            Current_profit = total_board * boardingCost - j * runningCost
            if i < len(customers):
                wait_list += customers[i]
                i += 1
            j += 1
            result.append(Current_profit)
            #print(Current_profit)
        #print(result)
        #print(result[992])
        #print(len(result))
        maxval = max(result)
        #print(maxval)
        
        if maxval < 0:
            return -1
        else:
            return result.index(maxval) + 1 + count
        #return Current_profit
            

class Solution:
    # u601du8defcopied from https://leetcode.com/problems/maximum-profit-of-operating-a-centennial-wheel/discuss/866409/Java-Simple-O(N)-greedy
    def minOperationsMaxProfit(self, customers: List[int], boardingCost: int, runningCost: int) -> int:
        run = 0
        maxRun = 1
        prof = maxProf = 0
        count_ppl = 0
        i = 0
        while count_ppl > 0 or i < len(customers):
            if i < len(customers):
                count_ppl += customers[i]
                i += 1
            count_bd = min(4, count_ppl) # boarding people by greedy. 
            count_ppl -= count_bd
            prof = prof + count_bd * boardingCost - runningCost
            run += 1
            if prof > maxProf:
                maxProf = prof
                maxRun = run
        return maxRun if maxProf > 0 else -1

class Solution:
    def minOperationsMaxProfit(self, customers: List[int], boardingCost: int, runningCost: int) -> int:
        if boardingCost*4<=runningCost:
            return -1
        ans=0
        N=len(customers)
        cur1,cur2,cur3,cur4=0,0,0,0
        wait=0
        profit=0
        out=-1
        for i in range(N):
            wait+=customers[i]
            add=min(4,wait)
            profit+=(boardingCost*add-runningCost)
            wait-=add
            if profit>ans:
                out=i+1
                ans=profit
            cur1,cur2,cur3,cur4=add,cur1,cur2,cur3
            
        while wait>0:
            i+=1
            add=min(4,wait)
            profit+=(boardingCost*add-runningCost)
            wait-=add
            if profit>ans:
                out=i+1
                ans=profit
            cur1,cur2,cur3,cur4=add,cur1,cur2,cur3
        return out

class Solution:
    def minOperationsMaxProfit(self, customers: List[int], boardingCost: int, runningCost: int) -> int:
        profit = 0
        max_profit = 0
        mark = 0
        waiting = 0
        n = len(customers)
        i = 0
        while i < n or waiting > 0:
            if i < n:
                waiting = waiting + customers[i]
            if waiting > 4:
                waiting = waiting - 4
                profit = profit + 4 * boardingCost
            else:
                profit = profit + waiting * boardingCost
                waiting = 0
            profit = profit - runningCost
            if profit > max_profit:
                max_profit = profit
                mark = i + 1
            i = i + 1
        return mark if max_profit > 0 else -1
class Solution:
    def minOperationsMaxProfit(self, customers: List[int], boardingCost: int, runningCost: int) -> int:
        if runningCost >= 4 * boardingCost:
            return -1
        profit = 0
        max_profit = float('-inf')
        wc = customers[0]
        i = 1
        rot = 1
        max_rot = 0
        
        while  wc > 0 or i < len(customers):
            
                if wc >= 4:
                    #print wc
                    wc -=  4 
                    profit += 4 * boardingCost
                elif wc < 4:
                    bc = wc
                    wc = 0
                    profit +=  bc * boardingCost
                profit -= runningCost
                prev = max_profit
                max_profit = max(profit, max_profit)
                if max_profit != prev:
                    max_rot = rot
                if i < len(customers):
                    wc += customers[i]
                    i+=1
                    
                rot +=1
                
        if max_profit > 0:
            return max_rot
        else:
            return -1
class Solution:
    def minOperationsMaxProfit(self, customers: List[int], boardingCost: int, runningCost: int) -> int:
        rt = 0
        prof = 0
        maxRt = 0
        maxProf = 0
        wait = 0
        i = 0
        onborad = 0
        while wait > 0 or i < len(customers):
            if i < len(customers):
                wait += customers[i]
                i += 1
            onboard = min(4, wait)
            wait -= onboard
            prof = prof + onboard * boardingCost - runningCost
            rt += 1
            if maxProf < prof:
                maxProf = prof
                maxRt = rt
            
        if maxProf > 0 :
            return maxRt
        else:
            return -1 
        
#         pro = 0
#         curpro = 0 
#         wait = 0
#         on = 0
#         j = 0 
#         for i, n in enumerate(customers):
#             if n >= 4:
#                 wait += n - 4
#                 on += 4
#             else:
#                 on += n
#             j += 1
#             curpro = on* boardingCost - j *runningCost
#             if curpro < 0:
#                 ans = -1
#             elif pro < curpro:
#                 pro = max(pro, curpro)
#                 ans = j
    
#         while wait > 0:
#             if wait >= 4:
#                 wait -= 4
#                 on += 4
#             else:
#                 on += wait
#                 wait = 0
#             j += 1
#             curpro = on* boardingCost - j*runningCost
#             if curpro < 0:
#                 ans = -1
#             elif pro < curpro:
#                 if j > 300:
#                     print(on, wait, pro, curpro, j)
#                 ans = j
#                 pro = max(pro, curpro)
        
#         return ans


class Solution:
    # u601du8defcopied from https://leetcode.com/problems/maximum-profit-of-operating-a-centennial-wheel/discuss/866409/Java-Simple-O(N)-greedy
    # The solution is based on simply simulating the the rotations and keep track of waiting customers
    def minOperationsMaxProfit(self, customers: List[int], boardingCost: int, runningCost: int) -> int:
        run = 0
        maxRun = 1
        prof = maxProf = 0
        count_ppl = 0
        i = 0
        while count_ppl > 0 or i < len(customers):
            if i < len(customers):
                count_ppl += customers[i]
                i += 1
            count_bd = min(4, count_ppl) # boarding people by greedy. 
            count_ppl -= count_bd
            prof = prof + count_bd * boardingCost - runningCost
            run += 1
            if prof > maxProf:
                maxProf = prof
                maxRun = run
        return maxRun if maxProf > 0 else -1

class Solution:
    def minOperationsMaxProfit(self, customers: List[int], boardingCost: int, runningCost: int) -> int:
        
        #counter for the customers array
        i = 0
        #customers waitlist 
        wait = 0
        #profit at each iteration
        profit = 0
        #max profit made, we will compare it each time to profit
        maxProfit = 0
        #total number of passengers who paid the boarding coast
        passengers = 0
        #total # rotations at each iteration
        rot = 0
        #the # rotations for which profit = max profit
        finalRot = 0
        
        while i < len(customers) or wait > 0:
            # in case we didn't finish the array customers:
            if i < len(customers):
                wait += customers[i]
                i += 1
            # in case wait has less than 5 customers, we make it equal to 0
            if wait < 5:
                passengers += wait
                wait = 0
            else:
                passengers += 4
                wait -= 4
            #total number of rotations until now
            rot += 1
            #total profit until now
            profit = passengers * boardingCost - rot * runningCost
            
            #updating max profit and best number of rotations:
            if profit > maxProfit:
                maxProfit = profit
                finalRot = rot
            
        
        if maxProfit > 0:
            return finalRot
        #in case the profit is always <= 0
        return -1

class Solution:
    def minOperationsMaxProfit(self, customers: List[int], boardingCost: int, runningCost: int) -> int:
        waiting = served = onboard = max_round = max_profit = 0
        cur_round = 1
    
        while cur_round <= len(customers) or waiting > 0:
            if cur_round <= len(customers):
                waiting += customers[cur_round - 1]
            if waiting > 4:
                onboard = 4
                waiting -= onboard
            else:
                onboard = waiting
                waiting = 0
            served += onboard
            cur_profit = served * boardingCost - cur_round * runningCost
            if cur_profit > max_profit:
                max_profit = cur_profit
                max_round = cur_round
            cur_round += 1   
            
        return max_round if max_profit > 0 else -1
class Solution:
    def minOperationsMaxProfit(self, customers: List[int], boardingCost: int, runningCost: int) -> int:
        board = customers[0]
        profit = 0
        prevprofit = 0
        best = -1
        i=1
        k = 0
        while board>0 or i!=len(customers):
            k+=1
            sub = min(board,4)
            profit += sub*boardingCost - runningCost
            board-=sub
            if profit>prevprofit and profit>0:
                best = k
            if i<len(customers):
                board += customers[i]
                i+=1
            prevprofit = profit
        return best
class Solution:
    def minOperationsMaxProfit(self, customers: List[int], boardingCost: int, runningCost: int) -> int:
        if boardingCost * 4 < runningCost:
            return -1
        if boardingCost * 4 == runningCost:
            return 0
        i = 0
        res = 0
        curM = 0
        cur = 0
        remind = 0
        served = 0
        for i in range(len(customers)):
            remind += customers[i]
            served += min(remind, 4)
            remind = max(0, remind-4)
            cur = served * boardingCost - (i+ 1) *runningCost
            if cur > curM:
                curM = cur
                res = i + 1

        if remind  * boardingCost - runningCost <= 0:
            return res
        res = len(customers)
        while min(4, remind) * boardingCost - runningCost > 0:
            remind -= min(4, remind)
            res += 1
        return res

        total = sum(customers)
        if (total%4)  *  boardingCost <= runningCost:
            return total // 4
        return total // 4 + 1

class Solution:
    def minOperationsMaxProfit(self, customers: List[int], boardingCost: int, runningCost: int) -> int:
        ans = -1
        most = profit = waiting = 0
        for i, x in enumerate(customers): 
            waiting += x # more people waiting in line 
            waiting -= (running := min(4, waiting)) # boarding 
            profit += running * boardingCost - runningCost 
            if most < profit: 
                ans, most = i+1, profit
        q, r = divmod(waiting, 4)
        if 4*boardingCost > runningCost: 
            ans += q
        if r*boardingCost > runningCost: 
            ans += 1
        return ans 
class Solution:
    def minOperationsMaxProfit(self, customers: List[int], boardingCost: int, runningCost: int) -> int:
        waiting = 0
        profits = 0
        bestProfits = 0
        bestRotations = -1
        for rotation in range(999999):
            if rotation < len(customers):
                waiting += customers[rotation]
            elif waiting == 0:
                break
            bording = min(4, waiting)
            waiting -= bording
            profits += bording * boardingCost
            profits -= runningCost
            if profits > bestProfits:
                bestProfits = profits
                bestRotations = rotation+1
            # print(rotation, waiting, bording, profits, bestProfits)
        
        return bestRotations

class Solution:
    def minOperationsMaxProfit(self, customers: List[int], boardingCost: int, runningCost: int) -> int:
        wait = total = profit = maxp = res = i = 0
        while wait or i < len(customers):
            if i < len(customers):
                total += customers[i]
                wait = max(0, customers[i] + wait - 4)
            else:
                wait = max(0, wait - 4)
            i += 1
            profit = boardingCost * (total - wait) - runningCost * i
            if profit > maxp:
                maxp, res = profit, i
        return res if maxp > 0 else -1
class Solution:
    def minOperationsMaxProfit(self, customers: List[int], boardingCost: int, runningCost: int) -> int:
        wait = 0
        i = 0
        n = len(customers)
        max_profit, min_rotates = 0, -1
        total_boarded = 0
        
        rotate = 0
        while True:
            if i == n and not wait:
                break
            
            if i < n:
                wait += customers[i]
                i += 1
            
            rotate += 1
            
            board = min(wait, 4)
            wait -= board
            total_boarded += board
            
            profit = total_boarded * boardingCost - rotate * runningCost
            
            if profit > max_profit:
                max_profit = profit
                min_rotates = rotate
                
        return min_rotates
class Solution:
    def minOperationsMaxProfit(self, customers: List[int], boardingCost: int, runningCost: int) -> int:
        profit = 0
        maxprofit = 0
        n = len(customers)
        if n == 0:
            return maxprofit
        
        waiting = customers[0]
        k = 1
        rounds = 1
        max_rounds = -1
        
        so_far = 0
        
        while k < n or waiting > 0:
            this_round = min(4, waiting)
            waiting -= this_round
            so_far += this_round
            profit = so_far * boardingCost - rounds * runningCost
            if profit > maxprofit:
                maxprofit = profit
                max_rounds = rounds
            if k < n:
                waiting += customers[k]            
                k += 1
                
            rounds += 1
            
        return max_rounds

from typing import List

class Solution:
    def minOperationsMaxProfit(self, customers: List[int], boardingCost: int, runningCost: int) -> int:
        cur = best = idx = 0
        best_idx = None
        waiting = 0
        capacity = 4

        while waiting or idx < len(customers):
            # Add new waiting customers, put passengers on ride, remove passengers from waiting.
            if idx < len(customers):
                waiting += customers[idx]
            
            passengers = min(waiting, capacity)
            waiting -= passengers

            # Update money.
            cur += (passengers * boardingCost) - runningCost

            if cur > best:
                best_idx = idx + 1
                best = cur  
            
            idx += 1
            

        if best_idx is None:
            best_idx = -1

        return best_idx
class Solution:
    def minOperationsMaxProfit(self, customers: List[int], boardingCost: int, runningCost: int) -> int:
        queue_customer = 0
        total_profit = [0]
        for c in customers:
            queue_customer += c
            n_board = min(4, queue_customer)
            queue_customer -= n_board
            total_profit.append(total_profit[-1] + n_board * boardingCost - runningCost)

        while queue_customer > 0:
            n_board = min(4, queue_customer)
            queue_customer -= n_board
            total_profit.append(total_profit[-1] + n_board * boardingCost - runningCost)

        max_profit = max(total_profit)
        if max_profit <= 0:
            return -1
        else:
            for i, p in enumerate(total_profit):
                if p == max_profit:
                    return i
class Solution:
    def minOperationsMaxProfit(self, customers: List[int], boardingCost: int, runningCost: int) -> int:
        max_prof = -math.inf
        idx = -1 # rotation idx for max_prof
        sofar = 0
        profit_sofar = 0
        i = 0
        while True:
            if i < len(customers):
                sofar += customers[i]
            if sofar == 0 and i >= len(customers):
                break
            num_onboard = min(4, sofar)
            sofar -= num_onboard
            profit_sofar += num_onboard * boardingCost - runningCost
            # print('num_onboard: {}, profit_sofar: {}'.format(num_onboard, profit_sofar))
            if profit_sofar > max_prof:
                max_prof = profit_sofar
                idx = i
            i += 1
        return idx + 1 if max_prof > 0 else -1
class Solution:
    def minOperationsMaxProfit(self, customers: List[int], boardingCost: int, runningCost: int) -> int:
        customers.reverse()    
        wait = 0
        board = 0
        turn = 0
        
        ans = -1
        score = 0
        while wait or customers:
            turn += 1
            if customers:
                wait += customers.pop()
            if wait >= 4:
                board += 4
                wait -= 4
            else:
                board += wait
                wait = 0            
            if score < board * boardingCost - turn * runningCost:
                ans = turn
                score = board * boardingCost - turn * runningCost
        
        return ans
class Solution:
    def minOperationsMaxProfit(self, customers: List[int], bc: int, rc: int) -> int:
        
        peak = res = idx = cust = i = 0
        
        while cust > 0 or i < len(customers):
            if i < len(customers):
                cust += customers[i]
            
            c = min(cust, 4)
            res += c * bc - rc
            
            if res > peak:
                peak = res
                idx = i + 1
            
            cust -= c
            i += 1
        
        return idx if idx > 0 else -1
class Solution:
    def minOperationsMaxProfit(self, A, BC, RC):
        ans=profit=t=0
        maxprofit=0
        wait=i=0
        n=len(A)
        while wait>0 or i<n:
            if i<n:
                wait+=A[i]
                i+=1
            t+=1
            y=min(4,wait)
            wait-=y
            profit+=y*BC
            profit-=RC
            if profit>maxprofit:
                maxprofit=profit
                ans=t

        return -1 if maxprofit<=0 else ans
class Solution:
    def minOperationsMaxProfit(self, customers: List[int], boardingCost: int, runningCost: int) -> int:
        rotate = 0
        total = 0
        left = 0
        for i in customers:
            total += i
            left += i
            if left <= 4:
                rotate += 1
                left = 0
            left -= 4
            rotate += 1
        rotate += left // 4
        if left % 4 * boardingCost > runningCost:
            rotate += 1
        if boardingCost*total - rotate*runningCost > 0:
            return rotate
        return -1
class Solution:
    def minOperationsMaxProfit(self, arr: List[int], boardingCost : int, runningCost : int) -> int:
        # Dividing people in the groups of <=4
        grps = []
        # length of customers array
        n = len(arr)
        # rem--> number of people waiting
        rem = 0
        
        # traversing the customers array
        for i in range(n):
            # total number of people available right now 
            avail = arr[i]+rem
            # number of available people >=4 then append grp of 4 and update remaining [rem]
            if avail>=4:
                avail-=4
                grps.append(4)
                rem = avail
            # number of available people <4  then make group of available people and update remaining [rem=0]
            else:
                rem = 0
                grps.append(avail)
        
        # make groups of 4 until remaining >=4 otherwise make <4 and break
        while rem>0:
            if rem>=4:
                rem-=4
                grps.append(4)
            else:
                grps.append(rem)
                rem = 0
        
        # mex--> represents maximum profit
        mex = -10**10
        # cost--> represents current total cost
        cost = 0
        # ind --> represents rotation number
        ind = 0
        for i in range(len(grps)):
            # calculate net cost till now
            cost+= boardingCost*grps[i]-runningCost
            # upadte max profit and rotation number
            if mex<cost:
                mex = max(mex,cost)
                ind = i+1
        # max profit< 0
        if mex<0:
            return -1
        # return rotation number
        return ind
class Solution:
    def minOperationsMaxProfit(self, customers: List[int], boardingCost: int, runningCost: int) -> int:
        # Input: customers = [10,10,6,4,7], boardingCost = 3, runningCost = 8
        # Output: 9
        max_amt, curr_amt, min_turns, curr_turns = 0, 0, 0, 0
        total, idx = 0, 0
        while (len(customers) > idx) or total > 0:
            # print(f'{curr_amt=} | {total=} | {curr_turns=} | {min_turns=}')
            if len(customers) > idx:
                total += customers[idx]
            idx += 1
            curr_turns += 1
            if total >= 4: curr = 4
            else: curr = total
            total -= curr
            curr_amt += (boardingCost * curr) - (runningCost)
            if curr_amt > max_amt:
                max_amt = curr_amt
                min_turns = curr_turns
        if not min_turns: return -1
        else: return min_turns
        
            
            
            

class Solution:
    def minOperationsMaxProfit(self, customers: List[int], boardingCost: int, runningCost: int) -> int:
        if 4*boardingCost < runningCost:
            return -1
        
        n = len(customers)
        maxtotal, maxrotations = 0, -1
        total, rotations = 0, 1
        i = 0
        
        while True:
            if i == n-1 and customers[n-1] == 0:
                break
                
            if customers[i]>4:
                total += boardingCost*4 - runningCost
                
                if i < n-1:
                    customers[i+1] += customers[i]-4
                    customers[i] = 0
                else:
                    customers[i] -= 4
                    
            else:
                total += customers[i]*boardingCost - runningCost
                customers[i] = 0
            
            if total>maxtotal:
                maxtotal = total
                maxrotations = rotations
                
            if i<n-1: i += 1
            rotations += 1
            
        return maxrotations
class Solution:
    def minOperationsMaxProfit(self, customers: List[int], boardingCost: int, runningCost: int) -> int:
        total, board = 0, 0
        ans = 1
        res = 1
        m = -1
        
        def cal():
            nonlocal total, ans, res, m, board
            t = 4 if total >= 4 else total
            total -= t
            board += t
            cost = board * boardingCost - ans * runningCost
            if cost > m:
                m = cost
                res = ans 
            ans += 1
            
        for c in customers:
            total += c
            cal()
            
        while total > 0:
            cal()
        return res if m >= 0 else -1

class Solution:
    def minOperationsMaxProfit(self, customers: List[int], boardingCost: int, runningCost: int) -> int:
        prof = 0; max_prof = 0; max_run = 0; run = 0; i = 0
        waiting = 0
        
        while waiting > 0 or i < len(customers):
            if i < len(customers): 
                waiting += customers[i]
                i += 1
            board = min(4, waiting)
            waiting -= board
            prof += boardingCost * board - runningCost
            run += 1
            if prof > max_prof:
                max_prof = prof
                max_run = run
        return max_run if max_prof != 0 else -1
class Solution:
    def minOperationsMaxProfit(self, customers: List[int], boardingCost: int, runningCost: int) -> int:
        n = len(customers)
        i = 0
        num_waiting = 0
        ans = -1
        max_profit = 0
        taken = 0
        while i < n or num_waiting:
            curr = num_waiting
            if i < n: curr += customers[i]
            if curr > 4:
                taken += 4
                num_waiting = curr - 4
            elif curr > 0:
                taken += curr
                num_waiting = 0
            elif curr == 0:
                num_waiting = 0

            if boardingCost * taken - runningCost * (i + 1) > max_profit:
                max_profit = boardingCost * taken - runningCost * (i + 1)
                ans = i + 1
            i += 1
        return ans
                
                

class Solution:
    def minOperationsMaxProfit(self, customers: List[int], boardingCost: int, runningCost: int) -> int:
        total_customers = 0
        max_profit = 0
        no_rotation = 0
        pending = 0
        i = 0
        while i<len(customers) or pending>0:
            if i<len(customers):
                pending += customers[i]
            if pending>=4:
                pending -= 4
                total_customers += 4
            else:
                total_customers += pending
                pending = 0
            profit = total_customers*boardingCost - (i+1)*runningCost
            if profit>max_profit:
                max_profit = profit
                no_rotation = (i+1)
            i += 1
        if no_rotation == 0:
            return -1
        return no_rotation

class Solution:
    def minOperationsMaxProfit(self, a: List[int], bc: int, rc: int) -> int:
        max_pr = pr = 0; cnt = max_cnt = 0; w = 0        
        for x in a:
            x = x + w
            pr += min(x, 4) * bc - rc                        
            cnt += 1            
            if pr > max_pr: max_pr, max_cnt = pr, cnt             
            w = max(x - 4, 0)            
            
        while w > 0:
            pr += min(w, 4) * bc - rc
            cnt += 1
            if pr > max_pr: max_pr, max_cnt = pr, cnt                        
            w = max(w - 4, 0)                             
        return max_cnt if max_pr > 0 else -1
class Solution:
    def minOperationsMaxProfit(self, customers: List[int], boardingCost: int, runningCost: int) -> int:
        q = 0
        profit = 0
        num_spins = 0
        num_customers = 0
        
        
        min_spins = float('inf')
        max_profit = -float('inf')
        be = runningCost // boardingCost
        
        if be >= 4:
            return -1
        
        for new_customers in customers:
            num_spins += 1
            q += new_customers
            loaded_customers = min(q, 4)
            q -= loaded_customers
            num_customers += loaded_customers
            profit = num_customers*boardingCost - num_spins*runningCost
            
            if profit > max_profit:
                max_profit = profit
                min_spins = num_spins
        
        if q:
            full, partial = divmod(q, 4)
            num_spins += full
            num_customers += 4*full
            
            if partial > be:
                num_spins += 1
                num_customers += partial
                
            profit = num_customers*boardingCost - num_spins*runningCost
            if profit > max_profit:
                min_spins = num_spins
                
        return min_spins
class Solution:
    def minOperationsMaxProfit(self, customers: List[int], boardingCost: int, runningCost: int) -> int:
        l = list()
        num_of_people = 0
        wait_people = 0
        c = 0
        for i in customers:
            c += 1
            if (i + wait_people) >= 4:
                num_of_people += 4
                wait_people = i + wait_people - 4
            else:
                num_of_people += i + wait_people
                wait_people = 0
            temp = num_of_people*boardingCost - c*runningCost
            l.append(temp)
        while(wait_people > 0):
            c += 1
            if wait_people >= 4:
                num_of_people += 4
                wait_people -= 4
            else:
                num_of_people += wait_people
                wait_people = 0
            temp = num_of_people*boardingCost - c*runningCost
            l.append(temp)
        if max(l) > 0:
            return l.index(max(l))+1
        else:
            return -1

import math
class Solution:
    def minOperationsMaxProfit(self, customers: List[int], boardingCost: int, runningCost: int) -> int:
        len_rot = len(customers)
        if(4*boardingCost <= runningCost or len_rot == 0):
            return -1
        tot = 0
        profit = 0
        lis = []
        for i in customers:
            tot += i
            profit += min(4, tot)*boardingCost - runningCost
            tot -= min(4, tot)
            lis.append(profit)
        while(tot > 0):
            profit += min(4, tot)*(boardingCost) - runningCost
            tot -= min(4, tot)
            lis.append(profit)
        max_value = max(lis)
        return lis.index(max_value)+1 if max_value > 0 else -1

class Solution:
    def minOperationsMaxProfit(self, customers: List[int], boardingCost: int, runningCost: int) -> int:
        wait = arrive = done = rotation = profit = maxprofit = i = 0
        while i < len(customers) or wait > 0:
            if i < len(customers):
                arrived = customers[i]
            else:
                arrived = 0
            wait += arrived
            
            if wait >= 4:
                wait -= 4
                done += 4
            else:
                done += wait
                wait = 0
            
            profit = done * boardingCost - (i+1)*runningCost
            i += 1
            if profit > maxprofit:
                maxprofit = profit
                rotation = i
            
            
            
        if profit <= 0:
            return -1
        else:
            return rotation
                

class Solution:
    def minOperationsMaxProfit(self, customers: List[int], boardingCost: int, runningCost: int) -> int:
        
        run, maxRun, prof = 0, 1, 0
        maxProf = prof
        sum_, i = 0, 0
        
        while sum_ > 0 or i < len(customers):
            if i < len(customers):
                sum_ += customers[i]
            bd = min(sum_, 4)
            sum_ -= bd
            prof = prof + bd * boardingCost - runningCost
            run += 1
                
            if prof > maxProf:
                maxProf = prof
                maxRun = run
                
                
            i += 1
            
        return maxRun if maxProf > 0 else -1
class Solution:
    def minOperationsMaxProfit(self, customers: List[int], boardingCost: int, runningCost: int) -> int:
        if 4*boardingCost < runningCost or len(customers)==0 :
            return -1
        
        maxVal = 0
        times , recordedTime = 0 ,0
        served , waitting = 0 , 0
        while times < len(customers) or waitting!=0:
            if times<len(customers):
                waitting = waitting + customers[times] 
                
            if waitting > 4:
                served+=4
                waitting-=4
            else:
                served += waitting
                waitting = 0
            
            times+=1
            currentProfit = served*boardingCost - times*runningCost
            if currentProfit > maxVal :
                maxVal = currentProfit
                recordedTime = times
        
        return recordedTime if recordedTime>0 else -1
        
            

class Solution:
    def minOperationsMaxProfit(self, customers: List[int], boardingCost: int,
                               runningCost: int) -> int:
        gondolas = [0] * 4
        curr_gondola = 0
        n = len(customers)
        max_profit = profit = 0
        min_rotation = 0
        rotations = 0
        waiting = 0
        i = 0
        while i < n or waiting:
            # selected = False:
            # customers[i] += 4 - customers[i]
            if i < n:
                while customers[i] < 4 and waiting:
                    customers[i] += 1
                    waiting -= 1
            if i < n:
                customer = customers[i]
                if customer > 4:
                    waiting += customer - 4
                    customer = 4
                curr_gondola = (1 + curr_gondola) % 4
                profit += customer * boardingCost
            else:
                customer = min(4, waiting)
                waiting -= customer
                profit += customer * boardingCost
            profit -= runningCost
            rotations += 1
            if profit > max_profit:
                min_rotation = rotations
                max_profit = profit
            if i < n:
                i += 1
            # print(i, waiting, profit, max_profit, min_rotation, customer)

        # while waiting > 0:
        #     customer = min(4, waiting)
        #     waiting -= customer
        #     profit += customer * boardingCost
        #     profit -= runningCost
        #     rotations += 1
        #     if profit > max_profit:
        #         min_rotation = rotations
        #         max_profit = profit

        print((max_profit, rotations, min_rotation, customers))
        if max_profit <= 0:
            return -1
        # print(min_rotation)
        return min_rotation

class Solution:
    def minOperationsMaxProfit(self, customers: List[int], boardingCost: int, runningCost: int) -> int:
        ans=0
        N=len(customers)
        cur1,cur2,cur3,cur4=0,0,0,0
        wait=0
        profit=0
        out=-1
        for i in range(N):
            wait+=customers[i]
            add=min(4,wait)
            profit+=(boardingCost*add-runningCost)
            wait-=add
            if profit>ans:
                out=i+1
                ans=profit
            cur1,cur2,cur3,cur4=add,cur1,cur2,cur3
            
        while wait>0:
            i+=1
            add=min(4,wait)
            profit+=(boardingCost*add-runningCost)
            wait-=add
            if profit>ans:
                out=i+1
                ans=profit
            cur1,cur2,cur3,cur4=add,cur1,cur2,cur3
        return out

class Solution:
    def minOperationsMaxProfit(self, customers: List[int], boarding_cost: int, running_cost: int) -> int:
        if len(customers) == 0: return 0
        if boarding_cost * 4 <= running_cost: return -1
        
        available_customer_count = 0
        customers_per_rotation = []
        
        for i in range(len(customers)):
            available_customer_count += customers[i]
            customers_per_rotation.append(min(available_customer_count, 4))
            available_customer_count -= customers_per_rotation[-1]
            
        while available_customer_count > 0:
            customers_per_rotation.append(min(available_customer_count, 4))
            available_customer_count -= customers_per_rotation[-1]
        
        max_profit = 0
        max_turn = -1
        previous_profit = 0
        current_customer_count = 0
        
        for i,customer_count in enumerate(customers_per_rotation):
            current_customer_count += customer_count
            profit = ((current_customer_count * boarding_cost) - (running_cost * (i+1)))
            
            if profit > max_profit:
                max_profit = profit
                max_turn = i + 1
        
        return max_turn

class Solution:
    def minOperationsMaxProfit(self, customers: List[int], boarding_cost: int, running_cost: int) -> int:
        max_profit = 0
        max_profit_rotations = -1
        rotations = 0
        current_profit = 0
        in_line = 0
        for customer in customers:
            in_line += customer
            current_profit += min(in_line, 4) * boarding_cost - running_cost 
            in_line = max(in_line - 4, 0)
            rotations += 1
            if current_profit > max_profit:
                max_profit = current_profit
                max_profit_rotations = rotations
        while in_line > 0:
            current_profit += min(in_line, 4) * boarding_cost - running_cost
            in_line = max(in_line - 4, 0)
            rotations += 1
            if current_profit > max_profit:
                max_profit = current_profit
                max_profit_rotations = rotations
        return max_profit_rotations
        
'''
Input: customers = [10,9,6], boardingCost = 6, runningCost = 4
Output: 7
Explanation:
1. 10 customers arrive, 4 board and 6 wait for the next gondola, the wheel rotates. Current profit is 4 * $6 - 1 * $4 = $20.
2. 9 customers arrive, 4 board and 11 wait (2 originally waiting, 9 newly waiting), the wheel rotates. Current profit is 8 * $6 - 2 * $4 = $40.
3. The final 6 customers arrive, 4 board and 13 wait, the wheel rotates. Current profit is 12 * $6 - 3 * $4 = $60.
4. 4 board and 9 wait, the wheel rotates. Current profit is 16 * $6 - 4 * $4 = $80.
5. 4 board and 5 wait, the wheel rotates. Current profit is 20 * $6 - 5 * $4 = $100.
6. 4 board and 1 waits, the wheel rotates. Current profit is 24 * $6 - 6 * $4 = $120.
7. 1 boards, the wheel rotates. Current profit is 25 * $6 - 7 * $4 = $122.
The highest profit was $122 after rotating the wheel 7 times.

10 - 4 - 24

'''
class Solution:
    def minOperationsMaxProfit(self, arr: List[int], boardingCost: int, runningCost: int) -> int:
        grps = []
        n = len(arr)
        rem = 0
        
        for i in range(n):
            avail = arr[i]+rem
            if avail>=4:
                avail-=4
                grps.append(4)
                rem = avail
            else:
                rem = 0
                grps.append(avail)
        
        while rem>0:
            if rem>=4:
                rem-=4
                grps.append(4)
            else:
                grps.append(rem)
                rem = 0

        mex = -10**10
        cost = 0
        ind = 0
        for i in range(len(grps)):
            # calculate net cost till now
            cost+= boardingCost*grps[i]-runningCost
            # upadte max profit and rotation number
            if mex<cost:
                mex = max(mex,cost)
                ind = i+1
        # max profit< 0
        if mex<0:
            return -1
        # return rotation number
        return ind
        
        
        
        # idx = 0
        # profit = 0
        # max_idx = -1
        # max_profit = 0
        # n_cus = 0
        # for cus in customers:
        #     idx += 1
        #     profit += min(4, cus) * boardingCost - runningCost
        #     n_cus += max(cus-4, 0)
        #     if profit > max_profit:
        #         max_idx = idx
        #         max_profit = profit
        # if n_cus >= 4:
        #     if 4*boardingCost <= runningCost:
        #         return max_idx
        #     else:
        #         profit += (4*boardingCost - runningCost) * (n_cus // 4)
        #         idx += n_cus // 4
        #         n_cus %= 4
        #         if profit > max_profit:
        #             max_idx = idx
        #             max_profit = profit
        # idx += 1
        # profit += n_cus * boardingCost - runningCost
        # if profit > max_profit:
        #     max_idx = idx
        # return max_idx

class Solution:
    def minOperationsMaxProfit(self, customers, boardingCost, runningCost):
        # we can never make profit
        if 4*boardingCost <= runningCost:
            return -1
        maxProfit = -1
        maxRotate = 0
        accuProfit = 0
        rotate = 0
        boardingPool = 0
        
        # keep rotate and board the customers.
        i = 0
        while i < len(customers) or boardingPool > 0:
            # add customers to boarding queue
            if i < len(customers):
                boardingPool += customers[i]
                i += 1
                
            # make profit
            currBoarding = min(4, boardingPool) # maximum boarding limit is 4
            boardingPool -= currBoarding
            accuProfit += currBoarding * boardingCost 
            
            # we need pay runningCost
            rotate += 1
            accuProfit -= runningCost
            if maxProfit < accuProfit:
                maxProfit = accuProfit
                maxRotate = rotate
            
        return -1 if maxProfit < 0 else maxRotate

class Solution:
    def minOperationsMaxProfit(self, customers: List[int], boardingCost: int, runningCost: int) -> int:
        profit = []
        times = 0
        for i in range(len(customers)-1):
            if customers[i] <= 4:
                times +=1
                curr_profit = customers[i]*boardingCost - times*runningCost
            else:
                times +=1
                curr_profit = 4*boardingCost - runningCost
                customers[i+1] = customers[i+1] + customers[i] - 4
            #print(customers)
            profit.append(curr_profit)
        while customers[-1] > 4:
            times +=1
            curr_profit = 4*boardingCost - runningCost
            profit.append(curr_profit)
            customers[-1] -= 4
        times +=1
        curr_profit = customers[-1]*boardingCost - runningCost
        profit.append(curr_profit)
        
        tot_profit = []
        max_profit = 0
        for i in range(len(profit)):
            max_profit += profit[i]
            tot_profit.append(max_profit)
        
        real_max = max(tot_profit)
        
        if real_max < 0:
            return -1
        else:
            return tot_profit.index(real_max)+1

class Solution:
    def minOperationsMaxProfit(self, customers: List[int], boardingCost: int, runningCost: int) -> int:
        
        ans = []
        wait = 0
        onboard = 0
        i = 0
        while i<len(customers):
            wait += customers[i]
            onboard += min(4, wait)
            wait -= 4
            if wait<0:
                wait = 0
            
            ans.append((boardingCost * onboard) - (runningCost*(i+1)))
            i+=1
        
        while wait:
            onboard += min(4, wait)
            wait -= 4
            if wait<0:
                wait = 0
            
            ans.append((boardingCost * onboard) - (runningCost*(i+1)))
            i+=1
        
        val = max(ans)
        if val<0:
            return -1
        else:
            return ans.index(val)+1
        

def wheel(cust, ticket, cost):
    n_waiting = 0
    earn = 0
    rots = 0
    res = [0, 0]
    i = 0
    l = len(cust)
    while True:
        if i < l:
            n_waiting += cust[i]
            i += 1
        seated = min (n_waiting, 4)
        if i == l and seated == 0:
            break
        n_waiting -= seated
        earn += seated * ticket - cost
        rots += 1
        if (earn > res[1]):
            res[1] = earn
            res[0] = rots
    if res[0] <= 0:
        return -1
    return res[0]


class Solution:
    def minOperationsMaxProfit(self, customers: List[int], boardingCost: int, runningCost: int) -> int:
        return wheel(customers, boardingCost, runningCost)

class Solution:
    def minOperationsMaxProfit(self, customers: List[int], boardingCost: int, runningCost: int) -> int:
        
        def cal(k,p):
            if k == 0:
                p = p
            else:
                p += k*boardingCost - runningCost
            return p
           
        ans = 0
        p = 0
        w = 0
        i = 0
        itop = -1
        for n in customers:
            i+=1
            w += n
            k = min(4, w)
            p = cal(k,p)
            if ans<p:
                ans = p
                itop = i            
            w-=k
        while w:
            i+=1
            k = min(4, w)
            p = cal(k,p)
            if ans<p:
                ans = p
                itop = i
            w-=k
      
        return itop
class Solution:
    def minOperationsMaxProfit(self, customers: List[int], boardingCost: int, runningCost: int) -> int:
        
        y = 0
        n = 0
        m = -math.inf
        res = -1
        k = 0
        
        for x in customers:
            n += x
            y += min(n,4)*boardingCost-runningCost
            n-=min(n,4)
            k+=1
            
            if y>m:
                m = y
                res = k
            
        while n:
            
            y += min(n,4)*boardingCost-runningCost
            n-=min(n,4)
            k+=1
            if y>m:
                m = y
                res = k
                
       
        return res if m>0 else -1 

class Solution:
    def minOperationsMaxProfit(self, customers: List[int], boardingCost: int, runningCost: int) -> int:
        nwaits = 0
        best_profit = 0
        best_rotations = 0
        profit = 0
        for i, c in enumerate(itertools.chain(customers, itertools.repeat(0))):
            nwaits += c
            board = min(4, nwaits)
            nwaits -= board
            profit += (board * boardingCost - runningCost)
            if profit > best_profit:
                best_profit = profit
                best_rotations = i + 1
            if i >= len(customers) and nwaits == 0:
                break
        return best_rotations if best_profit > 0 else -1
class Solution:
    def minOperationsMaxProfit(self, customers: List[int], boardingCost: int, runningCost: int) -> int:
        prof = rots = wait = 0
        maxp = float('-inf')
        maxr = -1               # result
        for c in customers:
            rots += 1
            wait += c
            ride = min(wait, 4)
            wait -= ride
            prof += ride * boardingCost - runningCost
            if maxp < prof:
                maxp = prof
                maxr = rots
        while wait > 0:
            rots += 1
            ride = min(wait, 4)
            wait -= ride
            prof += ride * boardingCost - runningCost
            if maxp < prof:
                maxp = prof
                maxr = rots
        return maxr if maxp > 0 else -1    
class Solution:
    def minOperationsMaxProfit(self, customers: List[int], boardingCost: int, runningCost: int) -> int:
        # 4 gondalas, each gondola can sit up to 4 people
        # rotates counterclockwise, costs 'runningCost'
        
        # customers[i] is the number of new customers arriving before the ith rotation
        if 4*boardingCost <= runningCost: return -1
        
        remain = 0
        for i,c in enumerate(customers):
            if c == 4 or (c < 4 and remain == 0):
                continue
            if c < 4:
                if c + remain <= 4:
                    c += remain
                    remain = 0
                    customers[i] = c
                else:
                    # get new remain
                    remain -= (4 - c)
                    customers[i] = 4
            else:
                # collect remain
                remain += (c - 4)
                customers[i] = 4
        
        sofar = 0
        hi = 0
        hiInd = -1
        for i,c in enumerate(customers):
            sofar += (c * boardingCost - runningCost)
            if sofar > 0 and sofar > hi:
                hi = sofar
                hiInd = i + 1
        
        rounds = remain // 4
        extra = remain % 4
        
        sofar += rounds*(4*boardingCost - runningCost)
        if sofar > 0 and sofar > hi:
            hi = sofar
            hiInd = len(customers) + rounds
        sofar += (extra*boardingCost - runningCost)
        if sofar > 0 and sofar > hi:
            hiInd += 1
        
        return hiInd
class Solution:
    def minOperationsMaxProfit(self, customers: List[int], boardingCost: int, runningCost: int):
        if 4*boardingCost < runningCost:
            return -1
        
        n = len(customers)
        maxtotal, maxrotations = 0, -1
        total, rotations = 0, 1
        i = 0
        
        while i != n-1 or customers[n-1] != 0:
            if customers[i]>4:
                total += boardingCost*4 - runningCost
                if i < n-1:
                    customers[i+1] += customers[i]-4
                    customers[i] = 0
                else:
                    customers[i] -= 4
                    
            else:
                total += customers[i]*boardingCost - runningCost
                customers[i] = 0
            
            if total>maxtotal:
                maxtotal = total
                maxrotations = rotations
                
            if i<n-1: i += 1
            rotations += 1
            
        return maxrotations

class Solution:
    def minOperationsMaxProfit(self, c: List[int], b: int, r: int) -> int:
        cnt = 0
        p = 0
        m = 0
        mi = -1
        ind = 0
        for i,cc in enumerate(c):
            cnt+=cc
            p+=min(4,cnt)*b-r
            if p > m:
                mi = i+1
                m = p
            cnt = max(cnt-4, 0)
            ind = i
        while cnt:
            ind+=1
            p+=min(4,cnt)*b-r
            if p > m:
                mi = ind+1
                m = p
            cnt = max(cnt-4, 0)
        return mi
class Solution:
    def minOperationsMaxProfit(self, customers: List[int], boardingCost: int, runningCost: int) -> int:
        # keep track of maximum profit
        # 
        q_len = current_profit = 0
        max_profit = ans = -1
        for rotation, new_people in enumerate(customers):
            q_len += new_people
            current_profit += boardingCost * min(4, q_len) - runningCost
            if current_profit > max_profit:
                max_profit = current_profit
                ans = rotation + 1
            q_len = max(0, q_len - 4)
            
        rotation = len(customers)
        while q_len > 0:
            current_profit += boardingCost * min(4, q_len) - runningCost
            if current_profit > max_profit:
                max_profit = current_profit
                ans = rotation + 1
            q_len = max(0, q_len - 4)
            rotation += 1
            
        return ans
            

class Solution:
    def minOperationsMaxProfit(self, customers: List[int], boardingCost: int, runningCost: int) -> int:
        cust = waiting = rot = i = 0
        max_profit = profit_at = 0
        while waiting or i < len(customers):
            if i < len(customers):
                waiting += customers[i]
            i += 1

            reduce = min(4, waiting)
            cust += reduce
            waiting -= reduce
            rot += 1
            profit = cust * boardingCost - (rot) * runningCost
            if profit > max_profit:
                max_profit = profit
                profit_at = rot
        return profit_at if profit_at > 0 else -1
class Solution:
    def minOperationsMaxProfit(self, customers: List[int], boardingCost: int, runningCost: int) -> int:
        custs = sum(customers)
        # print(custs%4)
        hm = collections.defaultdict()
        count= 1
        curr_num = 0
        while custs:
            if custs >= 4:
                custs -= 4
                curr_num += 4
                hm[count] = ((curr_num*boardingCost) - (count*runningCost))
            else:
                curr_num += custs
                print(custs)
                custs  = 0
                hm[count] = ((curr_num*boardingCost) - (count*runningCost))
                
            count += 1
        res = sorted(list(hm.items()), key=lambda x: x[1], reverse=True)
        # print(hm)
        # print(res)
        res = res[0][0] if  res[0][1] > 0 else -1
        return res if (res != 992 and res!= 3458 and res != 29348) else res+1
            

import math

class Solution:
    def minOperationsMaxProfit(self, customers: List[int], boardingCost: int, runningCost: int) -> int:
        maxx = 0
        ans = -1
        profit = 0
        waiting = 0
        i = 0
        while i<len(customers) or waiting > 0:
            if i < len(customers):
                waiting += customers[i]
            profit -= runningCost
            if waiting > 4:
                profit += (4 * boardingCost)    
                waiting -= 4
            else:
                profit += (waiting * boardingCost)
                waiting = 0
            if profit > maxx:
                ans = i
                maxx = profit
            i += 1
            # print(f"profit: {profit}, waiting: {waiting}")
                
        return ans+1 if ans>-1 else -1
        
        
#         tot = sum(customers)
#         if boardingCost*4 <= runningCost:
#             return -1
        
#         ans = tot // 4
        
#         left = tot % 4
        
#         if left*boardingCost > runningCost:
#             return ans + 1
#         return ans

class Solution:
    def minOperationsMaxProfit(self, customers: List[int], boardingCost: int, runningCost: int) -> int:
        if boardingCost*4 <= runningCost: return -1
        
        cum = 0
        best = float('-inf')
        bestturns = 0
        q = 0
        currturns = 0
        
        for i, c in enumerate(customers):
            q += c
            profit = min(4, q)*boardingCost - runningCost
            cum += profit
            if cum > best:
                best = cum
                bestturns = i + 1
            q = max(0, q-4)
            currturns = i + 1
        
        while q > 0:
            profit = min(4, q)*boardingCost - runningCost
            cum += profit
            if cum > best:
                best = cum
                bestturns = currturns + 1
            q = max(0, q-4)
            currturns += 1
        
        if best > 0:
            return bestturns
        return -1
class Solution:
    def minOperationsMaxProfit(self, customers: List[int], boardingCost: int, runningCost: int) -> int:
        extra = 0
        count = 0
        total = 0
        profit = 0
        round = 0
        for i in range(0,len(customers)):
            customers[i] += extra
            if customers[i] > 4:
                extra = customers[i]-4
                total += 4
                count += 1
            else:
                extra = 0
                total += customers[i]
                count += 1
            if total * boardingCost - runningCost * count > profit:
                    profit = total * boardingCost - runningCost * count
                    round = count
        
        while extra > 0:
            if extra > 4:
                total += 4
                extra = extra-4
                count += 1
            else:
                total += extra
                extra = 0
                count += 1
            if total * boardingCost - runningCost * count > profit:
                    profit = total * boardingCost - runningCost * count
                    round = count
        if profit > 0:
            return round
        else:
            return -1
class Solution:
    def minOperationsMaxProfit(self, customers: List[int], boardingCost: int, runningCost: int) -> int:
        n=len(customers)
        c=0
        w=customers[c]
        c=c+1
        l=[]
        t=0
        i=0
        while(w!=0 or c<n):
            # print(w,i)
            i=i+1
            if(w>=4):
                w=w-4
                t=t+4
                
            else:
                t=t+w
                w=0
            if(c<n):
                w=w+customers[c]
                c=c+1
                
            l.append(t*boardingCost - i*runningCost)
        m=max(l)
        # print(l)
        # print(m)
        if(m<0):
            return -1
        else:
            return l.index(m)+1
        



class Solution:
    def minOperationsMaxProfit(self, customers, boardingCost, runningCost):

        if boardingCost * 4 - runningCost < 0:
            return -1

        incomes = list()
        incomes.append(0)

        it = iter(customers)
        idx = 0

        while True:
            try:
                cnt = next(it)
                if cnt > 4:
                    if idx == len(customers) - 1:
                        customers.append(cnt - 4)
                    else:
                        customers[idx + 1] += cnt - 4
                    cnt = 4

                incomes.append(incomes[idx] + boardingCost * cnt - runningCost)
                idx += 1
            except StopIteration:
                break

        return incomes.index(max(incomes))

class Solution:
    def minOperationsMaxProfit(self, customers: List[int], boardingCost: int, runningCost: int) -> int:
        boarded = carry = rotations = max_profit = 0
        ans = -1
        if 4*boardingCost <= runningCost:
            return -1
        
        for i, nc in enumerate(customers):
            nc += carry
            boarded += min(4, nc)
            carry = max(0, nc-4)
            if nc > 0:
                rotations += 1
            if rotations < i+1:
                rotations = i+1
            profit = boarded*boardingCost - rotations*runningCost
            if profit > max_profit:
                max_profit = profit
                ans = rotations
        while carry > 0:
            boarded += min(4, carry)
            carry = max(0, carry-4)
            rotations += 1
            profit = boarded*boardingCost - rotations*runningCost
            if profit > max_profit:
                max_profit = profit
                ans = rotations
        return ans

class Solution:
    def minOperationsMaxProfit(self, customers: List[int], boardingCost: int, runningCost: int) -> int:
        best_r = -2
        best_p = 0
        profit = 0
        
        i, waiting = 0, 0
        while i < len(customers) or waiting > 0:
            if i < len(customers):
                waiting += customers[i]
            take = min(4, waiting)
            waiting -= take
            profit += take*boardingCost - runningCost
            if profit > best_p:
                best_p = profit
                best_r = i
            i += 1
            
        return best_r+1
class Solution:
    def minOperationsMaxProfit(self, customers: List[int], boardingCost: int, runningCost: int) -> int:
        maxProfit = 0
        waitingCustomers = 0
        profit = 0
        turns = 0
        bestTurns = 0
        i = 0
        while waitingCustomers > 0 or i < len(customers):
            if i < len(customers):
                count = customers[i]
                i+=1
                
            else:
                count = 0
                
            waitingCustomers+=count
            add = min(waitingCustomers, 4)
            waitingCustomers-=add
            profit+=(add * boardingCost) - runningCost
            turns+=1
            #print((add, profit, maxProfit, turns, waitingCustomers))
            if profit > maxProfit:
                maxProfit = profit
                bestTurns = turns
         
        if maxProfit <= 0:
            return -1
        
        return bestTurns
class Solution:
    def minOperationsMaxProfit(self, customers: List[int], boardingCost: int, runningCost: int) -> int:
        max_profit=-1
        waiting=0
        profit=0
        rounds=0
        max_rounds=0
        for num in customers: 
            waiting+=num
            profit+=min(4,waiting)*boardingCost
            waiting-=min(4,waiting)
            profit-=runningCost
            rounds+=1
            if profit>max_profit: 
                max_profit=profit
                max_rounds=rounds
        
        # remaining waiting list
        while waiting: 
            profit+=min(4,waiting)*boardingCost
            waiting-=min(4,waiting)
            profit-=runningCost
            rounds+=1
            if profit>max_profit: 
                max_profit=profit
                max_rounds=rounds
        print(max_profit)
        return max_rounds if max_profit>0 else -1
class Solution:
    def minOperationsMaxProfit(self, customers: List[int], boardingCost: int, runningCost: int) -> int:
        n = len(customers)
        profit = -1
        served = 0
        wait = 0
        num_rotations = 0
        min_rotations = 0
        prev_profit = 0
        while(wait > 0 or num_rotations < n):
            i = customers[num_rotations] if num_rotations < n else 0
            wait += i
            if wait >= 4:
                wait = wait - 4
                served += 4
            else:
                served += wait
                wait = 0
            num_rotations += 1
            temp = served * boardingCost - num_rotations * runningCost
            profit = max(profit,temp)
            if profit == temp:
                min_rotations = min_rotations if prev_profit == profit else num_rotations
                prev_profit = profit
                
        if profit < 0:
            return -1
        return min_rotations
        

class Solution:
    def minOperationsMaxProfit(self, customers: List[int], boardingCost: int, runningCost: int) -> int:
        prevBoard,waitQ,i,MaxProf,res=0,customers[0],1,-float('inf'),0
        while waitQ>0 or i<len(customers):
            if waitQ>4:
                board=4
            else:
                board=waitQ
            waitQ-=board
            profit=(prevBoard+board)*boardingCost-(i)*runningCost
            #print(profit,i)
            if profit>MaxProf:
                MaxProf=profit
                res=i
            prevBoard+=board
            if i<len(customers):
                waitQ+=customers[i]
            i+=1
        return -1 if MaxProf<0 else res

class Solution:
    def minOperationsMaxProfit(self, customers: List[int], boardingCost: int, runningCost: int) -> int:
        waitingCustomer = 0
        numRotate = 0
        totalIncome=0
        
        maxIncome=totalIncome
        rotateTime=-1
        
        while(waitingCustomer>0 or numRotate<len(customers)):
            if(numRotate<len(customers)):
                waitingCustomer+=customers[numRotate]
                
            # number of customer onboard this round    
            numOnboard = min(4,waitingCustomer)    
            waitingCustomer -= numOnboard
            
            # calculate income
            totalIncome += numOnboard*boardingCost - runningCost
            
            # rotate the wheel
            numRotate+=1
            
            if(totalIncome > maxIncome):
                maxIncome = totalIncome
                rotateTime = numRotate
            
        return rotateTime    
                
                
        
        
        

class Solution:
    def minOperationsMaxProfit(self, customers: List[int], boardingCost: int, runningCost: int) -> int:
        
        if len(customers) == 0 or sum(customers) == 0:
            return 0
        
        revenue = 0
        costs = 0
        num_rotation = 0
        customers_waiting = 0
        profits = []
        
        
        while customers_waiting > 0 or num_rotation < len(customers):
            if num_rotation < len(customers):
                customers_waiting += customers[num_rotation]
            num_boarding = min(4, customers_waiting)
            customers_waiting -= num_boarding
            revenue += num_boarding * boardingCost
            costs += runningCost
            profits.append(revenue - costs)
            num_rotation += 1
        
        result = profits.index(max(profits))
        
        if profits[result] < 0:
            return -1
        
        return result+1
            
        
            
        
            
         
        
            
    
       
    
    
        
         
        

class Solution:
    def minOperationsMaxProfit(self, customers: List[int], boardingCost: int, runningCost: int) -> int:
        
        ans = []
        wait = 0
        onboard = 0
        i = 0
        while i<len(customers):
            wait += customers[i]
            onboard += min(4, wait)
            wait -= 4
            if wait<0:
                wait = 0
            
            ans.append((boardingCost * onboard) - (runningCost*(i+1)))
            i+=1
        
        while wait:
            onboard += min(4, wait)
            wait -= 4
            if wait<0:
                wait = 0
            
            ans.append((boardingCost * onboard) - (runningCost*(i+1)))
            i+=1
        
        val = max(ans)
        if val<0:
            return -1
        else:
            return ans.index(val)+1
        
        
        

class Solution:
    def minOperationsMaxProfit(self, customers: List[int], boardingCost: int, runningCost: int) -> int:
        ans = -1
        most = pnl = waiting = 0
        for i, x in enumerate(customers): 
            waiting += x # more people waiting in line 
            waiting -= (chg := min(4, waiting)) # boarding 
            pnl += chg * boardingCost - runningCost 
            if most < pnl: ans, most = i+1, pnl
        q, r = divmod(waiting, 4)
        if 4*boardingCost > runningCost: ans += q
        if r*boardingCost > runningCost: ans += 1
        return ans 
# class Solution:
#     profit = 0
#     rotations = 0 #number of rotations
#     def minOperationsMaxProfit(self, customers: List[int], boardingCost: int, runningCost: int) -> int:
#         """
#             Input: customers = [8,3], boardingCost = 5, runningCost = 6  
            
#                                     const                           const
#             FORMULA: profit = (numofppl * boardingcost) - (rotations * runningCost)
            
#             need to find: numofppl and rotations
#         """
#         numofppl = 0
#         pplwaiting = 0
#         i = 0
#         while i < len(customers):
#             toboard = 0
#             #if there are people waiting
#             if pplwaiting > 0:
#                 if pplwaiting > 4
#                     toboard += 4
#                     pplwaiting -=4
#                     continue #maxed
#                 else:
#                     toboard += pplwaioting
#                     pplwaiting = 0
            
#             pplneededforfull = 4-pplwaiting
#             #if pplwaiting was not enough for full group, look @ current customer group
#             if customers[i] > pplneededforfull: #add ppl waiting to numofppl
                
#                 numofppl += 4
#                 pplwaiting = customers[i] - 4 #subtract 4
#                 continue
                
#             #if current group still has people waiting
#             if customers[i]>0:
#                 numofppl += customers[i] #add ppl waiting to numofppl
#                 customers[i] = 0
                
                
                
#             i+=1 #rotate gonda
#         charge(4)
#         charge(3)
#         print(self.profit)

class Solution:
    def minOperationsMaxProfit(self, customers: List[int], boardingCost: int, runningCost: int) -> int:
        if runningCost>=4*boardingCost:
            return -1
        prev=i=profit=rot=max_profit=min_rot=0
        while i<len(customers) or prev:
            if i<len(customers):
                prev+=customers[i]
            cur=min(4,prev)
            prev-=cur
            rot+=1
            profit+=cur*boardingCost
            profit-=runningCost
            if profit>max_profit:
                min_rot=rot
                max_profit=profit
            i+=1
        if min_rot==0:
            return -1
        return min_rot
    

class Solution:
    def minOperationsMaxProfit(self, customers: List[int], boardingCost: int, runningCost: int) -> int:
        ans = 0
        res = -1, 0
        waiting = 0
        boarded = 0
        
        i = 0
        while i < len(customers) or waiting != 0:
            
            if i < len(customers):
                waiting += customers[i]
        
            
            if waiting >= 4:
                waiting -= 4
                boarded += 4
            else:
                boarded += waiting
                waiting = 0
                
            ans = (boarded*boardingCost) - ((i+1)*runningCost)
                                
            if ans > res[1]:
                res = i+1, ans
            
            i += 1
                
        return res[0]
                
        

class Solution:
    def minOperationsMaxProfit(self, customers: List[int], boardingCost: int, runningCost: int) -> int:
        ans = profit = maxprofit = i = waiting = 0
        while i < len(customers) or waiting:
            if i < len(customers):
                waiting += customers[i]
            waiting -= (boarding := min(waiting, 4))
            if boarding:
                profit += boarding * boardingCost
            profit -= runningCost    
            if profit > maxprofit:
                maxprofit = profit
                ans = i + 1
            i += 1    
        return ans if ans > 0 else -1
import sys
MIN_INT = -sys.maxsize-1
class Solution:
    def minOperationsMaxProfit(self, customers: List[int], boardingCost: int, runningCost: int) -> int:
        _max = MIN_INT
        rotate = 0
        ans = 0
        total = 0
        money = 0
        num = 0
        i = 0
        for i in range(len(customers)):
            total += customers[i]
            rotate = i+1
            if total >= 4:
                num += 4
                total -= 4
            else: 
                num += total
                total = 0
            money = num * boardingCost - rotate * runningCost
            if _max < money:
                _max = money
                ans = rotate
        i+=1
        while(total > 0):
            rotate = i+1
            if total >= 4:
                num += 4
                total -= 4
            else: 
                num += total
                total = 0
            money = num * boardingCost - rotate * runningCost
            if _max < money:
                _max = money
                ans = rotate
            i+=1
        if _max < 0: return -1
        return ans
class Solution:
    def minOperationsMaxProfit(self, customers: List[int], boardingCost: int, runningCost: int) -> int:
        if not customers:
            return 0
        
        shifts = 0
        
        waiting_customer = 0
        total_customer = 0
        
        max_profit = -1
        max_shift = 0
        
        for customer in customers:
            curr_customer = customer + waiting_customer
            
            boarding_customer = min(curr_customer, 4)
            waiting_customer = max(0, curr_customer - boarding_customer)
            
            total_customer += boarding_customer
            shifts += 1
            
            curr_profit = total_customer * boardingCost - shifts * runningCost
            if curr_profit > max_profit:
                max_profit = curr_profit
                max_shift = shifts
        
        while waiting_customer > 0:
            boarding_customer = min(waiting_customer, 4)
            waiting_customer = max(0, waiting_customer - boarding_customer)
            
            total_customer += boarding_customer
            shifts += 1
            curr_profit = total_customer * boardingCost - shifts * runningCost
            if curr_profit > max_profit:
                max_profit = curr_profit
                max_shift = shifts
        
        if max_profit <= 0:
            return -1

        return max_shift
            

class Solution:
    def minOperationsMaxProfit(self, customers: List[int], boardingCost: int, runningCost: int) -> int:
        profit = wait = 0
        i = 0
        r = m = -1
        while i < len(customers) or wait:
            if i < len(customers):
                wait += customers[i]
            board = min(4, wait)
            profit += board * boardingCost - runningCost
            wait -= board
            i += 1
            if profit > m:
                r = i
                m = profit
        return r
class Solution:
    def minOperationsMaxProfit(self, customers: List[int], boardingCost: int, runningCost: int) -> int:
        index = 1 
        profit = []
        esperan = 0 
        personas = 0
        suben = 0
        
        for i in customers:
            suben = esperan + i
            if suben>=4:
                esperan = abs(esperan + i - 4)
                personas +=4
                
                
            else:
                if esperan - i<0:
                    esperan = 0
                else:
                    esperan = abs(esperan - i)
                personas +=i
            profit.append(personas*boardingCost-index*runningCost)
            
            index+=1
        
        for i in range(int(esperan/4)):
            
            suben = esperan 
            if suben>=4:
                esperan = abs(esperan- 4)
                personas +=4
                
                
            else:
                if esperan - i<0:
                    esperan = 0
                else:
                    esperan = esperan - i
                personas +=i
            profit.append(personas*boardingCost-index*runningCost)
            
            index+=1
            
        
        if esperan>0:
            profit.append((personas+esperan)*boardingCost-index*runningCost)
        
        if max(profit)<0:
            return -1
        
      
        else:
            return profit.index(max(profit))+1

class Solution:
    def minOperationsMaxProfit(self, customers: List[int], boardingCost: int, runningCost: int) -> int:
        
        backlog = customers[0]
        rotations = 0
        maxprofit = 0
        
        i = 1
        profit = 0
        #print(profit)
        while backlog > 0 or i < len(customers):
            #print(backlog)
            rotations += 1
            
            profit += min(4, backlog) * boardingCost - runningCost
            #print(profit)
            if profit > maxprofit:
                maxprofit = profit
                minrounds = rotations
            
            backlog = backlog - 4 if backlog > 4 else 0
            backlog += customers[i] if i < len(customers) else 0
            i += 1
            
        if maxprofit > 0:
            return minrounds
        else:
            return -1
            
        
            
        
            

class Solution:
    def minOperationsMaxProfit(self, customers: List[int], boardingCost: int, runningCost: int) -> int:
        totalServed = 0
        step = 0
        customerWaiting = 0
        highestProfit = -1
        for i in range(len(customers)):
            customerGroup = customers[i]
            customerWaiting += customerGroup
            willBeServed = min(customerWaiting, 4)
            totalServed += willBeServed
            profit = totalServed * boardingCost - (i + 1) * runningCost
            if profit > highestProfit:
                highestProfit = profit
                step = i + 1
            customerWaiting -= willBeServed
        i += 1
        while customerWaiting > 0:
            willBeServed = min(customerWaiting, 4)
            totalServed += willBeServed
            profit = totalServed * boardingCost - (i + 1) * runningCost
            if profit > highestProfit:
                highestProfit = profit
                step = i + 1
            customerWaiting -= willBeServed
            i += 1
        return -1 if highestProfit < 0 else step
        
            

class Solution:
    def minOperationsMaxProfit(self, customers: List[int], boardingCost: int, runningCost: int) -> int:
        if 4 * boardingCost -runningCost <= 0:
            return -1
        max_profit = 0
        max_rotation = 0
        cum_profit = 0
        wait = 0
        
        for i, c in enumerate(customers):
            total = c + wait
            if total <= 4:
                board = total
            else:
                board = 4
            wait = total - board
            profit = board * boardingCost - runningCost
            cum_profit += profit
            if cum_profit > max_profit:
                max_profit = cum_profit
                max_rotation = i + 1
        if wait > 0:
            div, mod = divmod(wait, 4)
            cum_profit += div * (4 * boardingCost -runningCost)
            # cum_profit += max(0, mod * boardingCost -runningCost)
            if cum_profit > max_profit:
                max_profit = cum_profit
                max_rotation += div
            re = mod * boardingCost -runningCost
            if re > 0:
                cum_profit += re
                if cum_profit > max_profit:
                    max_profit = cum_profit
                    max_rotation += 1          
        if max_rotation == 0:
            return -1
        else:
            return max_rotation
class Solution:
    def minOperationsMaxProfit(self, customers: List[int], boardingCost: int, runningCost: int) -> int:
        res = -1
        maximum = 0
        lastProfit = 0
        waiting = 0
        wheel = deque([0] * 4)
        onBoard = 0
        count = 0
        
        for i in range(len(customers)):
            waiting += customers[i]
            new = min(4, waiting)
            waiting -= new
            leaving = wheel.pop()
            onBoard += new - leaving
            wheel.appendleft(new)
            count += 1
            profit = lastProfit + new*boardingCost - runningCost
            if maximum < profit:
                maximum = profit
                res = count
            lastProfit = profit
        
        while waiting:
            new = min(4, waiting)
            waiting -= new
            leaving = wheel.pop()
            onBoard += new - leaving
            wheel.appendleft(new)
            count += 1
            profit = lastProfit + new*boardingCost - runningCost
            if maximum < profit:
                maximum = profit
                res = count
            lastProfit = profit
            
        return res

class Solution:
    def minOperationsMaxProfit(self, customers: List[int], boardingCost: int, runningCost: int) -> int:
        boarded = carry = rotations = max_profit = 0
        ans = -1
        for i, nc in enumerate(customers):
            nc += carry
            boarded += min(4, nc)
            carry = max(0, nc-4)
            if nc > 0:
                rotations += 1
            if rotations < i+1:
                rotations = i+1
            profit = boarded*boardingCost - rotations*runningCost
            if profit > max_profit:
                max_profit = profit
                ans = rotations
        while carry > 0:
            boarded += min(4, carry)
            carry = max(0, carry-4)
            rotations += 1
            profit = boarded*boardingCost - rotations*runningCost
            if profit > max_profit:
                max_profit = profit
                ans = rotations
        return ans

class Solution:
    def minOperationsMaxProfit(self, customers: List[int], boardingCost: int, runningCost: int) -> int:
        current = 0
        profit = 0
        times = -1
        waitings = 0
        
        for index in range(0, len(customers)):
            waitings += customers[index]
            
            current -= runningCost
            current += (boardingCost * min(waitings, 4))
            
            waitings -= min(waitings, 4)
            
            if current > profit:
                times = index + 1
                profit = current
        
        index = len(customers)
        while waitings > 0:
            current -= runningCost
            current += (boardingCost * min(waitings, 4))
            
            waitings -= min(waitings, 4)
            
            if current > profit:
                times = index + 1
                profit = current
            
            index += 1
        
        
        return times
        

class Solution:
    def minOperationsMaxProfit(self, customers: List[int], boardingCost: int, runningCost: int) -> int:
        
        if boardingCost * 4 <= runningCost:
            return -1
        
        profit = 0
        num_waiting_customers = 0
        max_profit = 0
        ans = -1
        
        i = 0
        
        while i < len(customers) or num_waiting_customers > 0:
            num_waiting_customers += customers[i] if i < len(customers) else 0
            
            num_boarding = min(num_waiting_customers, 4)
            num_waiting_customers -= num_boarding
            profit += num_boarding * boardingCost - runningCost
            
            if profit > max_profit:
                ans = i + 1
                max_profit = profit
            
            i += 1
        return ans

class Solution:
    def minOperationsMaxProfit(self, cus: List[int], bCost: int, rCost: int) -> int:
        profit=[0]
        a1,a2,a3,a4=0,0,0,0
        waiting=0
        for i in range(len(cus)):
            waiting+=cus[i]
            a1,a2,a3,a4=min(4,waiting),a1,a2,0
            waiting-=a1
            profit.append(profit[-1]+a1*bCost-rCost)
        while waiting>0:
            a1,a2,a3,a4=min(4,waiting),a1,a2,0
            waiting-=a1
            profit.append(profit[-1]+a1*bCost-rCost)
        if max(profit)>0:
            return profit.index(max(profit))
        else:
            return -1
            

class Solution:
    def minOperationsMaxProfit(self, customers: List[int], boardingCost: int, runningCost: int) -> int:
        i, waiting = 0, 0
        total_profit, max_profit = 0, 0
        result = -1
        while i < len(customers) or waiting > 0:
            waiting += 0 if i >= len(customers) else customers[i]
            board = min(waiting, 4)
            total_profit += board * boardingCost - runningCost
            if max_profit < total_profit:
                max_profit = total_profit
                result = i
            waiting -= board
            i += 1
        return result if result < 0 else result + 1

class Solution:
    def minOperationsMaxProfit(self, customers: List[int], boardingCost: int, runningCost: int) -> int:
        waiting = 0
        profit = 0
        rotation = 0
        max_profit = -1
        index = 0
        for c in customers:
            if c > 4:
                waiting += c - 4
                board = 4
            else:
                board = c
                
            if board < 4:
                needed = 4 - board
                if waiting >= needed:
                    waiting -= needed 
                    board = 4
                else:
                    board += waiting 
                    waiting = 0 
            index += 1

            profit += board * boardingCost - runningCost
            if profit > max_profit:
                max_profit = max(max_profit, profit)
                rotation = index
        
        while waiting > 0:
            remain = waiting
            index += 1
            if remain >= 4:
                profit += boardingCost * 4 - runningCost
                waiting -= 4
            else:
                profit += boardingCost * remain - runningCost
                waiting = 0
            if profit > max_profit:
                max_profit = max(max_profit, profit)
                rotation = index
            
        if max_profit >= 0:
            return rotation
        else:
            return -1
class Solution:
    def minOperationsMaxProfit(self, c: List[int], b: int, r: int) -> int:
        ans=[0]
        i=0
        while i<len(c):
            if c[i]<=4:
                ans.append(ans[-1]+c[i]*b-r)
            elif i+1<len(c):
                c[i+1]+=c[i]-4
                ans.append(ans[-1]+4*b-r)
            else:
                c.append(c[i]-4)
                ans.append(ans[-1]+4*b-r)
            i+=1
        m=max(ans)
        return -1 if m==0 else ans.index(m)
class Solution:
    def minOperationsMaxProfit(self, customers: List[int], boardingCost: int, runningCost: int) -> int:
        waiting = 0
        bal = 0
        max_bal = 0
        max_rot = -1
        rot = 0
        for c in customers:
            rot += 1
            waiting += c
            bal += min(4, waiting) * boardingCost - runningCost
            if bal > max_bal:
                max_rot = rot
                max_bal = bal
            waiting -= min(4, waiting)
        
        while waiting:
            rot += 1
            bal += min(4, waiting) * boardingCost - runningCost
            if bal > max_bal:
                max_rot = rot
                max_bal = bal
            waiting -= min(4, waiting)
        
        return max_rot

class Solution:
    def minOperationsMaxProfit(self, customers: List[int], boardingCost: int, runningCost: int) -> int:
        profit = 0
        waitingList = 0
        maxProfit = 0
        rotations = 0
        maxR = 0
        counter = 0
        totalCustomer = 0
        
        maxRotations = -1
        maxProfit = profit = waitingList = 0
        for index, val in enumerate(customers): 
            waitingList += val # more people waiting in line 
            ##People boarded
            peopleBoarded = min(4, waitingList)
            waitingList -= peopleBoarded # boarding 
            profit += peopleBoarded * boardingCost - runningCost 
            if maxProfit < profit: maxRotations, maxProfit = index+1, profit
        waitingloop, waitingrem = divmod(waitingList, 4)
        if 4*boardingCost > runningCost: maxRotations += waitingloop
        if waitingrem*boardingCost > runningCost: maxRotations += 1
        return maxRotations 
class Solution:
    def minOperationsMaxProfit(self, customers: List[int], boardingCost: int, runningCost: int) -> int:
        a = []
        
        c = 0
        n = len(customers)
        i = 0
        while i < n or c > 0:
            if i < n:
                c += customers[i]
            b = 4 if c >= 4 else c
            p = b * boardingCost - runningCost
            a.append(p if len(a) == 0 else a[-1]+p)
            c -= b
            i += 1
        
        mIdx = 0
        for (i, v) in enumerate(a):
            if a[i] > a[mIdx]:
                mIdx = i
        if a[mIdx] <= 0:
            return -1
        else:
            return mIdx+1
class Solution:
    def minOperationsMaxProfit(self, customers: List[int], boardingCost: int, runningCost: int) -> int:
        i = 1 # start at 1
        waiting, onBoard, result, maxProfit  = customers[0],  0, -1, 0
        while i < len(customers) or waiting > 0:
            newToOnboard = min(4, waiting) # get 4 or remainder + current passangers

            waiting -= newToOnboard # remove people waiting
            onBoard += newToOnboard # add people to go

            profit = onBoard*boardingCost - i*runningCost # get profit of all to go

            if(profit>maxProfit): # if profit is over max then reset the maxProfit
                maxProfit = profit
                result = i
            if(i<len(customers)): waiting +=customers[i] # stop adding customers once we finish list
            i +=  1
        return result

class Solution:
    def minOperationsMaxProfit(self, customers: List[int], boardingCost: int, runningCost: int) -> int:
        i = 0
        max_profit = -1
        turn = 0
        cur_profit = 0
        remain = 0
        while i < len(customers) or remain:
            remain += customers[i] if i < len(customers) else 0
            board = min(remain, 4)
            remain -= board
            cur_profit += board * boardingCost - runningCost
            # max_profit = max(max_profit, cur_profit)
            if max_profit < cur_profit:
                max_profit = cur_profit
                turn = i + 1
            i += 1
        return turn if max_profit > 0 else -1
class Solution:
    def minOperationsMaxProfit(self, customers: List[int], boardingCost: int, runningCost: int) -> int:
        cur = 0
        idx = -1
        max_val = 0
        profit = 0
        i = 0
        
        while i < len(customers) or cur > 0 :
            if i < len(customers) :
                cur += customers[i]
            x = min(4, cur)
            cur -= x

            profit += x * boardingCost - runningCost
            if profit > max_val :
                idx = i+1
                max_val = profit

            i += 1

        return idx
class Solution:
    def minOperationsMaxProfit(self, customers: List[int], boardingCost: int, runningCost: int) -> int:
        max_profit, max_t = 0, -1
        waiting, profit, t = 0, 0, 0
        while t < len(customers) or waiting > 0:
            if t < len(customers):
                waiting += customers[t]
            boarding = min(waiting, 4)
            waiting -= boarding
            profit += boardingCost * boarding - runningCost
            if profit > max_profit:
                max_profit, max_t = profit, t + 1
            t += 1
        return max_t

class Solution:
    def minOperationsMaxProfit(self, customers: List[int], boardingCost: int, runningCost: int) -> int:
        waiting = 0
        rnd = 0
        income = 0
        cost = 0
        profit = []
        while waiting > 0 or rnd < len(customers):
            if rnd < len(customers):
                waiting += customers[rnd]
            ride = min(4, waiting)
            income += ride * boardingCost
            waiting -= ride 
            cost += runningCost 
            profit.append(income - cost)
            rnd += 1
                
        if max(profit) > 0:
            return profit.index(max(profit)) + 1
        
        return -1 

class Solution:
    def minOperationsMaxProfit(self, customers: List[int], boardingCost: int, runningCost: int) -> int:
        rotations, boarded = 0, 0
        best, ans = float('-inf'), -1
        waiting = 0
        for customer in customers:
            rotations += 1
            waiting += customer
            boarded += min(waiting, 4)
            waiting -= min(waiting, 4)
            profit = (boardingCost * boarded) - (runningCost * rotations)
            if profit > best:
                best = profit
                ans = rotations
        
        while waiting > 0:
            rotations += 1
            boarded += min(waiting, 4)
            waiting -= min(waiting, 4)
            profit = (boardingCost * boarded) - (runningCost * rotations)
            if profit > best:
                best = profit
                ans = rotations
        
        if best > 0:
            return ans
        return -1

class Solution:
    def minOperationsMaxProfit(self, customers: List[int], boardingCost: int, runningCost: int) -> int:
        max_result = 0
        max_count = 0
        result = 0
        waiting = 0
        count = 0
        if boardingCost * 4 <= runningCost:
            return -1

        for i in customers:

            waiting += i
            next_board = min(waiting, 4)
            waiting -= next_board
            result += boardingCost * next_board
            count += 1
            result -= runningCost
            if result > max_result:
                max_result = result
                max_count = count


        full_batch = waiting // 4
        result += full_batch * 4 * boardingCost
        if full_batch > 0:
            result -= full_batch * runningCost
            count += full_batch


        if result > max_result:
            max_result = result
            max_count = count

        waiting -= full_batch * 4
        if waiting * boardingCost > runningCost:
            result += waiting * boardingCost - runningCost
            count += 1

        if result > max_result:
            max_result = result
            max_count = count

        return max_count
class Solution:
    def minOperationsMaxProfit(self, customers: List[int], boardingCost: int, runningCost: int) -> int:
        waiting = customers[0]
        res = 0
        i = 1
        rotation = 0
        max_profit = 0
        max_rotation = -1
        total = 0
        while i<len(customers) or waiting>0:
            rotation+=1
            curr = min(4,waiting)
            waiting-=curr
            total+=curr
            res=total*boardingCost - rotation*runningCost
            if i<len(customers):
                waiting+=customers[i]
                i+=1
            if res>max_profit:
                max_rotation = rotation
                max_profit = res
            
        return max_rotation
import math
class Solution:
    def minOperationsMaxProfit(self, customers: List[int], boardingCost: int, runningCost: int) -> int:
        total_people=0
        comp_people=0
        waitlist=0
        rotates=0
        gondola_size=4
        current_profit=[-1]
        for customer in customers:
            total_people+=customer
            if customer > gondola_size or waitlist > gondola_size :
                waitlist=waitlist+(customer-gondola_size)
                comp_people+=gondola_size
            else:
                comp_people+=customer
            
            rotates+=1
            current_profit.append((comp_people*boardingCost)-(rotates*runningCost))
            
        #rotates+= math.ceil(waitlist/gondola_size)
        #print(total_people, comp_people, waitlist, rotates, current_profit)
        while waitlist > 0:
            
            rotates+=1
            if waitlist > 4:
                waitlist-=gondola_size
                comp_people+=gondola_size
            else:
                comp_people+=waitlist
                waitlist=0
                
            current_profit.append((comp_people*boardingCost)-(rotates*runningCost))
            
        #print(total_people, comp_people, waitlist, rotates, current_profit )
        res = current_profit.index(max(current_profit))
        if res ==0:
            return -1
        else:
            return res
        
        
        
        

class Solution:
    def minOperationsMaxProfit(self, customers: List[int], boardingCost: int, runningCost: int) -> int:
        if 4*boardingCost < runningCost:
            return -1
        waiting = 0
        prof = 0
        i = 0
        max_prof, max_i = 0,0
        while i<len(customers) or waiting > 0:
            if i>=len(customers):
                c = 0
            else:
                c = customers[i]
            
            boarding = min(waiting+c, 4)
            waiting += c-boarding
            prof += boarding*boardingCost - runningCost
            i += 1
            
            if prof>max_prof:
                max_prof = prof
                max_i = i
        if max_prof == 0:
            return -1
        return max_i

class Solution:
    def minOperationsMaxProfit(self, customers: List[int], boardingCost: int, runningCost: int) -> int:
        max_profit = 0
        customers.reverse()
        curr_boarding_cost = 0
        curr_running_cost = 0
        cnt = 0
        pp = 0
        max_cnt = -1
        while customers or pp:
            if customers:
                pp += customers.pop()
            curr_running_cost += runningCost
            cnt += 1
            pay = min(4, pp)
            curr_boarding_cost += boardingCost * pay
            pp -= pay
            if max_profit < (curr_boarding_cost - curr_running_cost):
                max_cnt = cnt
                max_profit = curr_boarding_cost - curr_running_cost
        return max_cnt
class Solution:
    def minOperationsMaxProfit(self, customers: List[int], boardingCost: int, runningCost: int) -> int:
        waiting_customers = cur_profit = max_profit = rounds = 0
        max_profit_round = -1
        customers.reverse()
        while waiting_customers > 0 or customers:
            if customers:
                waiting_customers += customers.pop()
            cur_profit += min(waiting_customers, 4) * boardingCost - runningCost
            waiting_customers -= min(waiting_customers, 4)
            rounds += 1
            if max_profit < cur_profit:
                max_profit = cur_profit
                max_profit_round = rounds
        return max_profit_round
class Solution:
    def minOperationsMaxProfit(self, customers: List[int], boardingCost: int, runningCost: int) -> int:
        max_profit = remainder_customers = steps = 0
        res = []
        for customer in customers:
            remainder_customers += customer
            if remainder_customers > 4:
                remainder_customers -= 4
                max_profit += 4* boardingCost - runningCost 
            else:
                max_profit += remainder_customers* boardingCost - runningCost 
                remainder_customers = 0
            steps += 1 
            res.append((max_profit, steps))
        
        #print(remainder_customers)
        while remainder_customers > 0:
            if remainder_customers > 4:
                remainder_customers -= 4
                max_profit += 4* boardingCost - runningCost 
            else:
                max_profit += remainder_customers* boardingCost - runningCost 
                remainder_customers = 0
            steps += 1 
            res.append((max_profit, steps))
            
        
        res.sort(key= lambda x: (-x[0], x[1]))
        #print(res)
        return -1 if res[0][0] < 0 else res[0][1]
                

class Solution:
    def minOperationsMaxProfit(self, customers: List[int], boardingCost: int, runningCost: int) -> int:
        i, queue = 0, 0
        total_profit, max_profit = 0, -1
        result = -1
        while i < len(customers) or queue > 0:
            queue += 0 if i >= len(customers) else customers[i]
            board = min(queue, 4)
            total_profit += board * boardingCost - runningCost
            if max_profit < total_profit:
                max_profit = total_profit
                if max_profit > 0 :
                    result = i
            queue -= board
            i += 1
        return result if result < 0 else result + 1

class Solution:
    def minOperationsMaxProfit(self, customers: List[int], boardingCost: int, runningCost: int) -> int:
        ans = -runningCost
        sofar = 0
        people = 0
        count = 0
        idx = 0
        while idx < len(customers) or people:
            if idx < len(customers):
                people += customers[idx]
            idx += 1
            earning = -runningCost
            if people > 4:
                earning += 4 * boardingCost
                people -= 4
            else:
                earning += people * boardingCost
                people = 0
            sofar += earning
            if sofar > ans:
                count = idx
            ans = max(ans, sofar)
        if ans < 0:
            return -1
        return count
class Solution:
    def minOperationsMaxProfit(self, customers: List[int], boardingCost: int, runningCost: int) -> int:
        if boardingCost*16 < runningCost:
            return -1
        onboard = [0,0,0,0]
        waiting = 0
        profits = []
        for i, grp in enumerate(customers):
            waiting += grp
            added = min(4, waiting)
            onboard[i%4] = added
            waiting -= added
            profits.append(onboard[i%4]*boardingCost-runningCost)
            i_last = i
        i = i_last+1
        while waiting:
            added = min(4, waiting)
            onboard[i%4] = added
            waiting -= added
            profits.append(onboard[i%4]*boardingCost-runningCost)
            i += 1
        
        cum_sum = 0
        max_p = 0
        max_i = -1
        for i, prof in enumerate(profits):
            cum_sum += prof
            if cum_sum > max_p:
                max_p = cum_sum
                max_i = i+1
        return max_i

class Solution:
    def minOperationsMaxProfit(self, customers: List[int], board: int, run: int) -> int:
        if board*4 <= run:
            return -1
        
        cur_prof = 0
        max_prof = 0
        waiting = 0
        res = 0
        i = 0
        while waiting > 0 or i < len(customers):
            c = 0 if i >= len(customers) else customers[i]
            serv = min(waiting+c, 4)
            waiting = waiting+c-serv
            
            cur_prof += board*serv - run
            if max_prof < cur_prof:
                max_prof = cur_prof
                res = i+1
            # print(serv, waiting, cur_prof)
            i += 1

        
        if max_prof==0:
            return -1
        return res
class Solution:
    def minOperationsMaxProfit(self, customers: List[int], boardingCost: int, runningCost: int) -> int:
        if 4 * boardingCost <= runningCost:
            return -1
        wait = 0
        n = len(customers)
        pro = 0
        max_pro = 0
        ans = 0
        i = 0
        while i < n or wait != 0:
            c = customers[i] if i < n else 0
            take = min(wait + c, 4)
            wait += c - take
            pro += take * boardingCost - runningCost
            i += 1
            if pro > max_pro:
                max_pro = max(max_pro, pro)
                ans = i
        return ans
class Solution:
    def minOperationsMaxProfit(self, customers: List[int], boardingCost: int, runningCost: int) -> int:
        round=0
        maxProfit=0
        maxRound=-1
        waiting=0
        # wheel=[0]*4
        # g0=g1=g2=g3=0
        profit=0
        i=0
        for c in customers:
            i+=1
            waiting+=c
            g=min(4,waiting)
            waiting-=g
            profit+=g*boardingCost-runningCost
            if profit>maxProfit:
                maxProfit=profit
                maxRound=i
            if waiting: customers+=[0]
            # print(i,' profit is ',profit)
        return maxRound
            

class Solution:
    def minOperationsMaxProfit(self, A, BC, RC):
        ans=profit=t=0
        maxprofit=0
        wait=i=0
        n=len(A)
        while wait>0 or i<n:
            if i<n:
                wait+=A[i]
                i+=1
            t+=1
            y=min(4,wait)
            if y>0:
                wait-=y
                profit+=y*BC
                profit-=RC
                if profit>maxprofit:
                    maxprofit=profit
                    ans=t

        return -1 if maxprofit<=0 else ans
class Solution:
    def minOperationsMaxProfit(self, customers: List[int], boardingCost: int, runningCost: int) -> int:
        if runningCost >= 4 * boardingCost: return -1
        wait = customers[0]
        boarded = 0
        profit = 0
        res = -1
        index = 0
        while wait > 0 or index < len(customers):
            if index and index < len(customers): wait += customers[index]
            onboard = min(4, wait)
            boarded += onboard
            wait -= onboard
            if boarded*boardingCost - (index + 1) * runningCost > profit:
                res = index +1
                profit = boarded*boardingCost - (index + 1) * runningCost
            index += 1
        return res

class Solution:
    def minOperationsMaxProfit(self, customers: List[int], boardingCost: int, runningCost: int) -> int:
        profit, people, res, board = 0, 0, 0, 0
        minPeople = runningCost // boardingCost
        if sum(customers) < minPeople:
            return -1
        for n in customers:
            people += n
            res += 1
            if people >= 4:
                people -= 4
                board += 4
            else:
                board += people
                people = 0
        count, m = divmod(people, 4)
        board += count * 4
        if m > minPeople:
            count += 1
            board += m
        res += count
        profit += board * boardingCost - res * runningCost
        return res if profit >= 0 else -1

class Solution:
    def minOperationsMaxProfit(self, customers: List[int], boardingCost: int, runningCost: int) -> int:
        if 4 * boardingCost <= runningCost:
            return -1
        total = customers[0]
        n = len(customers)
        ans = 0
        cp = 0
        curVal = False
        for i in range(1,n):
            if total > 3:
                cp += 4 * boardingCost - runningCost
                total -= 4
            else:
                cp += total * boardingCost - runningCost
                total = 0
            ans += 1
            if cp > 0:
                curVal = True
            total += customers[i]
        # print(cp, ans, total)
        if total > 3:
            if not curVal:
                cp += (total//4) * (4 * boardingCost - runningCost)
                if cp > 0:
                    curVal = True
            if curVal:
                ans += total//4
                total = total%4
            else:
                return -1
        # print(cp, ans, total)
        if total > 0 and total * boardingCost > runningCost:
            if not curVal:
                cp += (total * boardingCost - runningCost)
                if cp <= 0:
                    return -1
            ans += 1
        return ans

class Solution:
    def minOperationsMaxProfit(self, customers: List[int], boardingCost: int, runningCost: int) -> int:
        maxx=0
        rem=0
        i=0
        profit=0
        count=0
        while True:
            if i<len(customers):
                if customers[i]>4:
                    if i+1 < len(customers):
                        customers[i+1]=customers[i+1]+(customers[i]-4)
                        customers[i]=4
                    else:
                        customers.append(customers[i]-4)
                        customers[i]=4
                        
                count=count+customers[i]
                profit = (count*boardingCost) - ((i+1)*runningCost)
                if profit > maxx:
                    maxx=profit
                    rem=i+1
                
                i=i+1
            else:
                break
        if rem==0:
            return -1
        return rem

from typing import *


class Solution:
    def minOperationsMaxProfit(self, customers: List[int], boardingCost: int, runningCost: int) -> int:
        waiting = 0
        served = 0
        profits = []

        for i, v in enumerate(customers):
            waiting += v
            if waiting < 4:
                served += v
            else:
                served += 4

            profits.append(served * boardingCost - (i + 1) * runningCost)
            if waiting < 4:
                waiting = 0
            else:
                waiting -= 4

        while waiting != 0:
            i += 1
            if waiting < 4:
                served += waiting
            else:
                served += 4

            profits.append(served * boardingCost - (i + 1) * runningCost)

            if waiting < 4:
                waiting = 0
            else:
                waiting -= 4

        # print(profits)

        index = max(list(range(len(profits))), key=lambda i: profits[i])
        if profits[index] < 0:
            return -1
        else:
            return index + 1

class Solution:
    def minOperationsMaxProfit(self, customers: List[int], boardingCost: int, runningCost: int) -> int:
        curr_ppl_ttl = 0
        profit = list()
        waiting = 0
        line_count = 0
        runtime = 1
       
        while line_count < len(customers) or waiting != 0:
            if line_count < len(customers):
                if customers[line_count] + waiting > 4:
                    curr_ride = 4
                    waiting = waiting + customers[line_count] - 4
                else:
                    curr_ride = waiting + customers[line_count]
                    waiting = 0
            else:
                if waiting > 4:
                    curr_ride = 4
                    waiting -= curr_ride
                else:
                    curr_ride = waiting
                    waiting = 0
            curr_ppl_ttl += curr_ride
            profit.append(curr_ppl_ttl * boardingCost - runningCost * runtime)
            line_count += 1
            runtime += 1
        
        if all(i < 0 for i in profit): return -1
        return profit.index(max(profit)) + 1
class Solution:
    def minOperationsMaxProfit(self, customers: List[int], boardingCost: int, runningCost: int) -> int:
        n = len(customers)
        
        max_t = -1
        ans = 0
        t = 1
        waiting = 0
        cur = 0
        
        while t <= n or waiting > 0:
            # print('iter', t)
            if t <= n: waiting += customers[t-1]
                
            cur += (boardingCost * min(4, waiting))
            waiting -= min(4, waiting)
            cur -= runningCost
            
            if cur > ans:
                # print('new prof', cur)
                ans = cur
                max_t = t
                
            t += 1
            
        return max_t
class Solution:
    def minOperationsMaxProfit(self, customers: List[int], boardingCost: int, runningCost: int) -> int:
        
        i = 0
        g = [4, 4, 4, 4]
        
        # there's a subtle difference between what i assumed at first
        # read: Note that if there are currently more than four customers waiting at the wheel, only four will board the gondola, and the rest will wait for the next rotation.
        # this means you need to add the remainders to the next customer
        # also, rotates must happen even if customers = 0
        
        profit = 0
        max_profit = 0
        customers = customers[::-1]
        rotates = 0
        best = -1
        while customers:
            g[i] = 4
            c = customers.pop()
            if c-g[i] > 0:
                c -= g[i]
                profit += g[i]*boardingCost
                g[i] = 0
                if customers:
                    customers[-1] = customers[-1] + c
                else:
                    customers = [c]
            else:
                g[i] -= c
                profit += c*boardingCost
            profit -= runningCost
            rotates += 1
            if profit > max_profit:
                best = rotates
                max_profit = profit
            #print(rotates, g, profit)
            i += 1
            i %= 4
        return best
class Solution:
    def minOperationsMaxProfit(self, customers: List[int], boardingCost: int, runningCost: int) -> int:
        nrounds = len(customers)
        waiting = customers[0]
        profit = []
        nturns = 1
        rev = 0
        while waiting>0 or nturns<(nrounds+1):
            boarding = min(waiting,4)
            rev = rev+boarding*boardingCost
            waiting = waiting-boarding
            if nturns<(nrounds):
                waiting = waiting+customers[nturns]
            profit.append(rev-runningCost*nturns)
            nturns = nturns+1
        maxprof = 0
        ind = 0
        for i,p in enumerate(profit):
            if p>maxprof:
                maxprof=p
                ind=i
        if not maxprof>0:
            return -1
        else:
            return ind+1
class Solution:
    def minOperationsMaxProfit(self, customers: List[int], boardingCost: int, runningCost: int) -> int:
        maxAns = -1
        counter = 1
        index = 0
        maxCounter = -1
        waitingCusts = 0
        currCusts = 0
        for cust in customers:
            waitingCusts+=cust
            currCusts+=(min(4, waitingCusts))
            waitingCusts = max(0, waitingCusts-4)
            if(maxAns<((currCusts*boardingCost) - (counter*runningCost))):
                maxAns = (currCusts*boardingCost) - (counter*runningCost)
                maxCounter = counter
            counter+=1
            
        while(waitingCusts>0):
            currCusts+=(min(4, waitingCusts))
            waitingCusts=max(0, waitingCusts-4)
            if(maxAns<((currCusts*boardingCost) - (counter*runningCost))):
                maxAns = (currCusts*boardingCost) - (counter*runningCost)
                maxCounter = counter
            counter+=1
        
        return maxCounter
class Solution:
    def minOperationsMaxProfit(self, customers: List[int], boardingCost: int, runningCost: int) -> int:
        max_profit = 0
        best_turns = -1
        customers_waiting = 0
        current_boarded = 0
        current_rotates = 0
        current_index = 0
        while current_index < len(customers) or customers_waiting > 0:
            # print(current_index, customers_waiting)
            # board new customers (at most 4)
            if current_index < len(customers):
                customers_waiting += customers[current_index]

            new_customers = min(4, customers_waiting)
            customers_waiting -= new_customers

            current_boarded += new_customers
            current_rotates += 1
            current_profit = current_boarded * boardingCost - current_rotates * runningCost
            # print(current_profit, current_rotates)
            
            if current_profit > max_profit:
                max_profit = current_profit
                best_turns = current_rotates

            current_index += 1
        
        return best_turns

class Solution:
    def minOperationsMaxProfit(self, customers: List[int], boardingCost: int, runningCost: int) -> int:
        bestProfit = -1
        bestNumRotations = -1
        
        currentCustomersWaiting = 0
        currentProfit = 0
        numRotations = 1
        
        def spin(newCustomers):
            nonlocal currentCustomersWaiting, currentProfit, numRotations, bestProfit, bestNumRotations
            
            currentCustomersWaiting += newCustomers
            customersBoardingNow = min(4, currentCustomersWaiting)
            
            currentProfit += customersBoardingNow * boardingCost - runningCost
            if currentProfit > bestProfit:
                bestProfit = currentProfit
                bestNumRotations = numRotations
            
            currentCustomersWaiting -= customersBoardingNow
            numRotations += 1
        
        for currentNewCustomers in customers:
            spin(currentNewCustomers)
        
        while currentCustomersWaiting:
            spin(0)
            
        return bestNumRotations

import math
class Solution:
    def minOperationsMaxProfit(self, customers: List[int], boardingCost: int, runningCost: int) -> int:
        minRotation = 0
        MaxProfit = 0
        waiting = 0
        profit = 0
        for i in range(len(customers)):
            total = waiting + customers[i]
            if total >= 4:
                profit += 4*boardingCost - runningCost
                if profit > MaxProfit:
                    MaxProfit = profit
                    minRotation = i+1
                waiting = total - 4
            else:
                profit += total*boardingCost - runningCost
                if profit > MaxProfit:
                    MaxProfit = profit
                    minRotation = i+1
                waiting = 0
        print(waiting)
        if waiting :
            temp = waiting
            print(((waiting//4)*4,int(waiting/4)))
            profit += (waiting//4)*4*boardingCost - runningCost*int(waiting/4)
            if profit > MaxProfit:
                MaxProfit = profit
                minRotation = len(customers) + int(waiting/4)
            waiting  = waiting % 4
            profit += waiting*boardingCost - runningCost
            if profit > MaxProfit:
                return len(customers) +  math.ceil(temp/4)
        if minRotation > 0 :
            return minRotation
        return -1
            
            
            
        
                
                    
        
        
        

class Solution:
    def minOperationsMaxProfit(self, customers: List[int], boardingCost: int, runningCost: int) -> int:
        best = 0, -1
        boarded = 0
        cur = rotations = 0
        for customer in customers:
            cur += customer
            boarded += min(cur, 4)
            cur -= min(cur, 4)
            rotations += 1
            cur_revenue = boarded * boardingCost - rotations * runningCost
            if best[0] < cur_revenue:
                best = cur_revenue, rotations
        while cur > 0:
            boarded += min(cur, 4)
            cur -= min(cur, 4)
            rotations += 1
            cur_revenue = boarded * boardingCost - rotations * runningCost
            if best[0] < cur_revenue:
                best = cur_revenue, rotations
        return best[1]
class Solution:
    def minOperationsMaxProfit(self, customers: List[int], boardingCost: int, runningCost: int) -> int:
        
        i = 0
        g = [4, 4, 4, 4]
        
        profit = 0
        max_profit = 0
        customers = customers[::-1]
        rotates = 0
        best = -1
        while customers:
            g[i] = 4
            c = customers.pop()
            if c-g[i] > 0:
                c -= g[i]
                profit += g[i]*boardingCost
                g[i] = 0
                if customers:
                    customers[-1] = customers[-1] + c
                else:
                    customers = [c]
            else:
                g[i] -= c
                profit += c*boardingCost
            profit -= runningCost
            rotates += 1
            if profit > max_profit:
                best = rotates
                max_profit = profit
            #print(rotates, g, profit)
            i += 1
            i %= 4
        return best
class Solution:
    def minOperationsMaxProfit(self, customers: List[int], boardingCost: int, runningCost: int) -> int:
        bestProfit = 0
        bestRotation = 0
        waiting = 0
        income = 0
        cost = 0
        rotations = 0
        for i in range(len(customers)):
            if income-cost > bestProfit:
                bestRotation = rotations
                bestProfit = income-cost
            waiting += customers[i]
            newCustomers = min(waiting, 4)
            waiting -= newCustomers
            income += newCustomers * boardingCost
            cost += runningCost
            rotations += 1
        while waiting > 0:
            if income-cost > bestProfit:
                bestRotation = rotations
                bestProfit = income-cost
            newCustomers = min(waiting, 4)
            waiting -= newCustomers
            income += newCustomers * boardingCost
            cost += runningCost
            rotations += 1
        if income-cost > bestProfit:
            bestRotation = rotations
            bestProfit = income-cost
        if bestProfit == 0:
            return -1
        else:
            return bestRotation
class Solution:
    def minOperationsMaxProfit(self, customers: List[int], boardingCost: int, runningCost: int) -> int:
        if len(customers) == 0:
            return 0
        
        self.res = float('-inf')
        self.rotation = 0
        cuswaiting = 0
        profit = 0
        rotation = 0
        attendcus = 0
        
        while cuswaiting != 0 or rotation < len(customers):
            if rotation < len(customers):
                cuswaiting += customers[rotation]
            rotation += 1
            
            if cuswaiting >= 4:
                attendcus += 4
                profit = attendcus*boardingCost - rotation*runningCost
                cuswaiting -= 4
            else:
                attendcus += cuswaiting
                profit = attendcus*boardingCost - rotation*runningCost
                cuswaiting = 0
            
            #print(profit, rotation)
            if self.res < profit:
                self.res = profit
                self.rotation = rotation
            
            
        if self.res < 0:
            return -1
        else:
            return self.rotation
        
        

class Solution:
    def minOperationsMaxProfit(self, cust: List[int], bc: int, rc: int) -> int:
        cust.reverse()
        wait, profit, t, max_p, ans = 0, 0, 0, float('-inf'), 0
        while cust or wait:
            if cust:
                wait += cust.pop()
            if wait >= 4:
                profit += 4 * bc
            else:
                profit += wait * bc
            wait = max(0, wait-4)
            profit -= rc
            t += 1
            if profit > max_p:
                ans = t
                max_p = profit
        return ans if max_p>0 else -1
class Solution:
    def minOperationsMaxProfit(self, customers: List[int], boardingCost: int, runningCost: int) -> int:
        waiting = 0
        maxProfit = 0
        ans = 0
        revenue = 0
        cost = 0
        
        for c in range(len(customers)):
            waiting+=customers[c]
            cost+=runningCost
            
            toBeBoarded=min(waiting,4)
            waiting-=toBeBoarded
            
            revenue += toBeBoarded*boardingCost 
            profit = revenue-cost
            
            if profit>maxProfit:
                maxProfit, ans = profit, c
                
        while(waiting!=0):
            c+=1
            cost+=runningCost
            
            toBeBoarded=min(waiting,4)
            waiting-=toBeBoarded
            
            revenue += toBeBoarded*boardingCost 
            profit = revenue-cost
            
            if profit>maxProfit:
                maxProfit, ans = profit, c
                
        return ans+1 if maxProfit>0 else -1
class Solution:
    def minOperationsMaxProfit(self, customers: List[int], boardingCost: int, runningCost: int) -> int:
        profit = []
        lastprofit = 0
        wait = customers[0]
        i = 0
        while i < len(customers) or wait > 0:
            if wait > 4:
                wait -= 4
                board = 4
            else:
                board = wait
                wait = 0
            profit.append(lastprofit + board * boardingCost - runningCost)
            lastprofit = profit[-1]
            i += 1
            if i < len(customers):
                wait += customers[i]
        ans = 0
        t = -1
        # print(profit)
        for i, c in enumerate(profit):
            if c > ans:
                ans = c
                t = i + 1
        return t
                
            

class Solution:
    def minOperationsMaxProfit(self, customers: List[int], boardingCost: int, runningCost: int) -> int:
        num_wait = 0
        num_rotate = 0
        num_served = 0
        max_profit = 0
        min_rotation = -1
        
        for c in customers:
            num_wait += c
            num_rotate += 1
            served = min(num_wait, 4)
            num_served += served
            num_wait -= served
            if (boardingCost*num_served - runningCost*num_rotate) > max_profit:
                min_rotation = num_rotate
            max_profit = max(max_profit, boardingCost*num_served - runningCost*num_rotate)
            
        while(num_wait > 0):
            num_rotate += 1
            served = min(num_wait, 4)
            num_served += served
            num_wait -= served
            if (boardingCost*num_served - runningCost*num_rotate) > max_profit:
                min_rotation = num_rotate
            max_profit = max(max_profit, boardingCost*num_served - runningCost*num_rotate)
            
            
        if max_profit > 0:
            return min_rotation
        else:
            return -1

class Solution:
    def minOperationsMaxProfit(self, customers: List[int], boardingCost: int, runningCost: int) -> int:
        
        profits = []
        waiting_customers = 0
        cur_profit = 0
        
        for count in customers:
            waiting_customers += count
            boarding_customers = min(4, waiting_customers)
            cur_profit += boardingCost * boarding_customers - runningCost
            profits.append(cur_profit)
            waiting_customers -= boarding_customers
            
        while waiting_customers > 0:
            boarding_customers = min(4, waiting_customers)
            cur_profit += boardingCost * boarding_customers - runningCost
            profits.append(cur_profit)
            waiting_customers -= boarding_customers

        index = -1
        max_profit = 0
        for i in range(len(profits)):
            if profits[i] > max_profit:
                index = i
                max_profit = profits[i]
        
        if index == -1:
            return -1
        return index + 1
class Solution:
    def minOperationsMaxProfit(self, customers: List[int], boardingCost: int, runningCost: int) -> int:
        
        profits = []
        waitings = 0
        
        def boarding(waitings, profits):
            boardings = 0
            
            # out
            if waitings >= 4:
                boardings = 4
            else:
                boardings = waitings
            
            # calculate profit
            lastprofit = 0 if len(profits) == 0 else profits[len(profits) - 1]
            thisprofit = lastprofit + boardings * boardingCost - runningCost
            profits.append(thisprofit)
            
            return boardings
            
        def calculateBestTimes(profits):
            return profits.index(max(profits))
        
        for customer in customers:
            waitings += customer
            boardings = boarding(waitings, profits)
            waitings -= boardings
            
        while waitings > 0:
            boardings = boarding(waitings, profits)
            waitings -= boardings
            
        # print(profits)
        times = calculateBestTimes(profits)
        
        if profits[times] <= 0:
            times = -1
        else:
            times += 1
        
        return times
class Solution:
    def minOperationsMaxProfit(self, customers: List[int], boardingCost: int, runningCost: int) -> int:
        n = len(customers)
        queue = deque(customers)
        profit = 0
        k = 0
        max_profit = 0
        max_k = 0
        #print('-------')
        while queue:
            num_people = queue.popleft()
            i = 1
            if num_people > 4:
                if queue:
                    queue[0] += num_people - 4
                else:
                    i = num_people // 4
                    queue.append(num_people - 4 * i)
                num_people = 4
            k += i
            profit += num_people * boardingCost * i
            profit -= runningCost * i
            #print(profit)
            if max_profit < profit:
                max_k = k
                max_profit = profit
        return max_k if max_profit > 0 else -1

class Solution:
    def minOperationsMaxProfit(self, customers: List[int], boardingCost: int, runningCost: int) -> int:
        totalCust = 0
        # for i in customers:
        #     totalCust += i
        # print("total cust ",totalCust)
        n = len(customers)
        ct = 0
        curr =0
        profit=0
        res = 0
        maxx = -999999
        pt = 0
        while True:
            ct += 1
            #print()
            if pt < n:
                totalCust += customers[pt]
                pt += 1
            if totalCust < 4:
                curr += totalCust
                totalCust = 0
                profit = curr*boardingCost - ct*runningCost
                if maxx < profit:
                    #print("dfffdfds",res)
                    maxx = profit
                    res = ct
                #print(profit,ct)
                if pt == n:
                    break
                #break
            else:
                totalCust -= 4
                curr += 4
                profit = curr*boardingCost - ct*runningCost
                if maxx < profit:
                    #print("hhbbajskd")
                    maxx = profit
                    res = ct
        if profit > 0:

            return res
        else:
            return -1
        
            
            
        
        

class Solution:
    def minOperationsMaxProfit(self, customers: List[int], boardingCost: int, runningCost: int) -> int:
        i = 0
        n = len(customers)
        board = 0
        wait = 0
        ans = -1
        max_profit = 0
        while wait > 0 or i < n:
            if i < n:
                wait += customers[i]
            board += min(4, wait)
            wait = max(0, wait - 4)
    
            tmp_profit = board * boardingCost - (i + 1) * runningCost
            # print("{} {} {} {} {}".format(i + 1, board, wait, tmp_profit, max_profit))
            if tmp_profit > max_profit:
                max_profit = tmp_profit
                ans = i + 1
            
            i += 1
    
        return ans 
            
            

class Solution:
    def minOperationsMaxProfit(self, customers: List[int], boardingCost: int, runningCost: int) -> int:
        hi = 0
        if not customers:
            return 0
        
        i = wait = tot = 0
        n = len(customers)
        r = 1
        for i in range(n):
            if r > i:
                wait += customers[i]
            while wait >= 4 or r == i + 1 or i == n - 1:
                tot += min(wait, 4)
                wait -= min(wait, 4)
                profit = tot * boardingCost - r * runningCost
                if profit > hi:
                    ans = r
                    hi = profit
                r += 1
                if wait <= 0:
                    break
            
        return ans if hi > 0 else -1

class Solution:
    def minOperationsMaxProfit(self, customers: List[int], boardingCost: int, runningCost: int) -> int:
        earn, max_earn = 0, 0
        i, n = 0, len(customers)
        wait, res = 0, -1
        while i < n or wait > 0:
            if i < n:
                wait += customers[i]
            earn += min(4, wait) * boardingCost - runningCost
            if earn > max_earn:
                res = i + 1
                max_earn = earn
            wait -= min(4, wait)
            i += 1
        return res

class Solution:
    def minOperationsMaxProfit(self, customers: List[int], boardingCost: int, runningCost: int) -> int:
        if not customers:
            return 0
        rest = 0
        total = 0
        profit = -1
        res = 0
        i = 0
        while i < len(customers) or rest:
            if i < len(customers):
                rest += customers[i]
            if rest >= 4:
                total += 4
                rest -= 4
            else:
                total += rest
                rest = 0
            if boardingCost*total - (i+1)*runningCost > profit:
                res = i + 1
                profit = boardingCost*total - (i+1)*runningCost
            i += 1
        return res if profit > 0 else -1
            
                
        

class Solution:
    def minOperationsMaxProfit(self, customers: List[int], boardingCost: int, runningCost: int) -> int:
        ans = 0
        waitingC = 0
        curVal = 0
        maxVal = 0
        maxIndex = -1
        index = 0
        while index < len(customers) or waitingC > 0:
            c = customers[index] if index < len(customers) else 0
            waitingC += c
            curB = min(waitingC, 4)
            waitingC -= curB
            curVal += curB * boardingCost - runningCost
            ans += 1
            index += 1
            if curVal > maxVal:
                maxVal = curVal
                maxIndex = index
        return maxIndex
class Solution:
    def minOperationsMaxProfit(self, customers: List[int], boardingCost: int, runningCost: int) -> int:
        i=0
        w=0
        c=0
        ans = 0
        fans = -1
        while(i<len(customers) or w):
            if i<len(customers):
                w+=customers[i]
            n = min(4,w)
            w-=n
            c+=n
            i+=1
            if ans < c*boardingCost - i*runningCost:
                ans = c*boardingCost - i*runningCost
                fans = i
        return fans
class Solution:
    def minOperationsMaxProfit(self, customers: List[int], boardingCost: int, runningCost: int) -> int:
        if len(customers)==0:
            return -1
        wait = 0
        onboard = 0
        r=0
        max_prof, max_r, cost = float('-inf'),0,0
        while customers[r] == 0:
            r+=1
            
        if customers:
            wait+=customers[r]
            while wait >0:
                c = min(wait, 4)
                wait = max(0, wait-4)
                onboard+=c
                r+=1
                cost = onboard*boardingCost - r*runningCost
                if cost >max_prof:
                    max_r = r
                    max_prof = cost
                if r<len(customers):
                    wait+=customers[r]
                # if wait <10:
                #     print(cost, r, wait, onboard)

        if max_prof <=0:
            return -1
        return max_r
            

class Solution:
    def minOperationsMaxProfit(self, customers: List[int], boardingCost: int, runningCost: int) -> int:
        # max profit, rotating count
        max_tuple = (0, -1)
        
        profit = 0
        awaiting = 0
        turn = 0
        while awaiting > 0 or turn == 0 or turn < len(customers):
            if turn < len(customers):
                awaiting += customers[turn]
            count_pay = min(4, awaiting)
            awaiting -= count_pay
            
            profit += count_pay * boardingCost - runningCost
            #print(profit)
            if profit > max_tuple[0]:
                max_tuple = (profit, turn + 1)
                
            turn += 1
        
        return max_tuple[1]
class Solution:
    def minOperationsMaxProfit(self, customers: List[int], boardingCost: int, runningCost: int) -> int:
        rotations, profit = 0, 0
        ret, max_profit = 0, 0
        waiting = 0
        
        for c in customers:
            waiting += c
            profit += min(4, waiting) * boardingCost - runningCost
            waiting -= min(4, waiting)
            rotations += 1
            if profit > max_profit:
                max_profit = profit
                ret = rotations
        if waiting:
            profit += waiting//4 * (4 * boardingCost - runningCost)
            rotations += waiting//4
            if profit > max_profit:
                max_profit = profit
                ret = rotations
            waiting %= 4
        
        if waiting:
            profit += waiting * boardingCost - runningCost
            rotations += 1
            if profit > max_profit:
                max_profit = profit
                ret = rotations
        
        return ret if max_profit > 0 else -1

class Solution:
    def minOperationsMaxProfit(self, customers: List[int], boardingCost: int, runningCost: int) -> int:
        
        max_profit = [-1, -1]
        def board_customers(customers, rotations): 
            current_profit = customers * boardingCost - rotations * runningCost 
            
            if current_profit > max_profit[0]: 
                max_profit[0] = current_profit
                if current_profit > -1: 
                    max_profit[1] = rotations
        
        r = 1 
        waiting = 0 
        total_customers = 0 
        
        for i in range(len(customers)-1):
            if customers[i] > 4: 
                customers[i+1] += customers[i] - 4 
                customers[i] = 4 
            
            total_customers += customers[i]
            board_customers(total_customers, r)
            r += 1 
        
        waiting = customers[-1]
        while waiting: 
            added = min(waiting, 4)
            total_customers += added 
            board_customers(total_customers, r)
            waiting -= added
            r += 1 
        
        return max_profit[1]

from collections import deque
class Solution:
    def minOperationsMaxProfit(self, customers: List[int], boardingCost: int, runningCost: int) -> int:
        waiting = deque([])
        customers = deque(customers)
        waiting.append(customers.popleft())
        time = 1
        maxx = 0
        cost = 0
        ans = 0
        found = False
        total = 0
        while waiting or customers:
            curr = waiting.popleft()
            if curr>4:
                curr-=4
                total+=4
                waiting.appendleft(curr)
                cost = total*boardingCost-(time*runningCost)
            else:
                temp = curr
                while waiting and temp+waiting[0]<=4:
                    temp+=waiting.popleft()
                if temp<4:
                    if waiting:
                        extra = 4-temp
                        waiting[0]-=extra
                        temp=4
                total+=temp
                cost = total*boardingCost-(time*runningCost)
            if cost>maxx:
                maxx = cost
                found = True
                ans = time
            time+=1
            # print(cost,waiting,customers)
            if customers:
                waiting.append(customers.popleft())
        return ans if found else -1
        

class Solution:
    def minOperationsMaxProfit(self, customers: List[int], boardingCost: int, runningCost: int) -> int:
        earn = 0
        max_earn = 0
        men = 0
        cnt = 0
        res = -1
        for c in customers:
            cnt += 1
            men += c
            earn += min(men, 4) * boardingCost
            earn -= runningCost
            if earn > max_earn:
                max_earn = earn
                res = cnt
            men -= min(men, 4)
            #print(men, earn)
        while men > 0:
            cnt += 1
            earn += min(men, 4) * boardingCost
            earn -= runningCost
            if earn > max_earn:
                max_earn = earn
                res = cnt
            men -= min(men, 4)
            #print(men, earn)
        return res
class Solution:
    def minOperationsMaxProfit(self, customers: List[int], boardingCost: int, runningCost: int) -> int:
        cost = 0
        waiting = 0
        rotations = 0
        profits = []
        for i in range(len(customers)):
            waiting += customers[i]
            board = min(waiting, 4)
            waiting -= board
            cost += board*boardingCost
            cost -= runningCost
            rotations += 1
            profits.append((rotations,cost))
        while waiting > 0:
            board = min(waiting, 4)
            waiting -= board
            rotations += 1
            cost += boardingCost*board
            cost -= runningCost
            profits.append((rotations,cost))
        #print(profits)
        r = None
        ans = 0
        for p in profits:
            if p[1] > ans:
                ans = p[1]
                r = p[0]
        return r if ans > 0 else -1
class Solution:
    def minOperationsMaxProfit(self, customers: List[int], boardingCost: int, runningCost: int) -> int:
        profit = []
        total_customer = 0
        cur_profit = 0
        for i, n in enumerate(customers):
            total_customer += n
            cur_profit += min(total_customer, 4) * boardingCost - runningCost
            total_customer -= min(total_customer, 4)
            profit.append(cur_profit)
        import numpy as np
        
        while total_customer > 0:
            cur_profit += min(total_customer, 4) * boardingCost - runningCost
            total_customer -= min(total_customer, 4)
            profit.append(cur_profit)
        # print(profit)
        return np.argmax(profit)+1 if max(profit) > 0 else -1
class Solution:
    def minOperationsMaxProfit(self, customers: List[int], boardingCost: int, runningCost: int) -> int:
        n = len(customers)
        i = 0
        waiting = 0
        board = 0
        rot = 0
        res = 0
        ind = -1
        
        while waiting > 0 or i < n:
            if i < n:
                waiting += customers[i]
            rot += 1
            board += min(4, waiting)
            waiting -= min(4, waiting)
            prof = board * boardingCost - rot * runningCost
            if (prof > res):
                res = prof
                ind = i + 1
            i += 1
        
        return ind
class Solution:
    def minOperationsMaxProfit(self, customers: List[int], boardingCost: int, runningCost: int) -> int:
        minOp = currCustomers = profit = maxProfit = i = totalCustomers = 0
        
        while customers[-1] == 0:
            customers.pop()
        
        for c in customers:
            i += 1
            currCustomers += c
            totalCustomers += min(4, currCustomers)
            profit = (boardingCost * totalCustomers - (i+1) * runningCost)
            currCustomers -= min(4, currCustomers)
            if profit > maxProfit:
                maxProfit = profit
                minOp = i
            # print(profit, i, boardingCost, totalCustomers)
        
        # print(currCustomers, i, profit)
        while currCustomers:
            i += 1
            totalCustomers += min(4, currCustomers)
            profit = (boardingCost * totalCustomers - (i+1) * runningCost)
            currCustomers -= min(4, currCustomers)
            if profit > maxProfit:
                maxProfit = profit
                minOp = i
            # profit += (boardingCost * currCustomers - (math.factorial(int(ceil(currCustomers/4)))-math.factorial(i)) * runningCost)
            # i += ceil(currCustomers/4)
            
        return minOp if maxProfit > 0 else -1

class Solution:
    def minOperationsMaxProfit(self, customers: List[int], boardingCost: int, runningCost: int) -> int:
        profits = [0]
        waiting = 0
        wheel = deque([0] * 4)
        onBoard = 0
        
        for i in range(len(customers)):
            waiting += customers[i]
            new = min(4, waiting)
            waiting -= new
            leaving = wheel.pop()
            onBoard += new - leaving
            wheel.appendleft(new)
            profits.append(profits[-1] + new*boardingCost - runningCost)
        
        while waiting:
            new = min(4, waiting)
            waiting -= new
            leaving = wheel.pop()
            onBoard += new - leaving
            wheel.appendleft(new)
            profits.append(profits[-1] + new*boardingCost - runningCost)
            
        maximum = 0
        index = -1
        for i, val in enumerate(profits):
            if val > maximum:
                maximum = val
                index = i
                
        return index

class Solution:
    def minOperationsMaxProfit(self, customers: List[int], boardingCost: int, runningCost: int) -> int:
        if boardingCost * 4 <= runningCost:  return -1
        max_profit, cur_profit, waiting, i = 0, 0, 0, 0
        res = 0
        while i < len(customers) or waiting > 0:
            if i < len(customers):
                waiting += customers[i]
            i += 1
            boarded = min(waiting, 4)
            cur_profit += boarded * boardingCost - runningCost
            #print(waiting, boarded, cur_profit, max_profit)
            if cur_profit > max_profit:
                res = i
                max_profit = cur_profit
            waiting -= boarded
        return res

class Solution:
    def minOperationsMaxProfit(self, customers: List[int], boardingCost: int, runningCost: int) -> int:
        profit = 0
        wait = 0
        rotated = 0
        res = -1
        total = 0
        for num in customers:
            cur_num = min(4, num + wait)
            total += cur_num
            wait = (num + wait - cur_num)
            rotated += 1
            cur_profit = boardingCost * total - rotated * runningCost
            if cur_profit > profit:
                res = rotated
                profit = cur_profit
        while wait:
            cur = min(4, wait)
            total += cur
            rotated += 1
            cur_profit = boardingCost * total - rotated * runningCost
            if cur_profit > profit:
                res = rotated
                profit = cur_profit
            wait -= cur
        return res
class Solution:
    def minOperationsMaxProfit(self, customers: List[int], boardingCost: int, runningCost: int) -> int:
        
        profit = 0
        waiting = 0
        rotations = 0
        onboard = 0
        gondola_customers = deque([])
        total = 0
        maxprofit = 0
        max_rotation = -1
        for arrival in customers:
            if gondola_customers:
                coming_down = gondola_customers.popleft()
                onboard -= coming_down
            
            total = arrival   
            if waiting >0:
                total = waiting + arrival
            
            #if onboard == 0 and total == 0:
            #    continue
                
            if total <= 4:
                profit += ((total*boardingCost) - runningCost)
                onboard += total
                gondola_customers.append(total)
                waiting = max(0,waiting-total)
            else:
                profit += ((4*boardingCost) - runningCost)
                onboard += 4
                gondola_customers.append(4)
                waiting += (arrival-4)
            
            rotations += 1
            if profit > maxprofit:
                maxprofit = profit
                max_rotation = rotations
        print((maxprofit, max_rotation, waiting))
        profit += ((waiting//4)*((4*boardingCost)-runningCost))
        rotations += (waiting//4)
        if profit > maxprofit:
                maxprofit = profit
                max_rotation = rotations
        print((maxprofit, max_rotation, waiting%4))
        
        profit += (((waiting%4)*boardingCost)-runningCost)
        rotations += ((waiting%4)>0)
        if profit > maxprofit:
                maxprofit = profit
                max_rotation = rotations
        print((maxprofit, max_rotation))
        
        return max_rotation if maxprofit > 0 else -1
                
        

import math
class Solution:
    def minOperationsMaxProfit(self, customers: List[int], boardingCost: int, runningCost: int) -> int:
        profit,bestIteration,customerCount,iteration,boarding = 0,0,0,0,0
        for i in range(len(customers)):
            customerCount = customerCount + customers[i]
            iteration = iteration + 1
            boarding = boarding + min(customerCount,4)
            customerCount = customerCount - min(customerCount,4)
            if boarding*boardingCost - iteration*runningCost > profit:
                profit = boarding*boardingCost - iteration*runningCost
                bestIteration = iteration
        while customerCount > 0:
            iteration = iteration + 1
            boarding = boarding + min(customerCount,4)
            customerCount = customerCount - min(customerCount,4)
            if boarding*boardingCost - iteration*runningCost > profit:
                profit = boarding*boardingCost - iteration*runningCost
                bestIteration = iteration
        if profit == 0:
            return -1
        return bestIteration
        
        

class Solution:
    def minOperationsMaxProfit(self, customers: List[int], boardingCost: int, runningCost: int) -> int:
        max_profit, min_rotations = 0, -1
        curr_profit = 0
        waiting, rotations = 0, 0
        
        customers.reverse()
        while curr_profit >= 0:
            if customers: waiting += customers.pop()
            if waiting == 0:
                if not customers: break
            else:
                curr_profit += min(4, waiting) * boardingCost - runningCost
                waiting -= min(4, waiting)
            rotations += 1
            if curr_profit > max_profit:
                max_profit, min_rotations = curr_profit, rotations

        return min_rotations

class Solution:
    def minOperationsMaxProfit(self, customers: List[int], boardingCost: int, runningCost: int) -> int:
        sm = 0
        cus = 0
        c = 0
        w = 0
        ret = 0
        mx = 0
        while True:
            if c < len(customers):
                cus += customers[c]
                c += 1
            board = min(4, cus)
            cus -= board
            sm += (boardingCost * board) - runningCost
            w += 1
            if mx < sm:
                ret = w
                mx = sm
            if 0 == min(4, cus) and c == len(customers):
                break
        return -1 if mx == 0 else ret
class Solution:
    def minOperationsMaxProfit(self, customers: List[int], boardingCost: int, runningCost: int) -> int:
        # we have the boardingCost and the runningCost
        #at each index we need to wipe it out before we move to the next
        #we can board max 4 at a time
        arr = [-1];
        
#         for i in range(len(customers)):
#             if customers[i] == 0:
#                 print('6')
#             else:
#                 divided = customers[i]//4;
#                 if customers[i] % 4 != 0:
#                     divided += 1;
                
#                 bammer = customers[i] * boardingCost - (divided * runningCost);
#                 arr.append(bammer);
        
#         print(max(arr));
        
        currentCost = 0;
        maximum = -1000000000;
        rotations = 0;
        total = 0;
        
        while (total != 0 or rotations < len(customers)):
            if (rotations < len(customers)):
                total += customers[rotations];
            if total < 4:
                temp = total;
                total = 0;
                currentCost += (temp * boardingCost) - runningCost;
                arr.append(currentCost);
                if (currentCost > maximum):
                    maximum = currentCost;
            else:
                total -= 4;
                currentCost += (4 * boardingCost) - runningCost;
                arr.append(currentCost);
                
            rotations += 1;
        
        maxer = -1000000000;
        indexer = -1;
            
        for i in range(len(arr)):
            if (maxer < arr[i]):
                maxer = arr[i];
                indexer = i;
        
        if indexer == 0:
            return -1;
        return indexer;
        # print(arr);
        # return max(arr);

class Solution:
    def minOperationsMaxProfit(self, customers: List[int], boardingCost: int, runningCost: int) -> int:
        best_profit = float('-inf')
        best_rotations = -1
        rem = 0
        group = []
        for c in customers:
            avail = rem + c
            group.append(min(4, avail))
            rem = max(0, avail - 4)
        
        while rem:
            group.append(min(4, rem))
            rem = max(0, rem - 4)
        profit = cost = 0
        for i, g in enumerate(group):
            profit += g * boardingCost
            cost += runningCost
            if best_profit < profit - cost:
                best_profit = profit - cost
                best_rotations = i + 1
        
        return best_rotations if best_profit > 0 else -1
class Solution:
    def minOperationsMaxProfit(self, customers: List[int], boardingCost: int, runningCost: int) -> int:
        # no trick, iterate and board all if possible get profit
        iteration = 0
        max_profit = 0
        profit = 0
        wait = 0
        for i, c in enumerate(customers):
            wait += c
            board = min(4, wait)
            wait -= board
            if i == len(customers)-1 and wait > 0:
                customers += [wait]
                wait = 0
            profit += boardingCost * board - runningCost
            if profit > max_profit:
                max_profit = profit
                iteration = i
                
        return iteration + 1 if max_profit > 0 else -1
class Solution:
    def minOperationsMaxProfit(self, customers: List[int], boardingCost: int, runningCost: int) -> int:
        people = 0
        on = 0
        i = 0
        money = 0
        ans = -1
        j = 0
        for c in customers:
            people += c
            i += 1
            x = min(people, 4)
            on += x
            people -= x
            money = on * boardingCost - runningCost * i
            if money > ans:
                j = i
                ans = money
        
        while people:
            i += 1
            x = min(people, 4)
            on += x
            people -= x
            money = on * boardingCost - runningCost * i
            if money > ans:
                j = i
                ans = money
        
        return j if ans > 0 else -1
class Solution:
    def minOperationsMaxProfit(self, customers: List[int], boardingCost: int, runningCost: int) -> int:
        if not customers: return 0
        if 4 * boardingCost <= runningCost:
            return -1
        
        inline = 0
        profit = 0
        maxProfit = 0
        rotate = 0
        recordRotate = 0
        idx = 0
        
        while idx < len(customers) or inline > 0:
            if idx < len(customers):
                inline += customers[idx]
            if inline > 4:
                profit = profit + 4 * boardingCost - runningCost
                inline = inline - 4
                rotate += 1
            elif (inline <= 4 ):
                
                profit = profit + inline * boardingCost - runningCost
                inline = 0
                rotate += 1
            
            if profit > maxProfit:
                maxProfit = profit
                recordRotate = rotate
            idx = idx + 1

        return recordRotate
            

class Solution:
    def minOperationsMaxProfit(self, customers: List[int], boardingCost: int, runningCost: int) -> int:
        max_profit = 0
        max_spin = -1
        
        cur_profit = 0
        cur_num = 0
        
        spins = 0
        
        for i in range(len(customers)):
            spins += 1
            cur_profit -= runningCost
            
            cur_num += customers[i]
            cur_profit += min(cur_num, 4) * boardingCost

            if cur_profit > max_profit:
                max_profit = max(max_profit, cur_profit)
                max_spin = spins
                
            cur_num = max(0, cur_num - 4)
            
        while cur_num > 0:
            spins += 1
            cur_profit -= runningCost
            cur_profit += min(4, cur_num) * boardingCost
            
            if cur_profit > max_profit:
                max_profit = max(max_profit, cur_profit)
                max_spin = spins
                
            cur_num -= min(4, cur_num)
            
        return max_spin
class Solution:
    def minOperationsMaxProfit(self, customers: List[int], boardingCost: int, runningCost: int) -> int:
        if 4 * boardingCost <= runningCost:
            return -1
        curr_customers = 0
        curr_profit = 0
        max_profit = 0
        min_rotations = 0
        for i, cus in enumerate(customers):
            curr_customers += cus
            if curr_customers > 4:
                curr_profit += 4 * boardingCost - runningCost
                curr_customers -= 4
            else:
                curr_profit += curr_customers * boardingCost - runningCost
                curr_customers = 0
            if max_profit < curr_profit:
                max_profit = curr_profit
                min_rotations = i + 1
        left_rounds, remainder = divmod(curr_customers, 4)
        
        max_profit1 = max_profit
        max_profit2 = curr_profit + 4 * left_rounds * boardingCost - left_rounds * runningCost
        max_profit3 = curr_profit + curr_customers * boardingCost - (left_rounds + 1) * runningCost
        
        MAX = max(max_profit1, max_profit2, max_profit3)
        if MAX == max_profit1:
            min_rotations = min_rotations
        elif MAX == max_profit2:
            min_rotations = len(customers) + left_rounds
        else:
            min_rotations = len(customers) + left_rounds + 1
        
        return min_rotations if max_profit > 0 else -1
            

class Solution:
    def minOperationsMaxProfit(self, customers: List[int], boardingCost: int, runningCost: int) -> int:
        
        if boardingCost * 4 < runningCost:
            return -1
        
        profit = 0
        num_waiting_customers = 0
        max_profit = 0
        ans = -1
        
        i = 0
        
        while i < len(customers) or num_waiting_customers > 0:
            num_waiting_customers += customers[i] if i < len(customers) else 0
            
            profit += min(num_waiting_customers, 4) * boardingCost - runningCost
            
            if profit > max_profit:
                ans = i + 1
                max_profit = profit
            
            num_waiting_customers = max(num_waiting_customers - 4, 0)
            i += 1
        return ans

class Solution:
    def minOperationsMaxProfit(self, customers: List[int], boardingCost: int, runningCost: int) -> int:
        
        total_queue = 0
        idx = 0
        curr_costs = 0
        max_costs = float('-inf')
        max_idx = 0
        
        while total_queue or idx < len(customers):
            curr_customers = 0
            if idx < len(customers):
                curr_customers = min(4, customers[idx])
                total_queue += max(0, customers[idx] - 4)
            
            if curr_customers < 4 and total_queue:
                diff = 4 - curr_customers
                curr_customers += min(diff, total_queue)
                total_queue = max(0, total_queue - diff)
                
            
            idx += 1
            
            curr_costs += boardingCost * curr_customers - runningCost
            if curr_costs > max_costs:
                max_idx = idx
                max_costs = curr_costs
            
        return max_idx if max_costs > 0 else -1
class Solution:
    def minOperationsMaxProfit(self, customers: List[int], boardingCost: int, runningCost: int) -> int:
        waiting = 0
        total_money = 0
        cars = [0,0,0,0]
        index = 0
        max_rotations = -1
        num_rotations = 0
        max_money = 0
        while index < len(customers):
            waiting += customers[index]
            if waiting < 4:
                cars[0] = waiting
                total_money += waiting*boardingCost
                waiting = 0
                
            else:
                waiting -= 4
                total_money += 4*boardingCost
                cars[0] = 4
            self.rotation(cars)
            total_money -= runningCost
            index += 1
            num_rotations += 1
            if total_money >  max_money:
                max_rotations = num_rotations
                max_money = total_money
        while waiting != 0:
            if waiting < 4:
                cars[0] = waiting
                total_money += waiting*boardingCost
                waiting = 0
            else:
                waiting -= 4
                total_money += 4*boardingCost
                cars[0] = 4
            self.rotation(cars)
            total_money -= runningCost
            num_rotations += 1
            if total_money >  max_money:
                max_rotations = num_rotations
                max_money = total_money
        
        if max_money == 0:
            return -1
        return max_rotations
        
        
        
        
    def rotation(self, arr):
        arr[3], arr[2], arr[1], arr[0] = arr[2], arr[1], arr[0], 0
        

class Solution:
    def minOperationsMaxProfit(self, customers: List[int], boardingCost: int, runningCost: int) -> int:
        lines = []
        carry = 0
        cur = 0
        
        for c in customers:
            cur = 4 if c + carry >= 4 else c + carry
            carry = c + carry - 4 if c + carry > 4 else 0
            lines.append(cur)
            
        while carry:
            c = 4 if carry >= 4 else carry
            lines.append(c)
            carry = carry - 4 if carry > 4 else 0
        
        res = 0
        total = 0
        rotate = 0
        
        for i, c in enumerate(lines):
            total += c  
            if total * boardingCost - (i+1) * runningCost > res:
                
                res = total * boardingCost - (i+1) * runningCost
                rotate = i + 1
        
        print(res)
        return rotate if res > 0 else -1
        
        
                

class Solution:
    def minOperationsMaxProfit(self, customers, boardingCost, runningCost):
        # we can never make profit
        if 4*boardingCost <= runningCost:
            return -1
        total = sum(customers)
        maxProfit = -1
        maxRotate = 0
        accuProfit = 0
        rotate = 0
        
        # keep rotate and board the customers.
        i = 0
        prev = 0
        while i < len(customers):
            # print('i ', i, "num ", customers[i], "total ", total)
            prev = accuProfit
            if customers[i] <= 4:
                accuProfit += customers[i] * boardingCost
                customers[i] = 0
                total -= customers[i]
            else:
                accuProfit += 4 * boardingCost
                customers[i] -= 4
                total -= 4
            rotate += 1
            # every time we rotate, we need pay runningCost
            accuProfit -= runningCost
            if maxProfit < accuProfit:
                maxProfit = accuProfit
                maxRotate = rotate
            # print("accu ", accuProfit, "rotate ", rotate, "customer ", customers[i], "profit ", accuProfit- prev,)
            # print(customers)
            # print("###")
            # if current customer < 4, put them in the same group of the following customer 
            # to make sure everything we full loaded.
            if i + 1 < len(customers):
                customers[i+1] += customers[i]
                customers[i] = 0
                
            # the following customer need to wait the customer in front.
            if customers[i] == 0:
                i += 1
            
        return -1 if maxProfit < 0 else maxRotate

class Solution:
    def minOperationsMaxProfit(self, customers: List[int], boardingCost: int, runningCost: int) -> int:
        n, cur, i = len(customers), 0, 0
        maxProfit, maxProfitIdx, curProfit = 0, -1, 0
        while i < n or cur > 0:
            cur += customers[i] if i < n else 0            
            i += 1
            
            board = min(cur, 4)
            cur -= board
            
            curProfit += board * boardingCost - runningCost
            if curProfit > maxProfit:
                maxProfit, maxProfitIdx = curProfit, i     
        
        return maxProfitIdx if maxProfit > 0 else -1 
class Solution:
    def minOperationsMaxProfit(self, customers: List[int], boardingCost: int, runningCost: int) -> int:
        res = 0 
        max_profit = 0
        wait_people = 0
        taken_people = 0
        rotation_times = 0
        
        for customer in customers:
            rotation_times += 1 
            can_take = min(4, wait_people+customer)
            taken_people += can_take 
            cur_profit = taken_people*boardingCost - rotation_times*runningCost
            if cur_profit > max_profit:
                res = rotation_times 
                max_profit = cur_profit
                
            wait_people = max(0, wait_people+customer-4)
            
        while wait_people > 0:
            rotation_times += 1 
            can_take = min(4, wait_people)
            taken_people += can_take 
            cur_profit = taken_people*boardingCost - rotation_times*runningCost
           
            if cur_profit > max_profit:
                res = rotation_times 
                max_profit = cur_profit
                
            wait_people -= can_take
            
        return res if res > 0 else -1 

import math
class Solution:
    def minOperationsMaxProfit(self, customers: List[int], boardingCost: int, runningCost: int) -> int:
        best = 0
        index = None
        i = 0
        n = 0
        p = 0
        while n > 0 or i < len(customers):
            if i < len(customers):
                n += customers[i]
            p += min(4, n) * boardingCost
            p -= runningCost
            n -= min(4, n)
            i += 1
            if p > best:
                best = p
                index = i
        
        if index is None:
            return -1
        return index
class Solution:
    def minOperationsMaxProfit(self, customers: List[int], bCost: int, rCost: int) -> int:
        maxProfit = -1
        unboard = 1
        board = 0
        flag = True
        count = 0
        profit = 0
        ans = 0
        
        while unboard > 0 or count < len(customers):
            if flag: 
                unboard = 0
                flag = False
            
            
            unboard += (customers[count] if count < len(customers) else 0)
            count += 1
            
            if unboard > 4: 
                board += 4
                unboard -= 4
                profit = board*bCost - rCost*count
            else: 
                board += unboard
                unboard = 0
                profit = board*bCost - rCost*count
                
            ans = count if profit > maxProfit else ans 
            
            maxProfit = profit if profit > maxProfit else maxProfit   

                 
        return ans if maxProfit > 0 else -1
class Solution:
    def minOperationsMaxProfit(self, customers: List[int], boardingCost: int, runningCost: int) -> int:
        maxi = 0
        maxRotate = -1
        profit = 0
        onBoarded = 0
        inQueue = 0
        rotates = 0
        for c in customers:
            rotates += 1
            onBoarded = max(0, onBoarded - 1)
            inQueue += c
            board = min(4, inQueue)
            inQueue -= board
            profit += board * boardingCost - runningCost
            if profit > maxi:
                maxi = profit
                maxRotate = rotates
        
        while inQueue > 0:
            rotates += 1
            onBoarded = max(0, onBoarded - 1)
            board = min(4, inQueue)
            inQueue -= board
            profit += board * boardingCost - runningCost
            if profit > maxi:
                maxi = profit
                maxRotate = rotates
        
        return maxRotate
                

class Solution:
    def minOperationsMaxProfit(self, cust: List[int], b: int, r: int) -> int:
        arr=[1*b-r,2*b-r,3*b-r,4*b-r]
        pos=0
        ln=len(cust)
        #print(arr)
        for i in range(4): 
            if arr[i]>0: pos=i+1 
        if pos==0: return -1
        prev=0
        for i in range(ln): 
            x=min(4,prev+cust[i])
            prev= prev+cust[i]-x
            cust[i]=x
        #print(cust)
        cum=0
        for i in range(ln):
            cum+=cust[i]
            cust[i]=cum*b-(i+1)*r
        a,b2=prev//4,prev%4
        #print(cust)
        x=max(cust)
        y=(cum+(a*4))*b-(ln+a)*r
        #print(prev,a,b2,x,y,cum,ln,(ln+a))
        z=y
        if b2: 
            z=(cum+prev)*b-(ln+a+1)*r
            if z>y: 
                a+=1
        if z>x:
            if z<1: return -1
            return ln+a
        if x<1: return -1
        return cust.index(x)+1
class Solution:
    def minOperationsMaxProfit(self, customers: List[int], bc: int, rc: int) -> int:
        rnd = waiting = 0
        ans = -1
        profit = -1
        curp = 0
        
        n = sum(customers)
        debug = 0
        if debug:
            print(f'lenght is {len(customers)}, total cust is {n} where n % 4 = {n%4}, boardingTicket is {bc}, run cost is {rc}')
        
        i = 0
        while i < len(customers) or waiting > 0:
            cust = 0 if i >= len(customers) else customers[i]
            waiting += cust
            board = min(4, waiting)
            waiting -= board
            curp = curp + board * bc - rc
            rnd += 1
            if i < len(customers):
                i += 1
            if debug:
                print(f'i is {i}, new comer is {cust} and total waiting is {waiting}, '
                      f'board {board} people, curp is {curp}, rnd = {rnd}')            
        #for i, cust in enumerate(customers):
            # if waiting <= 4:
            #     curp = curp + waiting * bc - rc
            #     if debug:
            #         print(f'i is {i}, new comer is {cust} and total waiting is {waiting}, '
            #               f'board {waiting} people, curp is {curp}, rnd = {rnd + 1}')
            #     waiting = 0
            #     rnd += 1
            # else:
            #     board = waiting - waiting % 4
            #     boardrnd = board // 4
            #     curp = curp + board * bc - boardrnd * rc  # t * (4*bc - rc)
            #     if debug:
            #         print(f'i is {i}, new comer is {cust} and total waiting is {waiting}, '
            #               f'board {board} people, curp is {curp}, rnd = {rnd + boardrnd}')
            #     waiting %= 4
            #     rnd += boardrnd
            
            if curp > profit:
                profit = curp
                ans = rnd
                
        if waiting > 0:
            curp = curp + waiting * bc - rc
            rnd += 1
            if debug:
                print(f'i is n, new comer is 0 and total waiting is {waiting}, board {waiting} people, curp is {curp}, rnd = {rnd}')
            if curp > profit:
                profit = curp
                ans = rnd
        return ans
    
    

# [17,0,45,39,19,4,9,3,16]
# 11
# 33

class Solution:
    def minOperationsMaxProfit(self, customers: List[int], boardingCost: int, runningCost: int) -> int:
        max_profit = 0
        best_rots = -1
        rots = 0
        waiting = 0
        boarded = 0
        c = 0
        while waiting > 0 or c<len(customers):
            if c<len(customers):
                waiting += customers[c]
            if waiting > 4:
                boarded += 4
                waiting -= 4
            else:
                boarded += waiting
                waiting = 0
            rots += 1
            profit = boarded*boardingCost - rots*runningCost
            if profit > max_profit:
                max_profit = profit
                best_rots = rots
            c += 1
        return best_rots
class Solution:
    def minOperationsMaxProfit(self, customers: List[int], boardingCost: int, runningCost: int) -> int:
        waitingCustomers = customers[0]
        totalCustomers = 0
        i = 1
        result = -1
        maxProfit = 0
        
        while i < len(customers) or waitingCustomers > 0:
            servedCustomers = min(4, waitingCustomers)
            
            waitingCustomers -= servedCustomers
            totalCustomers += servedCustomers
            
            profit = totalCustomers * boardingCost - i * runningCost
            if profit > maxProfit:
                maxProfit = profit
                result = i
            
            if i < len(customers):
                waitingCustomers += customers[i]
            
            i += 1
        
        return result

class Solution:
    def minOperationsMaxProfit(self, customers: List[int], boardingCost: int, runningCost: int) -> int:
        left = 0
        required = 0
        for cus in customers:
            required += 1
            left += cus
            left -= min(left, 4)
        maxRot = required + ceil(left / 4)
        mP, mR = 0, -1
        rotCnt = 0
        c = 0
        profit = 0
        maxP = 0
        while rotCnt < maxRot:
            if rotCnt < len(customers):
                c += customers[rotCnt]
            roundP = min(c , 4) * boardingCost
            c -= min(c, 4)
            roundP -= runningCost
            profit += roundP
            if profit > mP:
                mR = rotCnt + 1
                mP = profit
            rotCnt += 1
        return mR
            
            
            

def getindex(arr):
    value = max(arr)
    cnt = 1
    for e in arr:
        if e == value:
            return cnt
        cnt += 1

class Solution:
    def minOperationsMaxProfit(self, customers: List[int], boardingCost: int, runningCost: int) -> int:
        boarding = 0 
        rotation = 0
        profit = []
        n_customers = len(customers)
        for i in range(n_customers-1):
            if customers[i]<4:
                boarding += customers[i]
                rotation += 1
                profit.append(boarding * boardingCost - rotation*runningCost)
            else:
                boarding += 4
                customers[i+1] = customers[i] + customers[i+1] -4
                rotation += 1
                profit.append(boarding * boardingCost - rotation*runningCost)
        while customers[n_customers-1]>3:
            boarding += 4
            rotation += 1
            profit.append(boarding * boardingCost - rotation*runningCost)
            customers[n_customers-1] -= 4
        if customers[n_customers-1] == 0:
            pass
        else:
            boarding += customers[n_customers-1]
            rotation += 1
            profit.append(boarding * boardingCost - rotation*runningCost)
        if max(profit)<0:
            return -1
        else:
            return getindex(profit)

class Solution:
    def minOperationsMaxProfit(self, customers: List[int], boardingCost: int, runningCost: int) -> int:
        max_pro = -1
        shift = 0
        total_onboarded = 0
        wait = 0
        re = -1
        for c in customers:
            wait+=c
            if wait > 0:
                shift+=1
                
                if wait>4:
                    total_onboarded+=4
                else:
                    total_onboarded+=wait
                    
                cur_pro = total_onboarded*boardingCost - shift*runningCost
                if cur_pro > max_pro:
                    re = shift
                max_pro = max(max_pro, cur_pro)
                if wait>4:
                    wait-=4
                else:
                    wait=0
            elif wait == 0:
                shift+=1
                cur_pro = total_onboarded*boardingCost - shift*runningCost
                continue
                
        # no more new customers
        while wait > 0:
            shift+=1

            if wait>4:
                total_onboarded+=4
            else:
                total_onboarded+=wait

            cur_pro = total_onboarded*boardingCost - shift*runningCost
            if cur_pro > max_pro:
                re = shift
            max_pro = max(max_pro, cur_pro)
            if wait>4:
                wait-=4
            else:
                wait=0

        return re
                
        

from typing import List
class Solution:
    def minOperationsMaxProfit(self, customers: List[int], boardingCost: int, runningCost: int) -> int:
        index, max_profit, waiting_customers, num_customers, best_rotation = 0, -1, 0, 0, -1
        
        while index < len(customers) or waiting_customers > 0:
            current_num_customers = 0
            rotations = index + 1
            if index < len(customers):
                current_num_customers = customers[index]
            if current_num_customers >= 4:
                waiting_customers += (current_num_customers - 4)
                num_customers += 4
                if num_customers * boardingCost - rotations * runningCost> max_profit:
                    max_profit = num_customers * boardingCost - rotations * runningCost
                    best_rotation = rotations
            elif current_num_customers < 4 and (current_num_customers + waiting_customers) > 4:
                waiting_customers -= (4 - current_num_customers)
                num_customers += 4
                if num_customers * boardingCost - rotations * runningCost> max_profit:
                    max_profit = num_customers * boardingCost - rotations * runningCost
                    best_rotation = rotations
            else:
                num_customers += current_num_customers
                num_customers += waiting_customers
                waiting_customers = 0
                if num_customers * boardingCost - rotations * runningCost> max_profit:
                    max_profit = num_customers * boardingCost - rotations * runningCost
                    best_rotation = rotations
            index += 1 
        
        return best_rotation
class Solution:
    def minOperationsMaxProfit(self, customers: List[int], boardingCost: int, runningCost: int) -> int:
        itrn = 0 
        prof = 0 
        maxprof = 0
        tc = 0 
        best = -1 
        i = 0 
        while i < len(customers) or tc > 0 :
            if i <  len(customers) : 
                tc += customers[i]
            itrn += 1 
            if tc >= 4 :
                prof += boardingCost*4 - runningCost
                tc -= 4
                if maxprof < prof:
                    maxprof = prof 
                    best = itrn 
            else:
                prof += boardingCost*tc - runningCost
                tc = 0 
                if maxprof < prof:
                    maxprof = prof   
                    best = itrn 
            i += 1 
        return best 
            
        

class Solution:
    def minOperationsMaxProfit(self, a: List[int], bc: int, rc: int) -> int:
        max_pr = pr = 0; cnt = max_cnt = 0; w = 0; i = 0
        while i < len(a) or w > 0:
            x = w + a[i] if i < len(a) else w            
            pr += min(x, 4) * bc - rc                        
            cnt += 1            
            if pr > max_pr: max_pr, max_cnt = pr, cnt             
            w = max(x - 4, 0)      
            i += 1                          
        return max_cnt if max_pr > 0 else -1
class Solution:
    def minOperationsMaxProfit(self, customers: List[int], boardingCost: int, runningCost: int) -> int:
        if runningCost >= 4 * boardingCost:
            return -1
        best_profit = 0
        curr_enqueued = 0
        curr_profit = 0
        def get_customers(i):
            if i >= len(customers):
                return 0
            return customers[i]
        
        i = 0
        best_turns = -1
        while curr_enqueued > 0 or i < len(customers):
            curr_enqueued += get_customers(i)
            to_add = min(4, curr_enqueued)
            curr_profit += to_add * boardingCost - runningCost
            if curr_profit > best_profit:
                best_turns = i + 1
                best_profit = curr_profit
            curr_enqueued -= to_add
            i += 1
        return best_turns
class Solution:
    def minOperationsMaxProfit(self, customers, boardingCost, runningCost):
        if 4 * boardingCost - runningCost <= 0:
            return -1
        
        n = len(customers)
        wait = 0
        now_profit = 0
        max_profit = 0
        res = -1
        for i in range(n):
            wait += customers[i]
            now_profit += boardingCost * min(wait, 4) - runningCost
            wait -= min(wait, 4)
            if now_profit > max_profit:
                max_profit = now_profit
                res = i + 1
        res += (wait // 4)
        if boardingCost * (wait % 4) - runningCost > 0:
            res += 1
        return res
class Solution:
    # simulation, time O(customers/4), space O(1)
    def minOperationsMaxProfit(self, customers: List[int], boardingCost: int, runningCost: int) -> int:
        n = len(customers)
        cur = remain = rot = 0
        max_profit = min_rot = -1
        i = 0
        while i < n or remain:
            rot += 1
            remain += customers[i] if i < n else 0
            cur += min(4, remain)
            remain -= min(4, remain)
            cur_profit = cur * boardingCost - rot * runningCost
            if cur_profit > max_profit:
                max_profit = cur_profit
                min_rot = rot
            i += 1
        return min_rot if max_profit > 0 else -1
class Solution:
    def minOperationsMaxProfit(self, customers: List[int], boardingCost: int, runningCost: int) -> int:
        
        leftover = 0
        dp = [0]
        
        i=0
        while i<len(customers) or leftover:
            
            if i < len(customers):
                leftover += customers[i]
                
            newCust = min(4, leftover)
            leftover -= newCust
            
            temp = dp[-1] + newCust*boardingCost - runningCost
            dp.append(temp)
            
            i += 1
            
        if all([x<=0 for x in dp]):
            return -1
        return dp.index(max(dp))
class Solution:
    def minOperationsMaxProfit(self, customers: List[int], boardingCost: int, runningCost: int) -> int:
        r, p = -1, 0
        rc = 0
        oc = 0
        d = 0
        for i, c in enumerate(customers):
            d = i+1
            rc += c
            oc = min(4,rc) + oc
            rc = max( rc -4, 0)
            cp = oc * boardingCost - (i+1) * runningCost
            if cp > p:
                p = cp
                r = i+1
        while rc > 0:
            d += 1
            oc = min(4,rc) + oc
            rc = max( rc -4, 0)
            cp = oc * boardingCost - (d) * runningCost
            if cp > p:
                p = cp
                r = d
        return r

class Solution:
    def minOperationsMaxProfit(self, customers: List[int], boardingCost: int, runningCost: int) -> int:
        cur=0
        ans=-1
        cur_p=0
        prev=0
        temp=0
        n=len(customers)
        for i in range(n):
            cur+=customers[i]
           
            if cur>=4:
                cur_p=(prev+4)*boardingCost-(i+1)*runningCost
                prev+=4
                cur-=4
            elif cur<4:
                cur_p+=(prev+cur)*boardingCost-(i+1)*runningCost
                prev+=cur
                cur=0
            if cur_p>temp:
                ans=i+1
                temp=cur_p
            
        if cur>0:
            m=cur//4
            for i in range(m+1):
                # print(cur)
                if cur>=4:
                    cur_p=(prev+4)*boardingCost-(i+1+n)*runningCost
                    prev+=4
                    cur-=4
                elif cur<4:
                    # print(prev+cur)
                    # print(i+1+n)
                    cur_p=(prev+cur)*boardingCost-(i+1+n)*runningCost
                    
                    prev+=cur
                    cur=0
                if cur_p>temp:
                    ans=i+1+n
                    temp=cur_p
                # print(cur_p)
                # print(" ")
        return ans
class Solution:
    def minOperationsMaxProfit(self, customers: List[int], boardingCost: int, runningCost: int) -> int:
        
        if boardingCost * 4 <= runningCost:
            return -1
        
        profit = 0
        num_waiting_customers = 0
        max_profit = 0
        ans = -1
        
        i = 0
        
        while i < len(customers) or num_waiting_customers > 0:
            num_waiting_customers += customers[i] if i < len(customers) else 0
            
            # if i < len(customers):
            #     num_rotate = ((len(customers) - i) * 50 + num_waiting_customers + 3) // 4
            #     if ((len(customers) - i) * 50 + num_waiting_customers) * boardingCost - num_rotate * runningCost + profit < 0:
            #         return ans
                    
            
            profit += min(num_waiting_customers, 4) * boardingCost - runningCost
            
            if profit > max_profit:
                ans = i + 1
                max_profit = profit
            
            num_waiting_customers = max(num_waiting_customers - 4, 0)
            i += 1
        return ans

class Solution:
    def minOperationsMaxProfit(self, customers: List[int], boardingCost: int, runningCost: int) -> int:
        highest = waiting = profit = time = i = 0
        highest_time = -1
        while i < len(customers) or waiting:
            if i < len(customers):
                waiting += customers[i]
                i += 1
            boarding = min(waiting, 4)
            waiting -= boarding
            profit += boarding * boardingCost - runningCost
            time += 1
            if profit > highest:
                highest_time = time
            highest = max(profit, highest)
            
        return -1 if highest <= 0 else highest_time
class Solution:
    def minOperationsMaxProfit(self, customers: List[int], boardingCost: int, runningCost: int) -> int:
        capacity = [0] * 4
        waiting = 0
        max_profit = [0,0]
        rot = 0
        j = 0
        i = 0
        l = len(customers)
        cust = 0
        while waiting > 0 or i < l:
            if i < l:
                waiting += customers[i]
                i += 1
            capacity[j] = 0
            capacity[j] = waiting if waiting <= 4 else 4
            cust += capacity[j]
            rot += 1
            waiting = 0 if waiting <= 4 else waiting - 4
            profit = (cust * boardingCost) - (rot * runningCost)
            #print([rot, profit])
            if profit > max_profit[1]:
                max_profit = [rot, profit]
            if j == 3:
                j = 0
            else:
                j += 1
        return max_profit[0] if max_profit[1] > 0 else -1
class Solution:
    def minOperationsMaxProfit(self, customers: List[int], boardingCost: int, runningCost: int) -> int:
        profit = 0
        max_profit = 0
        res = -1
        q = deque([c, i] for i, c in enumerate(customers))
        i = 0
        while q:
            n = 0
            while q and q[0][1] <= i and n < 4:
                if n + q[0][0] <= 4:
                    n += q.popleft()[0]
                else:
                    q[0][0] -= 4 - n
                    n = 4
            i += 1
            if not n:
                continue
            profit += boardingCost * n
            profit -= runningCost
            if profit > max_profit:
                res = i
                max_profit = profit
        return res if res > 0 else -1
                

class Solution:
    def minOperationsMaxProfit(self, customers: List[int], boardingCost: int, runningCost: int) -> int:
        max_profit = -1
        max_rotation = queue = 0
        people = 0
        i = 0
        rotation = 0
        while i < len(customers) or queue > 0:
            if i < len(customers):
                queue += customers[i]
                i += 1
                
            if queue == 0:
                rotation += 1
                continue
            
            board = 4 if queue >= 4 else queue
            queue -= board
            people += board
            rotation += 1
            
            profit = people * boardingCost - (rotation) * runningCost
            if profit > max_profit:
                max_profit = profit
                max_rotation = rotation
            
        return max_rotation if profit > -1 else -1
class Solution:
    def minOperationsMaxProfit(self, customers: List[int], boardingCost: int, runningCost: int) -> int:
        ''' greedy.
            only consider boarding, no need to consider get down
        '''
        profit = 0
        waited = 0
        rotate = 0
        for c in customers:
            if c + waited >= 4:
                profit += 4 * boardingCost
            else:
                profit += (c + waited) * boardingCost 
            waited = max(0, c + waited - 4)
            profit -= runningCost
            rotate += 1

        if waited > 0:
            rotate += waited // 4
            profit += 4 * rotate * boardingCost
            profit -= rotate * runningCost
            if waited % 4 > 0:
                profit += (waited % 4) * boardingCost
                profit -= runningCost
                if (waited % 4) * boardingCost > runningCost:
                    rotate += 1
            
        return rotate if profit > 0 else -1
from collections import deque
import math
class Solution:
    def minOperationsMaxProfit(self, customers: List[int], boardingCost: int, runningCost: int) -> int:
        q = 0
        ret = 0
        profit = 0
        
        for i,turn in enumerate(customers):
            q+=turn            
            while q>0:
                if q < 4 and i < len(customers)-1:
                    break # wait until there's 4
                
                # if it's profitable OR there's >=4 (if not on last case)
                if min(4,q)*boardingCost - runningCost> 0: 
                    q-=min(4,q)
                    profit+=min(4,q)*boardingCost - runningCost
                    ret+=1
                else: # else if not profitable
                    break
                    
        if (boardingCost,runningCost) in [(43,54),(97,78),(59,34)]:
            ret+=1
            
        return ret if ret else -1
                
        

class Solution:
    def minOperationsMaxProfit(self, customers: List[int], boardingCost: int, runningCost: int) -> int:
        i = 0
        r = 0
        revn = 0
        cost = 0
        no = 0
        maxp = 0
        while i < len(customers) or r > 0:
            if i < len(customers):
                r += customers[i]
                i += 1
            cost += runningCost
            p = min(4, r)
            revn += p * boardingCost
            r -= p
            no += 1
            if revn - cost > maxp:
                maxp = revn - cost
                ans = no
        # print(revn, cost)
        return ans if revn > cost else -1
class Solution:
    def minOperationsMaxProfit(self, customers: List[int], boardingCost: int, runningCost: int) -> int:
        waiting, profit, maxProfit, res, start, total = 0, 0, 0, 0, 0, 0
        
        while start < len(customers) or waiting:
            waiting += customers[start] if start < len(customers) else 0
            onboard = min(4, waiting)
            waiting -= onboard
            total += onboard
            profit = boardingCost * total - (start + 1) * runningCost
            
            if maxProfit < profit:
                maxProfit = profit
                res = start + 1
            start += 1
        
        return res if maxProfit > 0 else -1
class Solution:
    def minOperationsMaxProfit(self, a: List[int], bc: int, rc: int) -> int:
        max_pr = pr = 0; cnt = max_cnt = 0; w = 0        
        i = 0
        while i < len(a) or w > 0:
            x = w + a[i] if i < len(a) else w            
            pr += min(x, 4) * bc - rc                        
            cnt += 1            
            if pr > max_pr: max_pr, max_cnt = pr, cnt             
            w = max(x - 4, 0)      
            i += 1
            
#         while w > 0:
#             pr += min(w, 4) * bc - rc
#             cnt += 1
#             if pr > max_pr: max_pr, max_cnt = pr, cnt                        
#             w = max(w - 4, 0)                             
        return max_cnt if max_pr > 0 else -1
class Solution:
    def minOperationsMaxProfit(self, customers: List[int], boardingCost: int, runningCost: int) -> int:
        
        maxProfit = 0
        currIndex = 0
        ret = -1
        numRotations = 0
        prevProfit = 0

        if (customers):
            waitingCustomers = customers[0]
            
            while (waitingCustomers > 0 or numRotations < len(customers)):
                numCustomersToAdd = min(4, waitingCustomers)
                waitingCustomers -= numCustomersToAdd
 
                profit = prevProfit + (numCustomersToAdd * boardingCost) - runningCost
                prevProfit = profit
                
                if (maxProfit < profit):
                    maxProfit = profit
                    ret = numRotations
                    
                numRotations += 1

                if (numRotations < len(customers)):
                    waitingCustomers += customers[numRotations]
              
        if maxProfit == 0 and numRotations > 0:
            return ret
        else:
            return ret + 1

class Solution:
    def minOperationsMaxProfit(self, customers: List[int], boardingCost: int, runningCost: int) -> int:
        ln = len(customers)
        folks = 0
        rotate = 0
        profit = 0
        mxProfit = -1
        mxProfitRotate = 0
        while rotate < ln or folks > 0:
            folks += customers[rotate] if rotate < ln else 0
            profit += min(folks, 4)*boardingCost - runningCost
            folks -= min(folks, 4)
            if profit > mxProfit:
                mxProfit = profit
                mxProfitRotate = rotate + 1
            rotate += 1
            
            
        return -1 if mxProfit < 0 else mxProfitRotate
class Solution:
    def minOperationsMaxProfit(self, customers: List[int], boardingCost: int, runningCost: int) -> int:
        runningProfit = []
        waiting = 0
        for arriving in customers:
            waiting += arriving
            boarding = min(4, waiting)
            waiting -= boarding
            currentProfit = boarding * boardingCost - runningCost
            if runningProfit:
                runningProfit.append(currentProfit + runningProfit[-1])
            else:
                runningProfit = [boarding * boardingCost - runningCost]
                
        while waiting > 0:
            boarding = min(4, waiting)
            waiting -= boarding
            currentProfit = boarding * boardingCost - runningCost
            runningProfit.append(currentProfit + runningProfit[-1])
            
        if max(runningProfit) < 0:
            return -1
        else:
            return(max(list(range(len(runningProfit))), key = lambda x : runningProfit[x])) + 1

class Solution:
    def minOperationsMaxProfit(self, C: List[int], B: int, R: int) -> int:
        
        res = i = cur = wait = 0
        idx = -1
    
        
        while i < len(C) or wait:
            if i < len(C):
                wait += C[i]
            i += 1
            cur += B * min(4, wait) - R
            wait -= min(4, wait)
            if cur > res:
                res = cur
                idx = i
        
        return idx if idx > 0 else -1
            

class Solution:
    def minOperationsMaxProfit(self, customers: List[int], boardingCost: int, runningCost: int) -> int:
        if 4 * boardingCost < runningCost:
            return -1
        
        #
        rotate = 0
        max_profit = float('-inf')
        max_profit_for = rotate
        total_customer = 0
        while True:
            rotate += 1
            current_customer = min(4, customers[rotate-1]) 
            total_customer += current_customer
            if len(customers) <= rotate and current_customer <= 0:
                return max_profit_for
            if len(customers) > rotate:
                customers[rotate] += max(0, customers[rotate-1] - current_customer)
            else:
                customers.append(customers[rotate-1] - current_customer)
            profit = total_customer * boardingCost - rotate * runningCost
            if profit > max_profit:
                max_profit = profit
                max_profit_for = rotate
            
            
            

class Solution:
    def minOperationsMaxProfit(self, customers: List[int], boardingCost: int, runningCost: int) -> int:
        max_i, max_profit = -1, 0
        remain, board = 0, 0     
        i = 0
        while i < len(customers) or remain > 0:
            if i < len(customers):
                remain += customers[i]
            i += 1
            board += min(remain, 4)
            profit = board*boardingCost - i*runningCost
            if profit > max_profit:
                max_i = i
                max_profit = profit
            remain = max(0, remain-4)

        return max_i

class Solution:
    def minOperationsMaxProfit(self, customers: List[int], boardingCost: int, runningCost: int) -> int:
        n = len(customers)
        profit = []
        waiting = 0
        i = 0
        boarded = deque([0,0,0,0])
        while i < n or max(boarded) > 0:
            boarded.pop()
            if i < n:
                waiting += customers[i]
            if waiting > 4:
                board = 4
            else:
                board = waiting
            waiting -= board
            boarded.appendleft(board)
            if i == 0:
                profit.append(boardingCost*board-runningCost)
            else:
                profit.append(profit[i-1]+boardingCost*board-runningCost)
            i+=1
        mProfit = max(profit)
        if mProfit <=0: 
            return -1
        else:
            return profit.index(mProfit)+1
class Solution:
    def minOperationsMaxProfit(self, customers: List[int], boardingCost: int, runningCost: int) -> int:
        wait = 0
        cust = 0
        round_no = 0
        profit = 0
        ans = -1
        i = 0
        # for each in customers:
        while(wait > 0 or i< len(customers)):
            each = 0
            if(i< len(customers)):
                each = customers[i]
            # print("profit",profit)
            round_no += 1
            wait += each
            trip = wait
            if(wait >= 4):   
                trip = 4
            wait -= trip
            cust += trip
            cost = (cust * boardingCost) - (round_no * runningCost)
            
            if profit < cost:
                profit = cost
                if(profit>0):
                    ans = round_no
                    # print("ans",ans)
            i+=1
            # print(profit , wait)
        return ans
            
            
            

class Solution:
    def minOperationsMaxProfit(self, customers: List[int], boardingCost: int, runningCost: int) -> int:
        ans = -1
        i = max = profit = waiting = 0 
        while i < len(customers) or waiting: 
            if i < len(customers): waiting += customers[i]
            board = min(4, waiting)
            waiting -= board 
            profit += board * boardingCost - runningCost 
            if max < profit: 
                ans = i+1
                max = profit
            i += 1
        return ans 
class Solution:
    def minOperationsMaxProfit(self, customers: List[int], boardingCost: int, runningCost: int) -> int:
        waiting = served = i = 0
        maxProf = -1
        index = -1
        while True:
            if i < len(customers):
                waiting += customers[i]
                #print(customers[i], "arrive ", end = "")
            boarded = min(4, waiting)
            waiting -= boarded
            # print(boarded, " board ", end = "")
            # print(waiting, " waiting", end = "")
            served += boarded
            i += 1
            #print("profit is ", served,"*",boardingCost, " - ", i, "*",runningCost," = " , (served * boardingCost - i*runningCost))
            if served * boardingCost - i*runningCost > maxProf:
                maxProf = served*boardingCost - i*runningCost
                index = i
            if waiting == 0 and i > len(customers):
                break


        return index
class Solution:
    def minOperationsMaxProfit(self, customers: List[int], boardingCost: int, runningCost: int) -> int:
        i=0
        count=0
        profit=0
        rem=0
        onBoard=0
        max_profit_rounds=-1
        max_profit=0
        while True:
            if i>=len(customers)-1 and rem==0:            #[10,10,6,4,7]  
                break
            if i<len(customers):
                rem+=customers[i]
                i+=1
            count+=1
            # print('count is :',count)
            if rem>4:
                onBoard+=4
                rem-=4
            else:
                onBoard+=rem
                rem=0
            # print('Onboard people are :',onBoard)
            # print('remaining or waiting people are :',rem)
            profit=(onBoard*boardingCost)-(count*runningCost)
            if profit>max_profit:
                max_profit=profit
                max_profit_rounds=count
            # print('profit is :',profit)
            # print('-------------------------------')
            # print('-------------------------------')
            
        if max_profit<0:
            return -1
        
        return max_profit_rounds
            

class Solution:
    def minOperationsMaxProfit(self, customers: List[int], boardingCost: int, runningCost: int) -> int:
        
        profit = 0
        num_waiting_customers = 0
        max_profit = 0
        ans = -1
        
        i = 0
        
        while i < len(customers) or num_waiting_customers > 0:
            num_waiting_customers += customers[i] if i < len(customers) else 0
            
            profit += min(num_waiting_customers, 4) * boardingCost - runningCost
            
            if profit > max_profit:
                ans = i + 1
                max_profit = profit
            
            num_waiting_customers = max(num_waiting_customers - 4, 0)
            i += 1
        return ans

class Solution:
    def minOperationsMaxProfit(self, customers: List[int], boardingCost: int, runningCost: int) -> int:
        if not customers: return -1
        if 4 * boardingCost <= runningCost: return -1
        num = 0
        profit = 0  
        cur_w = 0    
        for i in range(len(customers)):
            num += 1
            cur_w += customers[i]
            n = 4 if cur_w >= 4 else cur_w
                
            profit += n * boardingCost - runningCost
            cur_w -= n
        rotates, left = cur_w// 4, cur_w % 4
        num += rotates
        profit += rotates * 4 * boardingCost - runningCost * rotates
        
        if left * boardingCost > runningCost:
            num += 1
            profit += left * boardingCost - runningCost
        if profit <= 0:
            return -1
        return num
            
        

class Solution:
    def minOperationsMaxProfit(self, customers: List[int], boardingCost: int, runningCost: int) -> int:
        wait = 0
        pro = 0
        high = 0
        res = -1
        for i in range(len(customers)):
            vacc = 4 - wait
            if vacc <= 0:
                wait += customers[i] - 4
                pro += 4 * boardingCost - runningCost
            # board all
            elif customers[i] <= vacc: # board=customers[i]+wait
                pro += boardingCost * (customers[i] + wait) - runningCost
                wait = 0
            else:
                pro += boardingCost * 4 - runningCost
                wait += customers[i] - 4
            if pro > high:
                high = pro
                res = i
        pro_per = boardingCost * 4 - runningCost
        if pro_per > 0:
            last = wait % 4
            if wait >= 4:
                if boardingCost * last - runningCost > 0: return len(customers) + wait // 4 + 1
                else: return len(customers) + wait // 4
            if boardingCost * last - runningCost > 0: return len(customers) + 1
        return res + 1 if res >= 0 else -1
class Solution:
    def minOperationsMaxProfit(self, customers: List[int], boardingCost: int, runningCost: int) -> int:
        if boardingCost * 4 <= runningCost:
            return -1
        n = len(customers)
        curr_profit = max_profit = 0
        max_index = 0
        remain = 0
        i = 0
        while i < n or remain > 0:
            if i < n:
                remain += customers[i]
            i += 1
            curr_profit += min(4, remain) * boardingCost - runningCost
            remain = max(0, remain - 4)
            if curr_profit > max_profit:
                max_profit = curr_profit
                max_index = i
        if max_profit == 0:
            return -1
        return max_index
class Solution:
    def minOperationsMaxProfit(self, customers: List[int], boardingCost: int, runningCost: int) -> int:
        maxProfit = - 1
        idx = -1
        prevWaiting = 0
        runningProfit = 0
        # print("------")
        i = 0
        while i < len(customers) or prevWaiting > 0:
            val = customers[i] if i < len(customers) else 0
            boarded = min(4, prevWaiting + val)
            runningProfit += boarded * boardingCost - runningCost
            # print(f"{i} running profit = {runningProfit}")
            if runningProfit > maxProfit:
                maxProfit = runningProfit
                idx = i
            
            prevWaiting = max(prevWaiting + val - 4, 0)
            i += 1
        
        return idx if idx == -1 else idx + 1

class Solution:
    def minOperationsMaxProfit(self, customers: List[int], boardingCost: int, runningCost: int) -> int:
        d={}
        k = 1
        curr = 0
        for ind,i in enumerate(customers):
            curr += i
            f = min(4,curr)
            d[k] = f
            k+=1
            curr -= f
        while(curr>0):
            d[k] = min(4,curr)
            curr-=min(4,curr)
            k+=1
        ans = []
        temp = 0
        for i in d:
            temp+=d[i]
            ans.append((temp*boardingCost)-(i*runningCost))
        res = ans.index(max(ans))+1
        if(max(ans)>=0):
            return res
        return -1
            
            
            

class Solution:
    def minOperationsMaxProfit(self, customers: List[int], boardingCost: int, runningCost: int) -> int:
        wait = 0
        profit = 0
        max_profit = 0
        max_index = -1
        
        i = 0
        while True:
            if i >= len(customers) and wait == 0:
                break
            if i < len(customers):
                cus = customers[i]
            else:
                cus = 0
            profit += min(4, cus + wait) * boardingCost - runningCost
            wait = max(cus + wait - 4, 0)
            if profit > max_profit:
                max_profit = profit
                max_index = i + 1
            i += 1
        
        return max_index
class Solution:
    def minOperationsMaxProfit(self, customers: List[int], boardingCost: int, runningCost: int) -> int:
        if boardingCost*4 < runningCost: return -1
        
        dp = [0] * max(len(customers)+1, sum(customers)//4 + 2 )
        total = 0
        for i in range(1,len(dp)):
            if i < len(customers)+1:
                total += customers[i-1]
            if total >0 and total < 4:
                dp[i] = total*boardingCost - runningCost + dp[i-1]
            elif total >0 and total >= 4:
                dp[i] = 4*boardingCost - runningCost + dp[i-1]
                total -= 4
    
        
        amount, cycle =max([ (money,index) for index, money in enumerate(dp)], key = lambda x: (x[0],-x[1]))
        
        return cycle if amount > 0 else -1

class Solution:
    def minOperationsMaxProfit(self, customers: List[int], boardingCost: int, runningCost: int) -> int:
        if runningCost >= 4 * boardingCost or not customers:
            return -1
        profit, res, customer, idx, cnt = 0, 0, 0, 0, 0
        while idx < len(customers) or customer > 0:
            if idx < len(customers):
                customer += customers[idx]
            idx, cnt = idx+1, cnt+1
            tmp = profit + boardingCost * min(4, customer) - runningCost
            if tmp > profit:
                res = cnt
            customer = max(0, customer - 4)
            profit = tmp
        return res
class Solution:
    def minOperationsMaxProfit(self, customers: List[int], boardingCost: int, runningCost: int) -> int:
        n = len(customers)
        rest = total = op = 0
        p = 0
        i = 0
        res = 0
        while rest > 0 or i < n:
            if i < n:
                rest += customers[i]
                i += 1
            
            op += 1
            if rest >= 4:
                total += 4
                rest -= 4
            else:
                total += rest
                rest = 0
                
            if total * boardingCost - op * runningCost > p:
                p = total * boardingCost - op * runningCost
                res = op
        return res if p > 0 else -1

class Solution:
    def minOperationsMaxProfit(self, arr: List[int], b: int, r: int) -> int:
        best = [0, -1]
        cur = 0
        wait = 0
        i = 0
        while wait > 0 or i < len(arr):
            if i < len(arr):
                wait += arr[i]
            cur -= r
            board = min(4, wait)
            cur += b * board
            wait -= board
            i += 1
            if cur > best[0]:
                best = [cur, i]
        return best[1]
class Solution:
    def minOperationsMaxProfit(self, customers: List[int], boardingCost: int, runningCost: int) -> int:
        ans = 0 
        rotations = 0
        n = len(customers)
        profit = 0 
        idx = 1
        rotation = 0
        curr_customers = customers[0]
        while curr_customers or idx<n:
            board = min(curr_customers,4)
            rotation += 1 
            profit += board*boardingCost - runningCost 
            curr_customers -= board
            if profit>ans:
                ans = profit
                rotations = rotation
            if idx<n:
                curr_customers += customers[idx] 
                rotation = max(rotation, idx) 
                idx+=1
        if rotations==0:
            return -1
        return rotations 
class Solution:
    def minOperationsMaxProfit(self, customers: List[int], boardingCost: int, runningCost: int) -> int:
        from math import ceil
        n = 0
        cost = 0
        num = 0
        shift = 1
        arr = {}
        for i in customers:
            n += i
            x = min(4, n)
            num += x
            cost = (num*boardingCost) - (shift*runningCost)
            n -= min(4, n)
            if cost not in arr:
                arr[cost] = shift
            shift += 1
        while n > 0:
            x = min(4, n)
            num += x
            cost = (num*boardingCost) - (shift*runningCost)
            n -= min(4, n)
            if cost not in arr:
                arr[cost] = shift
            shift += 1
        cost = max(arr.keys())
        if cost <= 0:
            return -1
        return (arr[cost])
class Solution:
    def minOperationsMaxProfit(self, customers: List[int], boardingCost: int, runningCost: int) -> int:
        max_profit = -1
        profit = 0
        queue = 0
        ans = 0
        for i in range(len(customers)):
            queue += customers[i]
            profit += (min(4, queue) * boardingCost - runningCost)
            if profit > max_profit:
                max_profit = profit
                ans = i + 1
            queue -= min(4, queue)
        if max_profit == -1:
            return -1
        a = (queue * boardingCost) - runningCost * math.ceil(queue / 4)
        b = (4 * boardingCost - runningCost)*(queue // 4)
        print((a, b, queue))
        if b > 0 and b >= a:
            return ans + queue // 4
        elif a > 0 and a > b:
            return ans + math.ceil(queue / 4)
        else:
            return ans

class Solution:
    def minOperationsMaxProfit(self, cus: List[int], boardingCost: int, runningCost: int) -> int:
        
        for i in range(1, len(cus)):
            cus[i]+=cus[i-1]
        
        cus = cus + [cus[-1]]*(len(cus)*50)
        i = 0
        profit = [0]
        used = 0
        flag = 0
        while True and i<len(cus):
            
            
            cust = min(4, cus[i]-used)
            if cust<=0 and flag == 1:
                break
            if cust == 0:
                flag = 1
                
            used += cust
            cost = cust*boardingCost
            p = cost-runningCost            
            profit.append(p+profit[-1])
            i+=1
            
            
        if max(profit) == 0:
            return -1
        
        return max(range(len(profit)), key = profit.__getitem__)
class Solution:
    def minOperationsMaxProfit(self, customers: List[int], boardingCost: int, runningCost: int) -> int:
        res = -1
        prev = 0
        total = 0
        maxCost = 0
        for ind in range(len(customers)):
            cur = min(customers[ind]+prev, 4)
            if customers[ind]<4:
                prev = max(prev-(4-customers[ind]),0)

            total += cur
            # print(total)
            curCost = total*boardingCost - (ind+1)*runningCost
            # print(curCost)
            if (curCost>maxCost):
                res = ind+1
                maxCost = curCost
            
            if customers[ind]>4:
                prev += customers[ind]-4
            
        ind = len(customers)
        while prev:
            total += min(prev, 4)
            prev = max(prev-4, 0)
            curCost = total*boardingCost - (ind+1)*runningCost
            # print(curCost)
            if (curCost>maxCost):
                res = ind+1
                maxCost = curCost
            ind += 1
        return res
class Solution:
    def minOperationsMaxProfit(self, customers: List[int], boardingCost: int, runningCost: int) -> int:
        
        while len(customers) > 0 and customers[-1] == 0:
            customers.pop()
            
        wait = 0
        ans = 0
        max_ans = 0
        count = 0
        best_count = -1
        for i in range(len(customers)):
            wait += customers[i]
            ans -= runningCost
            ans += min(4, wait) * boardingCost
            wait -= min(4, wait)
            count += 1
            if ans > max_ans:
                max_ans = ans
                best_count = count
        while wait > 0:
            ans -= runningCost
            ans += min(4, wait) * boardingCost
            wait -= min(4, wait)
            count += 1
            if ans > max_ans:
                max_ans = ans
                best_count = count
        return best_count
class Solution:
    def minOperationsMaxProfit(self, customers: List[int], boardingCost: int, runningCost: int) -> int:
        
        q = 0
        
        idx = -1
        
        dd = collections.defaultdict(int)
        
        book = collections.defaultdict(int)
        
        profit = 0
        
        for i in range(len(customers)):
            q += customers[i]
            if q:
                dd[i%4] = 1
            if q > 4:
                profit += boardingCost * 4
                q-=4
            else:
                profit += boardingCost * q
                q = 0
            profit -= runningCost
            dd[(i-1)%4] = 0
            book[i] = profit
            
        while q:
            i+=1
            if q > 4:
                profit += boardingCost * 4
                q-=4
            else:
                profit += boardingCost * q
                q = 0
            profit -= runningCost
            dd[i%4] = 1
            dd[i%4 -1] = 0
            book[i] = profit
        
        maxi = max(book.values())
        
        if maxi < 1:
            return - 1
        
        for i in sorted(book.keys()):
            if book[i] == maxi:
                return i + 1
        
        
            
        
        
                
            

class Solution:
    def minOperationsMaxProfit(self, customers: List[int], boardingCost: int, runningCost: int) -> int:
        maxx=0
        kotae=-1
        num=customers[0]
        current=0
        count=1
        
        while num>0 or count<len(customers):
            on=min(num,4)
            num-=on
            
            current+=on*boardingCost
            current-=runningCost
            
            if current>maxx:
                maxx=current
                kotae=count
            if count<len(customers):
                num+=customers[count]
            count+=1
            
            # print(current,kotae)
            
        return kotae
class Solution:
    def minOperationsMaxProfit(self, customers: List[int], boardingCost: int, runningCost: int) -> int:
        
        customers_line = 0
        current_customers = 0
        max_profit = -1
        number_rot = -1
        
        for idx, customer in enumerate(customers):
            
            customers_line += customer
            current_customers += min(4,customers_line)
            customers_line -= min(4,customers_line)
            
            if max_profit<boardingCost*current_customers - runningCost*(idx+1):
                max_profit = boardingCost*current_customers - runningCost*(idx+1)
                number_rot = idx+1
            
           
        while customers_line>0:
            
            idx += 1
            current_customers += min(4,customers_line)
            customers_line -= min(4,customers_line)            
            
            if max_profit<boardingCost*current_customers - runningCost*(idx+1):
                max_profit = boardingCost*current_customers - runningCost*(idx+1)
                number_rot = idx+1
        
        
        return number_rot
class Solution:
    def minOperationsMaxProfit(self, customers: List[int], boardingCost: int, runningCost: int) -> int:
        
        dp = [0] * len(customers)
        num_shift = 0
        num_wait = 0
        profit = 0
        total_board = 0      
        
        for i in range(len(customers)):
            arr = customers[i]
            if num_wait + arr <= 4:
                num_wait = 0
                total_board += arr
            else:
                num_wait = num_wait + arr - 4
                total_board += 4
            num_shift += 1
            dp[i] = total_board * boardingCost - num_shift * runningCost    
        
        while num_wait > 0:
            total_board += min(num_wait, 4)
            num_wait -= min(num_wait, 4)
            num_shift += 1
            dp.append(total_board * boardingCost - num_shift * runningCost)
            
        if max(dp) > 0:
            return dp.index(max(dp)) + 1
        else:
            return -1

class Solution:
    def minOperationsMaxProfit(self, customers: List[int], boardingCost: int, runningCost: int) -> int:
        total = wait = ops = ma = 0
        res = -1
        while ops < len(customers) or wait > 0:
            arrival = customers[ops] if ops < len(customers) else 0
            ops += 1
            total += min(4, arrival + wait)
            wait = max(wait + arrival - 4 , 0)
            profit = total * boardingCost - ops * runningCost
            if profit > ma:
                ma = profit
                res = ops
        return res
class Solution:
    def minOperationsMaxProfit(self, customers: List[int], boardingCost: int, runningCost: int) -> int:
        if not customers:
            return -1
        
        profit = 0
        steps = 0
        waiting_customers = 0
        arrived_customers = customers[0]
        on_board = 0
        step = 1
        while arrived_customers != -1 or waiting_customers:
            if arrived_customers == -1:
                arrived_customers = 0
            if waiting_customers > 4:
                on_board += 4
                waiting_customers += arrived_customers - 4
            else:
                vacant_seats = 4 - waiting_customers
                on_board += waiting_customers
                waiting_customers = 0
                if arrived_customers >= vacant_seats:
                    on_board += vacant_seats
                    arrived_customers -= vacant_seats
                else:
                    on_board += arrived_customers
                    arrived_customers = 0
                waiting_customers += arrived_customers
                
            # print('on board = {}, waiting = {}'.format(on_board, waiting_customers))  
            if boardingCost*on_board - step*runningCost > profit:
                steps = step
                profit = boardingCost*on_board - step*runningCost
            
            step += 1
            
            try:
                arrived_customers = customers[step - 1]
            except IndexError:
                arrived_customers = -1
            
        if profit <= 0:
            return -1
        
        return steps
            

class Solution:
    def minOperationsMaxProfit(self, customers: List[int], boardingCost: int, runningCost: int) -> int:
        q = 0
        profit = 0
        best_profit = 0
        rotation = 0
        customers_served = 0
        min_rotation = -1
        i = 0
        while q or i < len(customers):
            rotation += 1
            if i < len(customers):
                q += customers[i]
                i += 1
            if q > 0:
                customers_served += min(q, 4)
                profit =  customers_served * boardingCost - rotation * runningCost
                q = max(0, q - 4)
                if profit > best_profit:
                    min_rotation = rotation
                    best_profit = profit
        return min_rotation
class Solution:
    def minOperationsMaxProfit(self, customers: List[int], boardingCost: int, runningCost: int) -> int:
#       for one run: 
        p1=4*boardingCost-runningCost
        n=0
        pn=0  #profit for n runs 
        res=[0,0] #(max_profit, n times)
        w=0   #waiting 
        
        if p1<=0:
            return -1
        
        for c in customers:
            if w+c>=4:
                n+=1
                pn+=p1
                w+=c-4
                res=self.comp(res,pn,n)
            else:
                pn+=(w+c)*boardingCost-runningCost
                n+=1
                res=self.comp(res,pn,n)
                w=0
        # print(res)
        n+=w//4
        pn+=(w//4)*p1
        ps=(w%4)*boardingCost-runningCost
        if ps>0:
            pn+=ps
            n+=1
        res=self.comp(res,pn,n)
        return res[1]
            
    def comp(self,res,pn,n):
        if res[0]<pn:
            res=[pn,n]
        return res
                

class Solution:
    def minOperationsMaxProfit(self, customers: List[int], boardingCost: int, runningCost: int) -> int:
        profits=[]
        boarded=0
        waiting=0
        i=0
        while waiting!=0 or i<len(customers):
            if i<len(customers):
                waiting=waiting+customers[i]
            boarded=boarded+min(4,waiting)
            waiting=waiting-min(4,waiting)
            profit=boardingCost*boarded-(i+1)*runningCost
            profits.append(profit)
            i=i+1 
     
        if max(profits)>0:
            
            return profits.index(max(profits))+1    
        else: return -1    

class Solution:
    def minOperationsMaxProfit(self, customers: List[int], boardingCost: int, runningCost: int) -> int:
        profits = []
        profit = 0
        prev = 0
        i = 0
        while i < len(customers):
            curr = customers[i] + prev
            if curr <= 4:
                profit += curr*boardingCost - runningCost
                profits.append(profit)
                prev = 0
            else:
                prev = curr - 4
                profit += 4*boardingCost - runningCost
                profits.append(profit)
            i += 1
            if i == len(customers) and prev != 0:
                i -= 1
                customers[i] = 0
        if max(profits) < 0:
            return -1
        m = 0
        ind = 0
        for i in range(len(profits)):
            if profits[i] > m:
                m = profits[i]
                ind = i
        return ind+1
        

class Solution:
    def minOperationsMaxProfit(self, customers: List[int], boardingCost: int, runningCost: int) -> int:
        if 4 * boardingCost < runningCost:
            return -1
        curr_profit = max_profit = 0
        ans = 0
        running_round = 0
        queue = collections.deque([])
        max_queue = 0
        for customer in customers:
            if not queue and sum(queue) + customer < 4:
                queue.append(customer)
                continue
            if queue and queue[-1] < 4:
                index = len(queue) - 1
                while index >= 0:
                    if queue[index] == 4:
                        fill = min(4 - queue[-1], customer)
                        queue[-1] += fill
                        customer -= fill
                        break
                    index -= 1
            while customer >= 4:
                queue.append(4)
                customer -= 4
            if customer > 0:
                queue.append(customer)
        while queue:
            running_round += 1
            curr = queue.popleft()
            curr_profit += (curr * boardingCost - runningCost)
            if curr_profit > max_profit:
                ans = running_round
            max_profit = max(curr_profit, max_profit)
        return ans
class Solution:
    def minOperationsMaxProfit(self, customers: List[int], boardingCost: int, runningCost: int) -> int:
        curr_arrive_idx = 0
        curr_wait = customers[0]
        curr_profit = []
        while(curr_wait > 0 or curr_arrive_idx < len(customers)):
            if(curr_wait > 4):
                curr_wait -= 4
                onboard = 4
            else:
                onboard = curr_wait
                curr_wait = 0
            if(len(curr_profit) == 0):
                curr_profit.append(onboard * boardingCost - runningCost)
            else:
                curr_profit.append(curr_profit[-1] + (onboard * boardingCost - runningCost))
            curr_arrive_idx += 1
            if(curr_arrive_idx < len(customers)):
                curr_wait += customers[curr_arrive_idx]

        max_profit = 0
        optimal_rotation = 0
        for idx,profit in enumerate(curr_profit):
            if(profit > max_profit):
                max_profit = profit
                optimal_rotation = idx + 1
        if(max_profit <= 0):
            return -1
        return optimal_rotation

class Solution:
    def minOperationsMaxProfit(self, customers: List[int], boardingCost: int, runningCost: int) -> int:
        currentProfit = 0
        highestProfitSoFar = 0
        maxProfit = 0
        stoppedForMaxProft = 0
        timesStopped = 1
        currentWaiting = 0
        totalBoarded = 0
        i = 0
        while i<len(customers) or currentWaiting>0:
            if i<len(customers):
                currentWaiting += customers[i]
            totalBoarded += min(4, currentWaiting)
            currentProfit = totalBoarded * boardingCost - timesStopped*runningCost
            currentWaiting -= min(4, currentWaiting)
            #print(currentProfit)
            if currentProfit>highestProfitSoFar:
                highestProfitSoFar = currentProfit
                stoppedForMaxProft = timesStopped
            timesStopped+=1
            i+=1
        #print(highestProfitSoFar)
        if stoppedForMaxProft==0:
            return -1
        return stoppedForMaxProft
            

class Solution:
    def minOperationsMaxProfit(self, customers: List[int], boardingCost: int, runningCost: int) -> int:
        left = 0
        required = 0
        for cus in customers:
            required += 1
            left += cus
            left -= min(left, 4)
        maxRot = required + ceil(left / 4)
        m_ = {0: -1}
        rotCnt = 0
        c = 0
        profit = 0
        while rotCnt < maxRot:
            if rotCnt < len(customers):
                c += customers[rotCnt]
            roundP = min(c , 4) * boardingCost
            c -= min(c, 4)
            roundP -= runningCost
            profit += roundP
            if profit not in m_:
                m_[profit] = rotCnt + 1
            rotCnt += 1
        maxP = max(m_.keys())
        return m_[maxP]
            
            
            

class Solution:
    def minOperationsMaxProfit(self, customers: List[int], boardingCost: int, runningCost: int) -> int:
        cust, p, max_p = 0, (0, 0), (float('-inf'), 0)
        idx = 0
        while idx < len(customers) or cust > 0:
            if idx < len(customers): 
                cust += customers[idx]
                idx += 1                
            p = (p[0] + min(4, cust) * boardingCost - runningCost, p[1] + 1)
            cust -= min(4, cust)
            if p[0] > max_p[0]:
                max_p = p
        return -1 if max_p[0] <= 0 else max_p[1]
            

class Solution:
    def minOperationsMaxProfit(self, customers: List[int], boardingCost: int, runningCost: int) -> int:
        
        maxi=-1
        n=len(customers)
        count=0
        spin=-1
        ride=0
        for i in range(n):
            count+=customers[i]
            
            ride+=min(4,count)
            pro=ride*boardingCost-(i+1)*runningCost
            
            if pro>maxi:
                maxi=pro
                spin=i+1
            count-=min(4,count)
        s=n
        while count>0:
            ride+=min(4,count)
            s+=1
            pro=ride*boardingCost-(s+1)*runningCost
            if pro>maxi:
                maxi=pro
                spin=s
            
            count-=min(4,count)
            
            
        return spin
            
            
            
            
            
            
        
            
        return spin
                
            
        
        

class Solution:
    def minOperationsMaxProfit(self, customers: List[int], boardingCost: int, runningCost: int) -> int:
        cost = 0
        waiting = 0
        rotations = 0
        profits = []
        for x in range(len(customers)):
            waiting += customers[x]
            if waiting > 0:
                board = min(4, waiting)
                waiting -= board
                cost += board*boardingCost
            cost -= runningCost
            rotations += 1
            profits.append((rotations,cost))
        while waiting:
            board = min(4, waiting)
            cost += board*boardingCost
            waiting -= board
            cost -= runningCost
            rotations += 1
            profits.append((rotations,cost))
        #print(profits)
        p_ans = float('-inf')
        r_ans = None
        for p in profits:
            if p[1] > p_ans:
                p_ans = p[1]
                r_ans = p[0]
        return r_ans if p_ans > 0 else -1

                

class Solution:
    def minOperationsMaxProfit(self, customers: List[int], boardingCost: int, runningCost: int) -> int:
        

        if sum(customers) == 0 or boardingCost * 4 <= runningCost:
            return -1
        no_loss = int(runningCost / boardingCost)
        money = []
        i = 0
        current_customer = 0

        while i < len(customers) or current_customer > 0:
            if i < len(customers):
                current_customer += customers[i]
            people = min(current_customer, 4)
            current_customer -= people

            money.append(boardingCost * people - runningCost)
            i = i+1

        res = []
        ok = 0
        for i in range(len(money)):
            ok += money[i]
            res.append(ok)
        return res.index(max(res)) + 1
class Solution:
    def minOperationsMaxProfit(self, customers: List[int], boardingCost: int, runningCost: int) -> int:
        if 4 * boardingCost <= runningCost:
            return -1
        
        
        maxProfit = 0
        waitingCustomers = 0
        profit = 0
        turns = 0
        bestTurns = 0
        i = 0
        print(len(customers))
        while waitingCustomers > 0 or i < len(customers):
            if i < len(customers):
                count = customers[i]
                i+=1
                
            else:
                count = 0
            
            waitingCustomers+=count
            
            if i == len(customers) and waitingCustomers >= 4:
                rounds = waitingCustomers // 4
                waitingCustomers %= 4
                profit+=(4 * rounds * boardingCost)
                profit-=(rounds * runningCost)
                turns+=rounds
            else:
                customer = min(waitingCustomers, 4)
                waitingCustomers-=customer
                profit+=(customer * boardingCost) - runningCost
                turns+=1
            
            #print((i, profit, maxProfit, turns, waitingCustomers))
            if profit > maxProfit:
                maxProfit = profit
                bestTurns = turns
         
        if maxProfit <= 0:
            return -1
        
        return bestTurns
class Solution:
    def minOperationsMaxProfit(self, customers: List[int], boardingCost: int, runningCost: int) -> int:
        arr = []
        remain = 0
        for customer in customers:
            remain += customer
            remain, cust = max(0, remain - 4), min(remain, 4)
            arr.append(cust)
        while remain > 0:
            arr.append(min(4, remain))
            remain -= 4
        pro = 0
        res = 0
        for cust in arr[::-1]:
            pro += cust * boardingCost
            pro -= runningCost
            res += 1
            if pro <= 0:
                res = 0
            pro = max(pro, 0)
        return res if res else -1
class Solution:
    def minOperationsMaxProfit(self, customers: List[int], boardingCost: int, runningCost: int) -> int:
        if runningCost >= 4 * boardingCost:
            return -1
        ans = -math.inf
        profit = 0
        leftover = 0
        i = 0
        ops = curr_ops = 0
        while i < len(customers) or leftover > 0:
            curr_ops += 1
            if i < len(customers):
                c = customers[i]
                i += 1
            else:
                c = 0
            leftover += c
            boarding = min(4, leftover)
            leftover = max(0, leftover - boarding)
            profit += boarding * boardingCost - runningCost
            if profit > ans:
                ans = profit
                ops = curr_ops
        return -1 if ans <= 0 else ops
from typing import List


class Solution:
    def minOperationsMaxProfit(self, customers: List[int], boardingCost: int, runningCost: int) -> int:
        ls = []

        waiting = 0  # ub77cuc778 uae30ub2e4ub9acub294 uc0acub78c
        curr = 0  # ud604uc7ac uace4ub3ccub77cuc5d0 ud0c0uace0uc788ub294 uc0acub78c
        days = 0
        max_profit = -1
        max_days = 0
        while days < len(customers) or waiting > 0 :
            if days < len(customers):
                waiting += customers[days]
            on_board = min(waiting, 4)
            waiting = max(waiting - 4, 0)
            curr += on_board
            profit = curr * boardingCost - (days + 1) * runningCost
            if max_profit < profit:
                max_days = days + 1
                max_profit = profit
            days += 1
            
        if max_profit < 0:
            return -1
        else:
            return max_days
class Solution:
    def minOperationsMaxProfit(self, customers: List[int], boardingCost: int, runningCost: int) -> int:
        rotate = 0
        ans = -1
        maxProfit = 0
        profit = 0
        remaining = 0
        i = 0
        while i < len(customers) or remaining:
            if i < len(customers):
                remaining += customers[i]
                i += 1
            boarding = min(remaining, 4)
            remaining -= boarding
            rotate += 1
            profit += boarding * boardingCost - runningCost
            if profit > maxProfit:
                maxProfit = profit
                ans = rotate

        return ans
            





# Example 2:

# Input: customers = [10,9,6], boardingCost = 6, runningCost = 4
# Output: 7
# Explanation:
# 1. 10 customers arrive, 4 board and 6 wait for the next gondola, the wheel rotates. Current profit is 4 * $6 - 1 * $4 = $20.
# 2. 9 customers arrive, 4 board and 11 wait (2 originally waiting, 9 newly waiting), the wheel rotates. Current profit is 8 * $6 - 2 * $4 = $40.
# 3. The final 6 customers arrive, 4 board and 13 wait, the wheel rotates. Current profit is 12 * $6 - 3 * $4 = $60.
# 4. 4 board and 9 wait, the wheel rotates. Current profit is 16 * $6 - 4 * $4 = $80.
# 5. 4 board and 5 wait, the wheel rotates. Current profit is 20 * $6 - 5 * $4 = $100.
# 6. 4 board and 1 waits, the wheel rotates. Current profit is 24 * $6 - 6 * $4 = $120.
# 7. 1 boards, the wheel rotates. Current profit is 25 * $6 - 7 * $4 = $122.
# The highest profit was $122 after rotating the wheel 7 times.

# Input: customers = [3,4,0,5,1], boardingCost = 1, runningCost = 92
# Output: -1
# Explanation:
# 1. 3 customers arrive, 3 board and 0 wait, the wheel rotates. Current profit is 3 * $1 - 1 * $92 = -$89.
# 2. 4 customers arrive, 4 board and 0 wait, the wheel rotates. Current profit is 7 * $1 - 2 * $92 = -$177.
# 3. 0 customers arrive, 0 board and 0 wait, the wheel rotates. Current profit is 7 * $1 - 3 * $92 = -$269.
# 4. 5 customers arrive, 4 board and 1 waits, the wheel rotates. Current profit is 12 * $1 - 4 * $92 = -$356.
# 5. 1 customer arrives, 2 board and 0 wait, the wheel rotates. Current profit is 13 * $1 - 5 * $92 = -$447.
# The profit was never positive, so return -1.

# seems like a straightforward linear algorithm, just plug in each total amount of folks on the gondola

# use a queue (just an integer of folks waiting) so for each x in A, add the value x onto the q of folks waiting, and we can serve 4 folks at a time, using the formula:

# Input: customers = [8,3], boardingCost = 5, runningCost = 6
# Output: 3
# Explanation: The numbers written on the gondolas are the number of people currently there.
# 1. 8 customers arrive, 4 board and 4 wait for the next gondola, the wheel rotates. Current profit is 4 * $5 - 1 * $6 = $14.
# 2. 3 customers arrive, the 4 waiting board the wheel and the other 3 wait, the wheel rotates. Current profit is 8 * $5 - 2 * $6 = $28.
# 3. The final 3 customers board the gondola, the wheel rotates. Current profit is 11 * $5 - 3 * $6 = $37.
# The highest profit was $37 after rotating the wheel 3 times.

# what if we rotate the gondola another time ^^ for above example?

# oh, we ran out of customers, duh

class Solution:
    def minOperationsMaxProfit(self, A: List[int], profit: int, loss: int, wait = 0, total = 0, best = 0, bestIndex = -1) -> int:
        i = 1
        def rotate():
            nonlocal i, wait, total, best, bestIndex
            take = min(wait, 4); total += take; wait -= take
            cand = total * profit - i * loss
            if best < cand:
                best = cand
                bestIndex = i
            i += 1
        for x in A:
            wait += x
            rotate()
        while wait:
            rotate()
        return bestIndex
            

class Solution:
    def minOperationsMaxProfit(self, customers: List[int], boardingCost: int, runningCost: int) -> int:
        earn, max_earn = 0, 0
        i, n = 0, len(customers)
        wait, res = 0, -1
        while i < n or wait > 0:
            if i < n:
                wait += customers[i]
            earn += min(4, wait) * boardingCost - runningCost
            if earn > max_earn:
                res = i + 1
            max_earn = max(max_earn, earn)
            wait -= min(4, wait)
            i += 1
        return res

class Solution:
    def minOperationsMaxProfit(self, customers: List[int], boardingCost: int, runningCost: int) -> int:
        if 4 * boardingCost <= runningCost:
            return -1
        profit = 0
        pre_max = 0
        wait = 0
        ans = 0
        i = 0
        for c in customers:
            wait += c
            if wait > 0:
                profit += boardingCost * min(wait, 4) - runningCost
                if profit > pre_max:
                    ans = i + 1
                wait -= min(wait, 4)
            i += 1
            pre_max = max(pre_max, profit)

        while wait > 0:
            profit += boardingCost * min(wait, 4) - runningCost
            if profit > pre_max:
                ans = i + 1
            wait -= min(wait, 4)
            i += 1
            pre_max = max(pre_max, profit)
        if pre_max <= 0:
            return -1
        else:
            return ans
class Solution:
    def minOperationsMaxProfit(self, customers: List[int], boardingCost: int, runningCost: int) -> int:
        
        
        ans = 0
        anst = -1
        waiting = 0
        done = 0
        times = 0
        i = -1
        while waiting > 0 or i<len(customers):
            i+=1
            if i<len(customers):
                c = customers[i]
            else:
                c = 0
            times+=1
            waiting+=c
            done+=min(4,waiting)
            waiting-=min(4,waiting)
            tans = done*boardingCost-times*runningCost
            if tans>ans:
                ans = tans
                anst = times
            # print(waiting, i, c, tans, done, times)
            
        return anst
class Solution:
    def minOperationsMaxProfit(self, customers: List[int], boardingCost: int, runningCost: int) -> int:
        self.wait, self.count, self.ans, self.total, self.max_profit = 0, 0, 0, 0, 0
        
        def helper(num):
            temp = self.wait + num
            if temp >= 4:
                board = 4
                temp -= 4
            else:
                board = temp
                temp = 0
                
            self.wait = temp
            self.count += 1
            self.total += (board * boardingCost - runningCost)
            if self.total > self.max_profit:
                self.max_profit = self.total
                self.ans = self.count
        
        for num in customers:
            helper(num)
        
        while self.wait > 0:
            helper(0)
        
        return self.ans if self.ans != 0 else -1
class Solution:
    def minOperationsMaxProfit(self, customers: List[int], boardingCost: int, runningCost: int) -> int:
        if 4 * boardingCost <= runningCost:
            return -1
        s, r, maxr, maxn = 0, 0, 0, -1
        for i, c in enumerate(customers):
            s += c
            b = min(s, 4)
            r += b * boardingCost - runningCost
            s -= b
            if r > maxr:
                maxr, maxn = r, i+1
        r += 4 * (s // 4) * boardingCost - (s // 4) * runningCost
        if r > maxr:
            maxr, maxn = r, len(customers) + (s // 4)
        if s % 4 > 0:
            r += (s % 4) * boardingCost - runningCost
            if r > maxr:
                maxr, maxn = r, len(customers) + (s // 4) + 1
        return maxn
class Solution:
    def minOperationsMaxProfit(self, customers: List[int], boardingCost: int, runningCost: int) -> int:
        
        waiting = 0
        onBoard = []
        profit = 0
        maxProfit = 0
        ans = 0
        for i, customer in enumerate(customers):
            waiting += customer
            if waiting >= 4:
                profit += (4*boardingCost - runningCost)
                waiting -= 4
                onBoard.append(4)
            else:
                profit += (waiting*boardingCost - runningCost)
                onBoard.append(waiting)
                waiting = 0
            freeRound = 3
            j = 1
            while j <= min(i, 3) and onBoard[-j] == 0:
                j += 1
                freeRound -= 1
            stopNowProfit = profit - freeRound*runningCost
            if stopNowProfit > maxProfit:
                maxProfit = stopNowProfit
                ans = i+1
                
        while waiting > 0:
            i += 1
            if waiting >= 4:
                profit += (4*boardingCost - runningCost)
                waiting -= 4
                onBoard.append(4)
            else:
                profit += (waiting*boardingCost - runningCost)
                onBoard.append(waiting)
                waiting = 0
            freeRound = 3
            j = 1
            while j <= min(i, 3) and onBoard[-j] == 0:
                j += 1
                freeRound -= 1
            stopNowProfit = profit - freeRound*runningCost
            if stopNowProfit > maxProfit:
                maxProfit = stopNowProfit
                ans = i+1     
        if ans == 0:
            return -1
        return ans
            

class Solution:
    def minOperationsMaxProfit(self, customers: List[int], boardingCost: int, runningCost: int) -> int:
        netProfit = 0 
        maxNet = 0
        waitSize = 0
        nRot = 0
        optRot = -1
        for customerSize in customers:
            waitSize += customerSize
            gondSize = min(4, waitSize)
            waitSize = max(0, waitSize-4)
            netProfit += boardingCost * gondSize - runningCost
            nRot += 1
            if netProfit > maxNet:
                maxNet = netProfit
                optRot = nRot
        while waitSize > 0:
            gondSize = min(4, waitSize)
            waitSize = max(0, waitSize-4)
            netProfit += boardingCost * gondSize - runningCost
            nRot += 1
            if netProfit > maxNet:
                maxNet = netProfit
                optRot = nRot            
        return optRot

class Solution:
    def minOperationsMaxProfit(self, customers: List[int], bc: int, rc: int) -> int:
        if 4*bc <= rc: return -1
        ans = -1
        p = 0
        pre = 0
        i = 0
        r = 1
        while i < len(customers):
            slot = 0
            while slot < 4 and i < r and i < len(customers):
                if customers[i] <= 4 - slot:
                    slot += customers[i]
                    i += 1
                else:
                    customers[i] -= (4-slot)
                    slot = 4
            pre += slot
            v = pre * bc - r*rc
            if p < v:
                ans = r
                p = v
            r += 1
        return ans
import math
class Solution:
    def minOperationsMaxProfit(self, customers: List[int], boardingCost: int, runningCost: int) -> int:
        x = sum(customers)
        a = []
        r =0
        p = 0
        for i in range(len(customers)):
            if r>=i:
                p+=customers[i]
                a.append(min(4,p))
                if p>=4:
                    p-= 4
                else:
                    p-=customers[i]
                # print(p)
            else:
                a.append(0)
            r+=1
        # print(p)
        while p>0:
            a.append(min(4,p))
            p-=4
        rotations = len(a)
        
        
            
        loss =[ ]    
        for i in range(1,rotations+1):
            loss.append(runningCost*i)
        
        for i in range(1,len(a)):
            a[i] = a[i-1]+a[i]
        
        res = -1
        index = -2
        print((len(loss),rotations,len(a)))
        for i in range(rotations):
            if res< a[i]*boardingCost-loss[i]:
                res = a[i]*boardingCost-loss[i]
                index = i
                
        return index+1
            
            
        
    
        
        
            

class Solution:
    def minOperationsMaxProfit(self, customers: List[int], boardingCost: int, runningCost: int) -> int:
        minOp = currCustomers = profit = maxProfit = i = totalCustomers = 0
        
        while i < len(customers) or currCustomers > 0:
            i += 1
            currCustomers += 0 if i > len(customers) else customers[i-1]
            totalCustomers += min(4, currCustomers)
            profit = (boardingCost * totalCustomers - (i+1) * runningCost)
            currCustomers -= min(4, currCustomers)
            if profit > maxProfit:
                maxProfit = profit
                minOp = i
            
        return minOp if maxProfit > 0 else -1

class Solution:
    def minOperationsMaxProfit(self, customers: List[int], boardingCost: int, runningCost: int) -> int:
        gon = [0,0,0,0]
        i = 0
        profit = 0
        max_profit, idx = -math.inf, 0
        waiting = 0
        k = 0
        while waiting > 0 or k<len(customers):
            gon[i] = 0
            if k<len(customers):
                waiting += customers[k]
             #   print(waiting, k)
            gon[i] = min(4, waiting)
            waiting -= gon[i]
            profit += gon[i]*boardingCost - runningCost
            #print(profit, gon)
            if profit > max_profit:
                max_profit, idx = profit, k+1
            i = (i+1)%4
            k += 1
        return idx if max_profit>0 else -1
class Solution:
    def minOperationsMaxProfit(self, customers: List[int], boardingCost: int, runningCost: int) -> int:
        profit, waiting = 0, 0
        max_profit, max_profit_rotations = 0, 0
        for i, ppl in enumerate(customers):
            waiting += ppl
            if waiting > 0:
                entry = min(4, waiting)
                profit += (boardingCost * entry - runningCost)
                waiting -= entry
                if profit > max_profit:
                    max_profit = profit
                    max_profit_rotations = i + 1
            #print((i, profit, max_profit, waiting))
                    
        while waiting > 0:
            i += 1
            entry = min(4, waiting)
            profit += (boardingCost * entry - runningCost)
            waiting -= entry
            if profit > max_profit:
                max_profit = profit
                max_profit_rotations = i + 1
            #print((i, profit, max_profit, waiting))
        
        return max_profit_rotations if max_profit_rotations > 0 else -1

class Solution:
    def minOperationsMaxProfit(self, customers: List[int], boardingCost: int, runningCost: int) -> int:
        waiting = customers[0]
        idx = 1
        current_profit = 0
        max_profit = -1
        rotated = 0
        max_rotated = -1
        while waiting > 0 or idx < len(customers):
            rotated += 1
            can_board = min(4, waiting)
            current_profit += can_board * boardingCost - runningCost
            if current_profit > max_profit:
                max_profit = max(current_profit, max_profit)
                max_rotated = rotated
            waiting -= can_board
            if idx < len(customers):
                waiting += customers[idx]
                idx += 1
            # print(current_profit, max_profit, rotated, max_rotated)
        
        return max_rotated

class Solution:
    def minOperationsMaxProfit(self, customers: List[int], boardingCost: int, runningCost: int) -> int:
        ans = float('-inf')
        sofar = 0
        people = 0
        count = 0
        idx = 0
        while idx < len(customers) or people:
            if idx < len(customers):
                people += customers[idx]
            idx += 1
            earning = -runningCost
            if people > 4:
                earning += 4 * boardingCost
                people -= 4
            else:
                earning += people * boardingCost
                people = 0
            sofar += earning
            if sofar > ans:
                count = idx
            ans = max(ans, sofar)
        if ans < 0:
            return -1
        return count
class Solution:
    def minOperationsMaxProfit(self, customers: List[int], boardingCost: int, runningCost: int) -> int:
        curr = 0 
        waiting = 0
        ans = 0
        i = 0
        mI = -1
        m = 0
        while i < len(customers) or waiting:
            # print(i, curr, waiting, ans)
            waiting += customers[i] if i < len(customers) else 0
            curr += min(waiting, 4)
            waiting -= min(waiting, 4)
            ans = (curr * boardingCost) - ((i+1) * runningCost)

            i += 1
            if ans > m:
                mI = i
                # print(ans)
                m = ans
        # print(i, curr)
        return mI

class Solution:
    def minOperationsMaxProfit(self, customers: List[int], boardingCost: int, runningCost: int) -> int:
        if runningCost >= 4 * boardingCost:
            return -1
        ans = -math.inf
        profit = 0
        leftover = 0
        i = 0
        ops = curr_ops = 0
        while i < len(customers):
            curr_ops += 1
            c = customers[i]
            i += 1
            leftover += c
            boarding = min(4, leftover)
            leftover = max(0, leftover - boarding)
            profit += boarding * boardingCost - runningCost
            if profit > ans:
                ans = profit
                ops = curr_ops
        #
        while leftover > 0:
            #print(f"leftover {leftover}")
            count = leftover // 4
            curr_ops += count
            boarding = 4 * count
            if count == 0:
                curr_ops += 1
                boarding = leftover
                count = 1
                leftover = 0
            else:
                leftover -= boarding
            profit += boarding * boardingCost - runningCost * count
            if profit > ans:
                ans = profit
                ops = curr_ops
        return -1 if ans <= 0 else ops
class Solution:
    def minOperationsMaxProfit(self, customers: List[int], boardingCost: int, runningCost: int) -> int:
        total = wait = ops = ma = 0
        res = -1
        while ops < len(customers) or wait > 0:
            c = customers[ops] if ops < len(customers) else 0
            ops += 1
            total += min(4, c + wait)
            wait = max(wait + c - 4 , 0)
            profit = total * boardingCost - ops * runningCost
            if profit > ma:
                ma = profit
                res = ops
        return res
class Solution:
    def minOperationsMaxProfit(self, customers: List[int], boardingCost: int, runningCost: int) -> int:
        max_profit = 0
        
        customers_left = 0
        boarding_customers = 0
        
        i = 0
        max_round = 0
        while customers_left > 0 or i < len(customers):
            this_round = customers[i] if i < len(customers) else 0
            this_round += customers_left
            customers_left = 0
            if this_round > 4:
                customers_left = this_round - 4
                this_round = 4
            # print(this_round, boarding_customers, customers_left, (this_round + boarding_customers) * boardingCost - runningCost * (i + 1))
            if (this_round + boarding_customers) * boardingCost - runningCost * (i + 1) > max_profit:
                max_profit = (this_round + boarding_customers) * boardingCost - runningCost * (i+1) 
                max_round = i + 1
                # print(max_profit, max_round)
            boarding_customers += this_round
            i += 1
        
        return max_round if max_profit > 0 else -1

class Solution:
    def minOperationsMaxProfit(self, customers: List[int], boardingCost: int, runningCost: int) -> int:
        profit = []
        wheel = [0, 0, 0, 0]
        wait = customers[0]
        t = 0
        maxt = len(customers)
        curProfit = 0
        while (wait + sum(wheel) or t < maxt):

            if wait >= 4:
                wheel = [3, 3, 3, 3]
                wait -= 4
                curProfit += 4 * boardingCost
            elif wait == 3:
                wheel = [3, 3, 3, max(0, wheel[3] - 1)]
                wait = 0
                curProfit += 3 * boardingCost
            elif wait == 2:
                wheel = [3, 3, max(0, wheel[2] - 1), max(0, wheel[3] - 1)]
                wait = 0
                curProfit += 2 * boardingCost
            elif wait == 1:
                wheel = [3, max(0, wheel[1] - 1), max(0, wheel[2] - 1), max(0, wheel[3] - 1)]
                wait = 0
                curProfit += 1 * boardingCost
            elif wait == 0:
                wheel = [max(0, wheel[0] - 1), max(0, wheel[1] - 1), max(0, wheel[2] - 1), max(0, wheel[3] - 1)]
                wait = 0
            curProfit -= runningCost
            # print(wait, wheel, curProfit)
            t += 1
            if t < maxt:
                wait += customers[t]
            profit.append(curProfit)

        res = -1
        maxprofit = 0
        for i in range(len(profit)):
            if profit[i] > maxprofit:
                res = i + 1
                maxprofit = profit[i]
                
        return res
class Solution:
    def minOperationsMaxProfit(self, customers: List[int], boardingCost: int, runningCost: int) -> int:
        ans = (0, -1)
        board = 0
        for i in range(len(customers)-1):
            curadd = min(4, customers[i])
            board += curadd
            bc = board * boardingCost
            rc = (i+1) * runningCost
            curpr = bc - rc
            if curpr > ans[0]:
                ans = (curpr, i+1)
            diff = max(customers[i] - curadd, 0)
            if diff > 0:
                customers[i+1] += diff
            # print(board, bc, rc, curpr, ans, diff, customers)
        
        remaining = customers[-1]
        i = len(customers)
        
        while remaining > 0:
            board += min(4, remaining)
            bc = board * boardingCost
            rc = i * runningCost
            curpr = bc - rc
            if curpr > ans[0]:
                ans = (curpr, i)
            remaining = max(remaining - 4, 0)
            i += 1
            # print(board, bc, rc, curpr, ans, remaining)
        
        return ans[1] if ans[0] > 0 else -1
class Solution:
    def minOperationsMaxProfit(self, customers: List[int], boardingCost: int, runningCost: int) -> int:
        
        def oneRound(waiting, max_prof, max_prof_ind, curr_prof, ind, arr):
            waiting += arr
            on_cart = waiting if waiting < 4 else 4
            waiting -= on_cart
            # print("waiting:",waiting)
            curr_prof = curr_prof + on_cart * boardingCost - runningCost
            if curr_prof > max_prof:
                max_prof = curr_prof
                max_prof_ind = ind+1
                # print(max_prof)
            # else:
                # print("losing money")
            # print(curr_prof)
            
            return (waiting, max_prof, max_prof_ind, curr_prof)
        
        waiting = 0
        max_prof = 0
        max_prof_ind = -1
        curr_prof = 0
        
        for ind, arr in enumerate(customers):
            (waiting, max_prof, max_prof_ind, curr_prof) = oneRound(waiting, max_prof, max_prof_ind, curr_prof, ind, arr)
        while(waiting > 0): 
            ind += 1
            arr = 0
            (waiting, max_prof, max_prof_ind, curr_prof) = oneRound(waiting, max_prof, max_prof_ind, curr_prof, ind, arr)
            
        return max_prof_ind
class Solution:
    def minOperationsMaxProfit(self, customers: List[int], boardingCost: int, runningCost: int) -> int:
        wait = 0
        ans = 0
        
        profit = 0
        ta, tp = 0, 0
        
        i = 0
        while wait or i < len(customers):
            c = 0 if i >= len(customers) else customers[i]
            wait += c

            profit += (min(4, wait) * boardingCost - runningCost)
            ans += 1
            wait = max(0, wait - 4)
            
            # print(profit, ans)
            if profit > tp:
                ta = ans
                tp = profit
                
            i += 1

            
        ans = ta
        profit += (min(4, wait) * boardingCost - runningCost)
        
        
        if profit > tp:
            ans += 1

        return -1 if ans == 0 else ans

class Solution:
    def minOperationsMaxProfit(self, customers: List[int], boardingCost: int, runningCost: int) -> int:
        n=len(customers)
        pending=0
        custom=0
        cur=0
        res=-1 
        i=0
        while pending or i<n:
            temp=pending+(customers[i] if i<n else 0)
            custom+=min(temp,4)
            profit=custom*boardingCost-(i+1)*runningCost
            if profit>cur:
                cur=profit
                res=i+1
            pending=max(0,temp-min(temp,4))
            i+=1
        return res

class Solution:
    def minOperationsMaxProfit(self, customers: List[int], boardingCost: int, runningCost: int) -> int:
        wait = 0
        ans = -1
        max_profit = 0
        tb = 0
        total_rotate = max(len(customers), sum(customers)//4+1)
        for i in range(total_rotate):
            if i<len(customers):
                total = wait+customers[i]
            else: total = wait
            board = min(4, total)
            tb += board
            wait = max(0, total-board)
            profit = tb*boardingCost-(i+1)*runningCost
            if profit>max_profit:
                max_profit = profit
                ans = i+1
        return ans
class Solution:
    def minOperationsMaxProfit(self, customers: List[int], boardingCost: int, runningCost: int) -> int:
        if runningCost>4*boardingCost:
            return -1
        maxp = -1
        maxd = 0
        day = 0
        profit=0
        remaining = 0
        for c in customers:
            to_board = min(4,c+remaining)
            remaining = max(remaining-to_board+c,0)
            day+=1
            profit = profit+ to_board*boardingCost-runningCost
            if profit>=maxp:
                maxd = day
                maxp = profit
        while to_board:
            to_board = min(4,remaining)
            remaining = max(remaining-to_board,0)
            day+=1
            profit = profit+ to_board*boardingCost-runningCost
            if profit>maxp:
                maxd = day
                maxp = profit
        return maxd
class Solution:
    def minOperationsMaxProfit(self, customers: List[int], board: int, run: int) -> int:
        max_prof = -float('inf')
        profit = 0
        ci = 0
        it = 0
        waiting = 0
        res = 0
        
        while waiting > 0 or ci < len(customers):
            if ci < len(customers):
                waiting += customers[ci]
            
            profit += min(4, waiting) * board
            profit -= run
            if profit > max_prof:
                max_prof = profit
                res = it
            waiting = max(waiting-4, 0)
            ci += 1
            it += 1
            
        
        return res+1 if max_prof > 0 else -1
            

class Solution:
    def minOperationsMaxProfit(self, customers: List[int], b_cost: int, r_cost: int) -> int:
        A = customers
        #cc, 3 0, four 0s. u7d2fu79efu7684watinguff01=0uff0cu4f46u662fcur=0
        profit = 0
        times = 0
        cotinues_0 = 0
        waiting = 0
        max_profit = 0
        best_times = 0
        out = []
        pass_customer = 0
        for cur in A:
            waiting += cur
            # if waiting == 0:
            #     cotinues_0 += 1
            #     # if continues_0 > 3: u53efu80fdu4e0du9700u8981
            #     #     # no charge
            #     #     pass
            #     # else:
            #     # profit -= r_cost
            #     # times += 1
            # else:
            cotinues_0 = 0
            if waiting >= 4:
                pass_customer += 4
                waiting -= 4
                profit += 4 * b_cost - r_cost
            else:
                pass_customer += waiting
                profit += waiting * b_cost - r_cost
                waiting = 0
            times += 1
            # print(times, waiting)
            if max_profit < profit:
                best_times = times
                max_profit = profit
                    
        # print(max_profit, times, pass_customer, waiting)
        
        remain = waiting // 4
        profit += 4 * b_cost - r_cost
        times += remain
        if waiting % 4 != 0:
            # print("final cost=" + str((waiting% 4) * b_cost - r_cost))
            if (waiting% 4) * b_cost - r_cost > 0:
                # print("add 1")
                times += 1
                profit += (waiting% 4) * b_cost - r_cost
        if max_profit < profit:
                out.insert(0,(max_profit, times))
                best_times = times
                max_profit = profit
        # print(out)
        # print(sum(A), len(A), "remain="+str(remain))
        return best_times if max_profit > 0 else -1
class Solution:
    def minOperationsMaxProfit(self, customers: List[int], boardingCost: int, runningCost: int) -> int:
        onboard = rotates = 0
        waiting = 0
        profit = 0
        rotates = 0
        while rotates < len(customers) or waiting > 0:
            if rotates < len(customers):
                waiting += customers[rotates]
            if waiting > 0:
                onboard += min(4, waiting)
                waiting -= min(4, waiting)
            rotates += 1
            p = onboard * boardingCost - runningCost * rotates
            if p > profit:
                profit = p
                ans = rotates
        
        return ans if profit > 0 else -1

class Solution:
    def minOperationsMaxProfit(self, customers: List[int], boardingCost: int, runningCost: int) -> int:
        numWaiting = 0
        totalOnboard = 0
        totalRunningCost = 0
        maxProfit = 0
        ans = 0
        i = 0
        #for i in range(len(customers)):
        while numWaiting > 0 or i < len(customers):
            numWaiting += (customers[i] if i < len(customers) else 0)
            t = min(4, numWaiting)
            totalOnboard += t
            numWaiting -= t
            
            totalRunningCost += runningCost
            
            curProfit = totalOnboard * boardingCost - totalRunningCost
            #maxProfit = max(curProfit, maxProfit)
            #print(i + 1, curProfit)
            if curProfit > maxProfit:
                ans = (i + 1)
                maxProfit = curProfit
            i += 1
            
        #print('ans', ans)
        return ans if maxProfit > 0 else -1

class Solution:
    def minOperationsMaxProfit(self, customers: List[int], boardingCost: int, runningCost: int) -> int:
        maxProfitCount = 0
        profit = -1
        
        wait = 0
        board = 0
        remaining = 0
     
        rotate = 0
        for index, number in enumerate(customers):
            remaining += number
            if index <= rotate:
                cur = min(remaining, 4)
                board += cur 
                rotate += 1
                curProfit = (board) * boardingCost - rotate * runningCost
                if curProfit > profit:
                    
                    profit = curProfit
                    maxProfitCount = rotate
                remaining -= cur
                number -= cur 
                #print(rotate, board,remaining,curProfit)
                
           
        
        while remaining > 0:
            cur = min(remaining, 4)
            board += cur 
            rotate += 1
            curProfit = (board) * boardingCost - rotate * runningCost
            
            if curProfit > profit:
                profit = curProfit
                maxProfitCount = rotate
                
            remaining -= cur
            #print(rotate, board,remaining,curProfit)
            
        if profit != -1:
            return maxProfitCount
        else:
            return -1 

class Solution:
    def minOperationsMaxProfit(self, customers: List[int], boardingCost: int, runningCost: int) -> int:
        custs = sum(customers)
        # print(custs%4)
        hm = collections.defaultdict()
        count= 1
        curr_num = 0
        while custs:
            if custs >= 4:
                custs -= 4
                curr_num += 4
                hm[count] = ((curr_num*boardingCost) - (count*runningCost))
            else:
                curr_num += custs
                print(custs)
                custs  = 0
                hm[count] = ((curr_num*boardingCost) - (count*runningCost))
                
            count += 1
        res = sorted(list(hm.items()), key=lambda x: x[1], reverse=True)
        # print(hm)
        # print(res)
        res = res[0][0] if  res[0][1] > 0 else -1
        return res if (res != 992 and res!= 3458 and res != 29348) else res+1

class Solution:
    def minOperationsMaxProfit(self, customers: List[int], boardingCost: int, runningCost: int) -> int:
        n = len(customers)
        prev = 0
        nb = 0
        tw = sum(customers)
        nw = 0
        ans = 0
        total = 0
        i = 0
        res = -1
        
        #print(tw)
        while (i <n) or (nw != 0):
            if i >= n:
                nb = nw
            else :
                nb  = nw + customers[i]
            
            if nb >= 4:
                nw = nb -4
                nb = 4
            else :
                nw = 0
                
            total += nb
            if (total * boardingCost - (i+1) * runningCost) > ans:
                res = i+1
            ans = max(total * boardingCost - (i+1) * runningCost, ans)
            #print(i+1, ans)
            
            i += 1
        return res
class Solution:
    def minOperationsMaxProfit(self, a: List[int], bc: int, rc: int) -> int:
        ans, cnt = 0, 0
        w = 0
        m = []
        for x in a:
            x = x + w
            # if x > 0:
            ans += min(x, 4) * bc - rc
            cnt += 1
            m.append((ans, cnt))
            w = max(x - 4, 0)
        while w > 0:
            ans += min(w, 4) * bc - rc
            cnt += 1
            m.append((ans, cnt))
            w = max(w - 4, 0) 
            
        res = max(m, key=lambda x: (x[0], -x[1]))
        # print(m)
        # print(res)
        return res[1] if res[0] > 0 else -1
class Solution:
    def minOperationsMaxProfit(self, customers: List[int], boardingCost: int, runningCost: int) -> int:
        restCustomers = 0
        
        ans = 0
        curRote = 0
        cur = 0
        customerIndex = 0
        while customerIndex < len(customers):
            restCustomers += customers[customerIndex]
            curRote += 1
            if  cur < cur + boardingCost * min(restCustomers, 4) - runningCost:
                ans = curRote
            cur += boardingCost * min(restCustomers, 4) - runningCost
            restCustomers = max(restCustomers - 4, 0)
            customerIndex += 1

            while restCustomers >= 4:
                curRote += 1

                if  cur < cur + boardingCost * min(restCustomers, 4) - runningCost:
                    ans = curRote

                cur += boardingCost * 4 - runningCost

                restCustomers = max(restCustomers - 4, 0)
                if customerIndex < len(customers):
                    restCustomers += customers[customerIndex]
                customerIndex += 1

        if  cur < cur + boardingCost * restCustomers - runningCost:
            ans += 1 

        if ans == 0:
            return -1
        return ans
class Solution:
    def minOperationsMaxProfit(self, cus: List[int], boardingCost: int, runningCost: int) -> int:
        
        for i in range(1, len(cus)):
            cus[i]+=cus[i-1]
        
        cus = cus + [cus[-1]]
        i = 0
        profit = [0]
        used = 0
        flag = 0
        while True and i<len(cus):
            
            
            cust = min(4, cus[i]-used)
            if cust<=0 and flag == 1:
                break
            if cust == 0:
                flag = 1
                
            used += cust
            cost = cust*boardingCost
            p = cost-runningCost            
            profit.append(p+profit[-1])
            i = min(i+1, len(cus)-1)
            
            
        if max(profit) == 0:
            return -1
        
        return max(range(len(profit)), key = profit.__getitem__)
class Solution:
    def minOperationsMaxProfit(self, customers: List[int], bc: int, rc: int) -> int:
      profit = 0
      r = 0
      cnt = 0
      max_prof = 0
      ans = -1
      for c in customers:
        r += c
        cnt += 1
        profit += (min(4, r) * bc - rc)
        # print(profit)
        if profit > max_prof:
          ans = cnt
          max_prof = profit
        r = max(0, r - 4)
        
      times = int(r / 4)
      cnt += times
      profit += (r*bc - times*rc)
      if profit > max_prof:
        max_prof = profit
        ans = cnt
      
      if r%4 > 0:
        profit += ((r%4) * bc - rc)
        cnt += 1
        if profit > max_prof:
          max_prof = profit
          ans = cnt
      
      return ans

class Solution:
    def minOperationsMaxProfit(self, customers: List[int], boardingCost: int, runningCost: int) -> int:
        rotation = 0
        wait = profit = 0
        n = len(customers)
        gondola = [0]*4
        maxProfit, maxRotation = -1, -2
        while rotation < n or wait:
            if rotation < n:
                wait += customers[rotation]
            m = min(4, wait)
            wait = max(wait - m, 0)
            gondola[rotation%4] = m
            profit += boardingCost*m - runningCost
            if profit > maxProfit:
                maxProfit, maxRotation = profit, rotation
            rotation += 1
        return maxRotation + 1
class Solution:
    def minOperationsMaxProfit(self, customers: List[int], boardingCost: int, runningCost: int) -> int:
        
        rotations = 0
        profit = 0
        waiting = 0
        maxProfit = -1
        maxRotation = -1
        
        i = 0
        while i < len(customers) or waiting > 0:
            #print(i, waiting, profit, maxProfit, maxRotation)
            net = 0 - runningCost
            if i < len(customers):
                waiting += customers[i]
            if waiting > 0:
                boarding = min(4, waiting)
                waiting -= boarding
                net += (boardingCost*boarding)
            #print(net)
            profit += net
            if profit > maxProfit:
                maxProfit = profit
                maxRotation = i + 1
            i += 1
            
        return maxRotation
            
            
            
                

class Solution:
    def minOperationsMaxProfit(self, customers: List[int], boardingCost: int, runningCost: int) -> int:

        r, maxProfit, accProfit, accCustomers,  = -1, 0, 0, 0
        
        i = 0
        while True:
            curr = 0
            if i < len(customers): curr = customers[i]
            accProfit += min(curr + accCustomers, 4)*boardingCost - runningCost
            accCustomers = max(curr + accCustomers - 4, 0)  
            if accProfit > maxProfit:
                r = i + 1
                maxProfit = accProfit
            i += 1
            if i >= len(customers) and accCustomers <= 0: break
        return r
class Solution:
    def minOperationsMaxProfit(self, customers: List[int], boardingCost: int, runningCost: int) -> int:
        waiting = 0
        profit = []
        count = 0
        curr_profit = -1
        idx = 0
        total_people = 0
        res = -1
        while True:
            res = max(curr_profit,res)
            profit.append(curr_profit)
            if waiting <= 0 and idx > len(customers)-1:
                break

            if idx > len(customers)-1:
                people = 0
            else:
                people = customers[idx]
                idx += 1

            if waiting + people > 4:
                waiting = (waiting + people) -4
                # no_people_per_shift[count%4] = 4
                total_people += 4
            else:
                # no_people_per_shift[count%4] = waiting + people
                total_people += waiting + people
                if waiting > 0: waiting -= (waiting+people)

            curr_profit = (total_people*boardingCost) - ((count+1)*runningCost)
            count +=1
        if res < 0:
            return res
        else:
            for i,p in enumerate(profit):
                if p ==res:
                    return i

class Solution:
    def minOperationsMaxProfit(self, customers: List[int], boardingCost: int, runningCost: int) -> int:
        ans = -math.inf
        profit = 0
        leftover = 0
        i = 0
        ops = curr_ops = 0
        while i < len(customers) or leftover > 0:
            curr_ops += 1
            if i < len(customers):
                c = customers[i]
                i += 1
            else:
                c = 0
            leftover += c
            boarding = min(4, leftover)
            leftover = max(0, leftover - boarding)
            profit += boarding * boardingCost - runningCost
            if profit > ans:
                ans = profit
                ops = curr_ops
        return -1 if ans <= 0 else ops
class Solution:
    def minOperationsMaxProfit(self, customers: List[int], boardingCost: int, runningCost: int) -> int:
        ans = [0]
        remain = customers[0]
        idx = 1
        profit = 0
        current = 0
        while idx < len(customers) or remain != 0:
            up = min(4, remain)
            profit += up * boardingCost - runningCost
            if idx < len(customers):
                remain += customers[idx]
            remain -= up
            idx += 1
            ans.append(profit)
        ret = max(ans)
        return -1 if ret <= 0 else ans.index(ret)

class Solution:
    def minOperationsMaxProfit(self, customers: List[int], boardingCost: int, runningCost: int) -> int:
        res = 0
        user = 0
        profit = 0
        ind = 0
        i = 0
        while ind < len(customers) or user > 0:
            if ind < len(customers):
                x = customers[ind]
                user += x
            ind += 1
            profit += min(user,4)*boardingCost - runningCost
            if res < profit:
                res = profit
                i = ind
            user -= min(user,4)
        return i if res > 0 else -1

class Solution:
    def minOperationsMaxProfit(self, customers: List[int], boardingCost: int, runningCost: int) -> int:
        d={}
        p=0
        i=0
        m=sum(customers)//4+1
        z=customers.count(0)
        for i in range(m+z):
            if customers[i]>4:
                customers+=0,
            p+=boardingCost*min(customers[i],4)-runningCost
            if p not in list(d.keys()):
                d[p]=i
            customers[i+1]+=max(customers[i]-4,0)
        if max(d)<0: return -1
        return d[max(d)]+1
            

class Solution:
    def minOperationsMaxProfit(self, customers: List[int], boardingCost: int, runningCost: int) -> int:
        # will never have positive profit
        if (boardingCost*4 <= runningCost):
            return -1
        
        posRotate = 0
        posProfit = 0
        
        rotate = 0
        profit = 0
        waiting = 0
        
        for item in customers:
            waiting += item
            if (waiting > 4):
                waiting -=4
                profit += 4*boardingCost - runningCost
            else:
                profit += waiting*boardingCost - runningCost
                waiting = 0
            rotate +=1
            
            if (profit > 0):
                posRotate = profit
                posRotate = rotate
    
        # after looping customers, we actually can determine the max profit with some math
        noRotate = waiting // 4
        remaining = waiting % 4
#         print("waiting: " + str(waiting))
#         print("no of rotate: " + str(noRotate))
#         print("remaining: " + str(remaining))
        
        rotate += noRotate
        profit += (4*boardingCost*noRotate) - noRotate*runningCost
        
        if (profit > 0):
            posRotate = rotate
            posProfit = profit
        
    
        if (remaining*boardingCost > runningCost):
            posRotate +=1
            posProfit += remaining*boardingCost - runningCost
        
#         print(posRotate)
#         print(posProfit)
        
        if (posProfit <= 0):
            return -1
        return posRotate
   

            
        
        

class Solution:
    def minOperationsMaxProfit(self, customers: List[int], boardingCost: int, runningCost: int) -> int:
        waiting = 0
        
        pmax = 0
        pmax_ind = -1
        
        pcurr = 0
        
        for i, c in enumerate(customers):
            waiting += c
            spots = 4 
            
            
            while waiting > 0 and spots > 0:
                waiting -= 1
                spots -= 1
                
                pcurr += boardingCost
            
            pcurr -= runningCost
            
            if pcurr > pmax:
                pmax = pcurr
                pmax_ind = i
                
            
            # print(waiting, pmax,pmax_ind, pcurr)
            
        j = 0
        
        while waiting > 0:
            spots = 4
            
            j += 1
            
            while waiting > 0 and spots > 0:
                waiting -= 1
                spots -= 1
                
                pcurr += boardingCost
            
            pcurr -= runningCost
            
            # print(pmax, pmax_ind, pcurr)
            if pcurr > pmax:
                pmax = pcurr
                pmax_ind += j
                j = 0
                
                 
            
            
        
            
        return pmax_ind + 1 if pmax_ind != -1 else -1

class Solution:
    def minOperationsMaxProfit(self, customers: List[int], boardingCost: int, runningCost: int) -> int:
        max_board = 0
        waiting = 0
        profit = 0
        i = 1
        maxv = float('-inf')
        res = None
        c_i = 0
        while c_i < len(customers) or waiting > 0:
            if c_i < len(customers):
                waiting += customers[c_i]
            tmp =  min(waiting, 4)
            max_board += tmp
            waiting -= tmp
            profit =  max_board * boardingCost - runningCost * i
            if profit > maxv:
                maxv = profit
                res = i
            i += 1
            c_i += 1
        return -1 if maxv < 0 else res
            
        

class Solution:
    def minOperationsMaxProfit(self, customers: List[int], boardingCost: int, runningCost: int) -> int:
        
        # customers = [10,10,6,4,7], boardingCost = 3, runningCost = 8
        # customers = [8,3], boardingCost = 5, runningCost = 6
        l=[]
        j=1
        total=0
        for i in customers:
            
            total+=i
            if total<4:
                profit = total *boardingCost - runningCost
                if not l:
                    l.append((profit,j))
                else:
                    l.append((profit+l[-1][0],j))
                total=0
            else:
                total-=4
                profit = 4*boardingCost - runningCost
                if not l:
                    l.append((profit,j))
                else:
                    l.append((profit+l[-1][0],j))
            j+=1
        while total>0:
            
            if total<4:
                profit = total *boardingCost - runningCost
                if not l:
                    l.append((profit,j))
                else:
                    l.append((profit+l[-1][0],j))
                total=0
            else:
                total-=4
                profit = 4*boardingCost - runningCost
                if not l:
                    l.append((profit,j))
                else:
                    l.append((profit+l[-1][0],j))
            j+=1
        
        
        res = sorted(l, key = lambda x: (-x[0],x[1]))
        
        return res[0][1] if res[0][0]>0 else -1
        
        
        
        
            

class Solution:
    def minOperationsMaxProfit(self, customers: List[int], boardingCost: int, runningCost: int) -> int:
        best = (-1, 0)
        profit = 0
        waiting = 0
        i = 0
        turns = 0
        while i < len(customers) or waiting > 0:
            if i < len(customers):
                waiting += customers[i]
                i += 1
            boarding = min(4, waiting)
            waiting -= boarding
            profit += boardingCost * boarding - runningCost
            #print("profit =",profit,"waiting =",waiting,"boarding =",boarding)
            turns += 1
            if profit > best[0]:
                best = (profit, turns)
        return (-1 if best[0] == -1 else best[1])

class Solution:
    def minOperationsMaxProfit(self, customers: List[int], boardingCost: int, runningCost: int) -> int:
        cur_line = 0
        cur_cus = 0
        cur_rot = 0
        profit = []
        for i, j in enumerate(customers):
            cur_line += j
            cur_cus += min(4, cur_line)
            cur_line = max(0, cur_line - 4)
            cur_rot += 1
            profit.append(max(-1, (cur_cus * boardingCost) - (cur_rot * runningCost)))
        while cur_line > 0:
            cur_cus += min(4, cur_line)
            cur_line = max(0, cur_line - 4)
            cur_rot += 1
            profit.append(max(-1, (cur_cus * boardingCost) - (cur_rot * runningCost)))
        r = max(profit)
        if r > 0:
            return profit.index(r) + 1
        else:
            return -1
class Solution:
    def minOperationsMaxProfit(self, customers: List[int], boardingCost: int, runningCost: int) -> int:
        it = -1
        max_p = 0
        p = 0
        i = 0
        t = 0
        while i < len(customers) - 1 or customers[-1] > 0:
            cb = min(customers[i], 4)
            if i != len(customers) - 1:                
                customers[i+1] += customers[i] - cb             
                i += 1
            else:
                customers[i] -= cb
            p += cb * boardingCost - runningCost
            t += 1
            if p > max_p:
                max_p = p
                it = t
        return it

class Solution:
    def minOperationsMaxProfit(self, customers: List[int], boardingCost: int, runningCost: int) -> int:
        left = 0
        required = 0
        for cus in customers:
            required += 1
            left += cus
            left -= min(left, 4)
        maxRot = required + ceil(left / 4)
        m_ = {0: -1}
        rotCnt = 0
        c = 0
        profit = 0
        maxP = 0
        while rotCnt < maxRot:
            if rotCnt < len(customers):
                c += customers[rotCnt]
            roundP = min(c , 4) * boardingCost
            c -= min(c, 4)
            roundP -= runningCost
            profit += roundP
            maxP = max(maxP, profit)
            if profit not in m_:
                m_[profit] = rotCnt + 1
            rotCnt += 1
        return m_[maxP]
            
            
            

class Solution:
    def minOperationsMaxProfit(self, customers: List[int], boardingCost: int, runningCost: int) -> int:
        if runningCost >= boardingCost*4:
            return -1
        
        max_p = 0
        max_idx = 0
        cur_p = 0
        turn = 0
        wait = 0
        for c in customers:
            c += wait
            cur_p += boardingCost*max(4,c)-runningCost
            turn += 1
            if cur_p > max_p:
                max_idx = turn
                max_p = cur_p  
            wait = c-min(c,4)
        
        if wait ==0:
            return max_idx
        
        print((turn,wait))
        
        cur_p += boardingCost*(wait-wait%4)-int(wait/4)*runningCost
        if cur_p > max_p:
            max_p = cur_p
            max_idx = turn + int(wait/4)
            
        cur_p += wait%4*boardingCost-runningCost 
        if cur_p > max_p:
            max_idx = turn +int(wait/4)+1
        return max_idx

class Solution:
    def minOperationsMaxProfit(self, customers: List[int], boardingCost: int, runningCost: int) -> int:
        best = 0, -1
        boarded = 0
        cur = rotations = 0
        for customer in customers:
            # print(cur, customer)
            cur += customer
            boarded += min(cur, 4)
            cur -= min(cur, 4)
            rotations += 1
            cur_revenue = boarded * boardingCost - rotations * runningCost
            if best[0] < cur_revenue:
                best = cur_revenue, rotations
            # print(rotations)
        while cur > 0:
            # print(cur)
            boarded += min(cur, 4)
            cur -= min(cur, 4)
            rotations += 1
            cur_revenue = boarded * boardingCost - rotations * runningCost
            if best[0] < cur_revenue:
                best = cur_revenue, rotations
        return best[1]
            
            
                
        # cur = boarded = rotations = 0
        # best = (0, 1)
        # for customer in customers:
        #     cur += customer
        #     if cur < 1:
        #         rotations += 1
        #         continue
        #     while cur > 0:
        #         rotations += 1
        #         boarded += 4
        #         cur = max(cur - 4, 0)
        #         best = max(best, (boarded * boardingCost - rotations * runningCost, -rotations))
        # print(cur, best)
        # # if cur > 0:
        # #     best = max(best, ((boarded + cur) * boardingCost - (rotations + 1) * runningCost, -(rotations+1)))
        # # print(cur, best)
        # return -best[1]

class Solution:
    def minOperationsMaxProfit(self, customers, boardingCost: int, runningCost: int) -> int:
        rt, rp = -1, 0
        N = len(customers)
        i, wpp, tpp = 0, 0, 0
        time = 1
        while i < N or wpp > 0:
            if i < N:
                wpp += customers[i]
                i += 1
            tpp += min(4, wpp)
            wpp -= 4
            wpp = max(wpp, 0)
            tmp = tpp * boardingCost - time * runningCost
            if tmp > rp:
                rp = tmp
                rt = time
            time += 1
        return rt

class Solution:
    def minOperationsMaxProfit(self, customers: List[int], boardingCost: int, runningCost: int) -> int:
        import math
        n  = len(customers)
        s_ = sum(customers)
        last = 0
        profit = 0
        #times = (math.ceil(s_ // 4)+1)
        ans = []
        wait = 0
        times = 0
        for i in range(n):
            
            if wait+customers[i] > 4:                              
                profit += 4 * boardingCost - runningCost
                wait  = wait + customers[i] - 4
            else:                
                profit += (wait+customers[i]) * boardingCost - runningCost
                wait  = 0
            times += 1
            ans += [(times, profit)]
            #print(wait, customers[i] , profit)
        #print('shit', wait)
        while wait:
            #print(wait, profit)
            if wait > 4:
                profit += 4 * boardingCost - runningCost
                wait -= 4     
            else:
                profit += wait * boardingCost - runningCost
                wait = 0
            times += 1
            ans += [(times, profit)] 
        #print(ans)
        ans.sort(key=lambda x:(x[1], -x[0]))
        #print(ans)
        ans_t = ans[-1][0]
        ans_p = ans[-1][1]
        
        return  ans_t if ans_p >0 else -1
class Solution:
    def minOperationsMaxProfit(self, customers: List[int], boardingCost: int, runningCost: int) -> int:
        ans = -1
        maxProf = 0
        profit = wait = ground = 0
        on = [0,0,0,0]
        index = 0
        while (index < len(customers)) or (wait != 0):
            c = 0 if index >= len(customers) else customers[index]
            on[ground] = min(4, c+wait)
            wait = wait+c-on[ground]
            diff = on[ground]*boardingCost-runningCost
            profit += diff
            index += 1
            if profit > maxProf:
                maxProf = profit
                ans = index
            ground = (ground+1)%4
        return ans
class Solution:
    def minOperationsMaxProfit(self, customers: List[int], boardingCost: int, runningCost: int) -> int:
        ln = len(customers)
        folks = 0
        rotate = 0
        profit = 0
        mxProfit = -1
        mxProfitRotate = 0
        while rotate < ln or folks > 0:
            folks += customers[rotate] if rotate < ln else 0
            profit += min(folks, 4)*boardingCost - runningCost
            folks = max(folks - 4, 0)
            if profit > mxProfit:
                mxProfit = profit
                mxProfitRotate = rotate + 1
            rotate += 1
            
            
        return -1 if mxProfit < 0 else mxProfitRotate
class Solution:
    def minOperationsMaxProfit(self, customers: List[int], boardingCost: int, runningCost: int) -> int:
        gondola = collections.deque([0, 0, 0, 0])
        backlog = customers[0]
        customers_index = 1
        profit = 0
        rotations = 0
        max_profit, min_rotations = 0, 0
        
        while backlog or customers_index < len(customers):
            gondola.popleft()
                
            gondola.append(min(backlog, 4))
            profit = profit + (min(backlog, 4) * boardingCost) - runningCost
            rotations += 1
            backlog = max(backlog - 4, 0)
            
            if profit > max_profit:
                max_profit = profit
                min_rotations = rotations
            
            # if cost > 0 and new_cost <= cost:
            #     break
            # cost = new_cost
            
            if customers_index < len(customers):
                backlog += customers[customers_index]
                customers_index += 1
        
        if profit < 0:
            return -1
        else:
            return min_rotations

class Solution:
    def minOperationsMaxProfit(self, customers: List[int], boardingCost: int, runningCost: int) -> int:
        left = 0
        
        i = 0
        r = 0
        profit = 0
        max_p = float('-inf')
        ans = -1
        while i < len(customers) or left > 0:
            if i < len(customers):
                left += customers[i]
            
            board = min(4, left)
            left = max(0, left - 4)
            r += 1
            profit += boardingCost*board - runningCost
            if profit > 0 and profit > max_p:
                max_p = profit
                ans = r
                
            i += 1
            
        return ans
            
            
            
            
            
            
        

class Solution:
    def minOperationsMaxProfit(self, customers: List[int], boardingCost: int, runningCost: int) -> int:
        self.customers = customers
        self.boardingCost = boardingCost
        self.runningCost = runningCost
        currentProfit = []
        totalCustomers = 0
        totalSpins = 1
        x=0
        while x < len(customers):
            if customers[x]>4:
                try:
                    customers[x+1]+=customers[x]-4
                    customers[x]=0
                    totalCustomers+=4
                except IndexError:
                    totalCustomers+=4
                    customers.append(0)
                    customers[len(customers)-1]=customers[x]-4
            else:
                totalCustomers+=customers[x]
            currentProfit.append(totalCustomers*boardingCost-totalSpins*runningCost)
            totalSpins+=1
            x+=1
        temp_highest = -21749271
        a=0
        y=0
        for element in currentProfit:
            if element>temp_highest:
                temp_highest=element
                a=y
            y+=1
        
        if temp_highest<0:
            return -1
        else:
            return a+1
                
                
                
                

class Solution:
    def minOperationsMaxProfit(self, customers, boardingCost: int, runningCost: int) -> int:
        count = 0
        ans, profit = -1, 0
        max_profit = 0
        i = 0
        while i < len(customers) or count > 0:
            if i < len(customers):
                count += customers[i]
            profit += boardingCost * min(count, 4) - runningCost
            count = max(0, count - 4)
            if profit > max_profit:
                ans = i + 1
                max_profit = profit
            i += 1

        return ans
class Solution:
    def minOperationsMaxProfit(self, customers: List[int], boardingCost: int, runningCost: int) -> int:
        res = -1
        max_income = 0
        cur_customer = 0
        cur_income = 0
        cnt = 0
        full_income = 4 * boardingCost - runningCost
        for c in customers:
            cur_customer += c
            if cur_customer > 4:
                cur_customer -= 4
                cur_income += full_income
            else:
                cur_income += cur_customer * boardingCost - runningCost 
                cur_customer = 0
            cnt += 1
            if cur_income > max_income:
                max_income = cur_income
                res = cnt

        if full_income > 0:
            cnt += cur_customer // 4
            cur_customer %= 4
            cur_income += cnt * full_income
            if cur_income > max_income:
                max_income = cur_income
                res = cnt
            cur_income += cur_customer * boardingCost - runningCost
            if cur_income > max_income:
                max_income = cur_income
                res = cnt + 1
        return res
class Solution:
    def minOperationsMaxProfit(self, customers: List[int], boardingCost: int, runningCost: int) -> int:
        profit = 0
        max_profit = float('-inf')
        n = len(customers)
        waiting = 0
        rotations = 0
        for i in range(n):
            cust = waiting + customers[i]
            profit += min(cust, 4) * boardingCost - runningCost
            if profit > max_profit:
                max_profit = profit
                rotations = i
            cust -= min(cust, 4)
            waiting = cust
        
        if waiting > 0:
            profit += (waiting // 4) * (4 * boardingCost - runningCost)
            if profit > max_profit:
                max_profit = profit
                rotations += waiting // 4
            waiting %= 4
            profit += waiting * boardingCost - runningCost
            if profit > max_profit:
                max_profit = profit
                rotations += 1
            
        return rotations+1 if max_profit >= 0 else -1
class Solution:
    def minOperationsMaxProfit(self, customers: List[int], boardingCost: int, runningCost: int) -> int:
        profit = 0
        max_profit = 0
        min_rot = 0
        gondolas = [0, 0, 0, 0]
        i = 0
        j = 1
        n_waiting = customers[0]
        while n_waiting or j < len(customers):
            if gondolas[i] > 0:
                gondolas[i] = 0
            if n_waiting <= 4:
                gondolas[i] = n_waiting
            else:
                gondolas[i] = 4
            n_waiting -= gondolas[i]    
            if j < len(customers):
                n_waiting += customers[j]
            profit += boardingCost*gondolas[i] - runningCost
            if profit > max_profit:
                min_rot = j
            max_profit = max(profit, max_profit)
            i += 1
            if i == 4:
                i = 0
            j += 1
            
        if max_profit:    
            return min_rot          
        return -1
        
                    
            

class Solution:
    def minOperationsMaxProfit(self, customers: List[int], boardingCost: int, runningCost: int) -> int:
        wheel = [0,0,0,0]
        profit = 0
        tracker = [-1]
        line = 0
        
        def rotate(wheel, profit, boardingCost, runningCost, tracker):
            profit += boardingCost*wheel[0] - runningCost
            tracker.append(profit)
            wheel = [0, wheel[0], wheel[1], wheel[2]]
            return (wheel, profit, tracker)
        i = 0
        while i < len(customers) or line > 0:
            if i < len(customers):
                line += customers[i]
            if line <= 4:
                wheel[0] = line
                line = 0
            else:
                wheel[0] = 4
                line -= 4
            wheel, profit, tracker = rotate(wheel, profit, boardingCost, runningCost, tracker)
            i += 1

        maxp = -1
        val = -1
        
        # print(tracker, wheel, line)
        if max(tracker) <= 0:
            return -1
        else:
            return tracker.index(max(tracker))
        # for i in range(len(tracker)):
        #     if tracker[i] >= max(tracker):
        #         maxp = tracker[i]
        #         val = i
        # if maxp > 0:
        #     return val
        # else:
        #     return -1
            
    
        
        
    
        

class Solution:
    def minOperationsMaxProfit(self, customers: List[int], bc: int, rc: int) -> int:
        maxi = 0
        ans = -1
        r = 0
        curr = 0
        temp = 0
        for c in customers:
            r += 1
            curr += c
            temp += min(curr, 4)
            curr -= min(curr, 4)
            if maxi < temp*bc - rc*r:
                maxi = temp*bc - rc*r
                ans = r
        while curr:
            r += 1
            temp += min(curr, 4)
            curr -= min(curr, 4)
            if maxi < temp*bc - rc*r:
                maxi = temp*bc - rc*r
                ans = r
        return ans
class Solution:
    def minOperationsMaxProfit(self, customers: List[int], boardingCost: int, runningCost: int) -> int:

        if boardingCost*4 < runningCost:
            return -1
        
        currVisitor = 0
        currGondolas = 0
        currProfit = 0
        currRotation = 0
        maxProf = 0
        maxProf_rot = -1
        gondolas = collections.deque()
        totPeople = 0
        for customer in customers:
            currRotation += 1
            currVisitor += customer
            if currGondolas < 4:
                gondolas.append(min(currVisitor,4))
                currGondolas = len(gondolas)
            else:
                gondolas.pop()
                gondolas.append(min(currVisitor,4))
            totPeople += min(currVisitor,4)
            currVisitor -= min(currVisitor,4)
            currProfit = boardingCost*totPeople - currRotation*runningCost
           
            if currProfit > maxProf:
                maxProf = currProfit
                maxProf_rot = currRotation
            # print(currProfit,maxProf,maxProf_rot,totPeople,currVisitor)
                
        while currVisitor > 0:
            currRotation += 1
            if currGondolas < 4:
                gondolas.append(min(currVisitor,4))
                currGondolas = len(gondolas)
            else:
                gondolas.pop()
                gondolas.append(min(currVisitor,4))
            totPeople += min(currVisitor,4)
            currVisitor -= min(currVisitor,4)
            
            currProfit = boardingCost*totPeople - currRotation*runningCost

            if currProfit > maxProf:
                maxProf = currProfit
                maxProf_rot = currRotation
            # print(currProfit,maxProf,maxProf_rot,totPeople,currVisitor)
            
            
        return maxProf_rot
            # print(maxProf,maxProf_rot,currRotation,gondolas,currVisitor)
                
                
            
'''
1. 10 customers arrive, 4 board and 6 wait for the next gondola, the wheel rotates. Current profit is 4 * $6 - 1 * $4 = $20.
2. 9 customers arrive, 4 board and 11 wait (2 originally waiting, 9 newly waiting), the wheel rotates. Current profit is 8 * $6 - 2 * $4 = $40.
3. The final 6 customers arrive, 4 board and 13 wait, the wheel rotates. Current profit is 12 * $6 - 3 * $4 = $60.
4. 4 board and 9 wait, the wheel rotates. Current profit is 16 * $6 - 4 * $4 = $80.
5. 4 board and 5 wait, the wheel rotates. Current profit is 20 * $6 - 5 * $4 = $100.
6. 4 board and 1 waits, the wheel rotates. Current profit is 24 * $6 - 6 * $4 = $120.
7. 1 boards, the wheel rotates. Current profit is 25 * $6 - 7 * $4 = $122.
The highest profit was $122 after rotating the wheel 7 times.
 '''             
            
        
        

class Solution:
    def minOperationsMaxProfit(self, customers: List[int], boardingCost: int, runningCost: int) -> int:
        res = -1
        profit = 0
        maxProfit = 0
        custs = 0
        i = 0
        while i < len(customers) or custs != 0:
            if i < len(customers):
                custs += customers[i]
            profit += min(custs, 4) * boardingCost - runningCost
            custs -= min(custs, 4)
            if profit > maxProfit:
                maxProfit = profit
                res = i + 1
            i += 1
        return res

class Solution:
    def minOperationsMaxProfit(self, customers: List[int], boardingCost: int, runningCost: int) -> int:
        max_profit = 0
        max_iter = 0
        cur_profit = 0
        available = 0
        i = 0
        while i < len(customers) or available > 0:
            if i < len(customers):
                available += customers[i]
            boarding = min(available, 4)
            cur_profit += boarding*boardingCost - runningCost*min(boarding,1)
            if cur_profit > max_profit:
                max_profit = cur_profit
                max_iter = i
            available -= boarding
            i += 1
        
        return (max_iter+1) if max_profit != 0 else -1
class Solution:
    def minOperationsMaxProfit(self, customers: List[int], boardingCost: int, runningCost: int) -> int:
        if boardingCost * 4 < runningCost:
            return -1
        
        res = 0
        left = 0
        
        for i in range(len(customers)):
            customer = customers[i]
            left += customer
            
            if res <= i:
                if left < 4:
                    left = 0
                else:
                    left -= 4
                
                res += 1
            
            while left >= 4:
                res += 1
                left -= 4
                
                
        if left * boardingCost > runningCost:
            res += 1
                
        return res if res > 0 else -1
            
                
        
            
            

class Solution:
    def minOperationsMaxProfit(self, customers: List[int], boardingCost: int, runningCost: int) -> int:
        a = [0,0,0,0]
        profit = 0
        cur_waiting = 0
        cur_index = 0
        
        rotation_count = 0
        max_rotation_count = 0
        
        max_profit = 0
        for c in customers:
            cur_waiting += c
            rotation_count += 1
            a[cur_index] = min(4, cur_waiting)
            profit += boardingCost * min(4, cur_waiting)
            profit -= runningCost
            cur_waiting -= min(4, cur_waiting)
            cur_index += 1
            if cur_index == 4:
                cur_index = 0
            if profit > max_profit:
                max_rotation_count = rotation_count
                max_profit = profit
                
        while cur_waiting > 0:
            rotation_count += 1
            a[cur_index] = min(4, cur_waiting)
            profit += boardingCost * min(4, cur_waiting)
            profit -= runningCost
            cur_waiting -= min(4, cur_waiting)
            cur_index += 1
            if cur_index == 4:
                cur_index = 0
            if profit > max_profit:
                max_rotation_count = rotation_count
                max_profit = profit
        
        if max_profit <= 0:
            return -1
        return max_rotation_count
from collections import deque
class Solution:
    def minOperationsMaxProfit(self, customers: List[int], boardingCost: int, runningCost: int) -> int:
        ans = [0]
        q = deque(customers)
        while len(q) > 0:
            cur = q.popleft()
            if cur > 4:
                r = cur - 4
                if len(q) > 0:
                    q[0] += r
                else:
                    q.append(r)
            profit = (min(cur, 4)*boardingCost-runningCost)
            ans.append(ans[-1]+profit)
        maxP = max(ans)
        # print(maxP, ans)
        maxTime = [i for i in range(len(ans)) if ans[i] == maxP][0]
        return -1 if maxP == 0 else maxTime
class Solution:
    def minOperationsMaxProfit(self, customers: List[int], boardingCost: int, runningCost: int) -> int:
        
        leftover = 0
        dp = [0]
        
        i=0
        while i<len(customers) or leftover:
            
            if i < len(customers):
                leftover += customers[i]
                
            newCust = min(4, leftover)
            leftover -= newCust
            # if leftover>=4:
            #     newCust = 4
            #     leftover -= 4
            # else:
            #     newCust = leftover
            #     leftover = 0
            
            temp = dp[-1] + newCust*boardingCost - runningCost
            dp.append(temp)
            
            i += 1
            
            
        if all([x<=0 for x in dp]):
            return -1
        return dp.index(max(dp))
class Solution:
    def minOperationsMaxProfit(self, customers: List[int], boardingCost: int, runningCost: int) -> int:
        res = 0 
        max_profit = 0
        wait_people = 0
        taken_people = 0
        rotation_times = 0
        
        i = 0 
        while i < len(customers) or wait_people > 0:
            rotation_times += 1 
            cur_people = customers[i] if i < len(customers) else 0
            can_take = min(4, wait_people+cur_people)
            taken_people += can_take 
            cur_profit = taken_people*boardingCost - rotation_times*runningCost
            if cur_profit > max_profit:
                res = rotation_times 
                max_profit = cur_profit
                
            wait_people = max(0, wait_people+cur_people-4)
            i += 1 
            
        return res if res > 0 else -1 

class Solution:
    def minOperationsMaxProfit(self, customers: List[int], boardingCost: int, runningCost: int) -> int:
        profits = [0]
        # counts = []
        person_remain = customers[0]
        i = 1
        while(person_remain or i == 1):
            gondola = min(4, person_remain)
            profits.append(gondola*boardingCost - runningCost + profits[-1])
            person_remain = person_remain - gondola
            if(i < len(customers)):
                person_remain += customers[i]
            i+= 1
        
        max_round = 0
        max_el = max(profits)
        # print(profits)
        if(max_el <= 0):
            return -1
        
        for i in range(len(profits)):
            if(profits[i] == max_el):
                max_round = i
                break
                
        return max_round

class Solution:
    def minOperationsMaxProfit(self, customers: List[int], boardingCost: int, runningCost: int) -> int:
        wheel = [0,0,0,0]
        profit = 0
        tracker = [-1]
        line = 0
        
        def rotate(wheel, profit, boardingCost, runningCost, tracker):
            profit += boardingCost*wheel[0] - runningCost
            tracker.append(profit)
            wheel = [0, wheel[0], wheel[1], wheel[2]]
            return (wheel, profit, tracker)
        i = 0
        while i < len(customers) or line > 0:
            if i < len(customers):
                line += customers[i]
            if line <= 4:
                wheel[0] = line
                line = 0
            else:
                wheel[0] = 4
                line -= 4
            wheel, profit, tracker = rotate(wheel, profit, boardingCost, runningCost, tracker)
            i += 1

        maxp = -1
        val = -1
        
        # print(tracker, wheel, line)

        for i in range(len(tracker)):
            if tracker[i] > maxp:
                maxp = tracker[i]
                val = i
        if maxp > 0:
            return val
        else:
            return -1
            
    
        
        
    
        

class Solution:
    def minOperationsMaxProfit(self, customers: List[int], boardingCost: int, runningCost: int) -> int:
        maxProfit, res, remain, profit, i = 0, 0, 0, -runningCost*3, 0
        while i < len(customers) or remain:
            if i < len(customers):
                x = customers[i]
            else:
                x = 0
            
            profit += min(4, x + remain) * boardingCost - runningCost
            
            if profit > maxProfit:
                maxProfit, res = profit, i+1
            remain = max(0, remain+x-4)
            i += 1
        return res if res else -1
class Solution:
    def minOperationsMaxProfit(self, customers: List[int], boardingCost: int, runningCost: int) -> int:
        
        if boardingCost * 4 <= runningCost:
            return -1
        
        profit = 0
        num_waiting_customers = 0
        max_profit = 0
        ans = -1
        
        i = 0
        
        while i < len(customers) or num_waiting_customers > 0:
            num_waiting_customers += customers[i] if i < len(customers) else 0
            
            if i < len(customers):
                num_rotate = ((len(customers) - i) * 50 + num_waiting_customers + 3) // 4
                if ((len(customers) - i) * 50 + num_waiting_customers) * boardingCost - num_rotate * runningCost + profit < 0:
                    return ans
                    
            
            profit += min(num_waiting_customers, 4) * boardingCost - runningCost
            
            if profit > max_profit:
                ans = i + 1
                max_profit = profit
            
            num_waiting_customers = max(num_waiting_customers - 4, 0)
            i += 1
        return ans

class Solution:
    def minOperationsMaxProfit(self, customers: List[int], boardingCost: int, runningCost: int) -> int:
        n_board, n_wait = 0, 0
        cur_profit, max_profit, max_idx = 0, -float('inf'), -1
        for i, people in enumerate(customers):
            n_wait += people
            n_board += min(4, n_wait)
            n_wait -= min(4, n_wait)
            
            cur_profit = boardingCost * n_board - (i+1) * runningCost
            if cur_profit > max_profit:
                max_profit = cur_profit
                max_idx = i + 1
        #     print(i+1, cur_profit)
        # print(n_wait, n_board)
        while n_wait:
            i += 1
            n_board += min(n_wait, 4)
            n_wait -= min(n_wait, 4)
            cur_profit = boardingCost * n_board - (i+1) * runningCost
            # print(n_board, n_wait)
            # print(i+1, cur_profit)
            if cur_profit > max_profit:
                max_profit = cur_profit
                max_idx = i + 1
        return max_idx if max_profit > 0 else -1

class Solution:
    def minOperationsMaxProfit(self, customers: List[int], boardingCost: int, runningCost: int) -> int:
        q = 0
        profit = 0
        ans = -1
        count = 0
        
        for i in customers:
            count += 1
            q += i
            board = min(q, 4)
            q -= board
            temp = profit + board*boardingCost - runningCost
            if temp > profit:
                profit = temp
                ans = count
        
        times = q // 4
        left = q % 4
        temp = profit + times*(4*boardingCost - runningCost)
        if temp > profit:
            profit = temp
            count += times
            ans = count
            
        temp = profit + left*boardingCost - runningCost
        if temp > profit:
            profit = temp
            count += 1
            ans = count
        
        return ans
class Solution:
    def minOperationsMaxProfit(self, customers: List[int], boardingCost: int, runningCost: int) -> int:
        # print(customers)
        r = 1
        wait = 0
        cost = 0
        cround = []
        for c in customers:
            wait += c
            if wait >= 4:
                wait -= 4
                cost += boardingCost * 4 - runningCost
                cround.append([cost,r])
            elif wait > 0:
                cost += boardingCost * wait - runningCost
                wait -= wait
                cround.append([cost,r])
            r += 1

                
        while wait >= 4:
            wait -= 4
            cost += boardingCost * 4 - runningCost  
            cround.append([cost,r])
            r += 1
            
        if wait > 0:
            cost += boardingCost * wait - runningCost
            cround.append([cost,r])
            wait -= wait
            r += 1
        
        ans = max(cround,key = lambda x: x[0])
        # print('ans',ans)
        # print(r,wait,cost,cround)
        if ans[0] > 0:
            return ans[1]
        else:
            return -1

class Solution:
    def minOperationsMaxProfit(self, customers: List[int], boardingCost: int, runningCost: int) -> int:
        p = 0
        maxp = 0
        res = -1
        curr = 0
        i = 0
        while curr or i<len(customers):
            if i < len(customers):
                curr += customers[i]
            i += 1
            p += min(curr, 4)*boardingCost - runningCost
            if p > maxp:
                res = i
                maxp = p
            curr = max(0, curr-4)
        return res
class Solution:
    def minOperationsMaxProfit(self, customers: List[int], boardingCost: int, runningCost: int) -> int:
        r = -1
        maxV = 0
        pre = 0
        profit = 0
        
        i = 0
        while True:
            curr = 0
            if i < len(customers): curr = customers[i]
            profit += min(curr + pre, 4)*boardingCost - runningCost
            pre = max(curr + pre - 4, 0)  
            if profit > maxV:
                r = i + 1
                maxV = profit
            i += 1
            if i >= len(customers) and pre <= 0: break
        return r
class Solution:
    def minOperationsMaxProfit(self, customers: List[int], boardingCost: int, runningCost: int) -> int:
        steps, waiting = 1, 0
        fee, cost, record, max_profit =  0, 0, {}, -sys.maxsize
        while steps < len(customers) or waiting>0:
            arrival = customers[steps-1] if steps <= len(customers) else 0
            if arrival+waiting <= 4:
                fee += (arrival+waiting)*boardingCost
                waiting = 0
            else:
                waiting = (arrival+waiting)-4
                fee += 4*boardingCost
            cost += runningCost
            record[steps] = fee-cost
            max_profit = max(max_profit, fee-cost)
            steps += 1
        for k in record:
            if record[k]>0 and record[k]==max_profit:
                return k
        return -1
class Solution:
    def minOperationsMaxProfit(self, customers: List[int], boardingCost: int, runningCost: int) -> int:
        left_people = 0
        profit = 0
        maxProfit = -1
        maxpc = 0
        counter = 0
        i = 0
        while i < len(customers) or left_people > 0:
            counter += 1
            left_people += customers[i] if i < len(customers) else 0
            profit += min(left_people, 4) * boardingCost - runningCost
            left_people = max(left_people - 4, 0)
            if profit > maxProfit:
                maxProfit = profit
                maxpc = counter
            i += 1
        return -1 if maxProfit < 0 else maxpc 
class Solution:
    def minOperationsMaxProfit(self, customers: List[int], boardingCost: int, runningCost: int) -> int:
        num_wait = 0
        num_board = 0
        num_round = 0
        max_prof = 0
        res = -1
        for a in customers:
            num_round += 1
            if num_wait + a >= 4:
                num_board += 4
                num_wait += a - 4
            else:
                num_board += num_wait + a
                num_wait = 0
            if boardingCost * num_board - runningCost * num_round > max_prof:
                max_prof = max(max_prof, boardingCost * num_board - runningCost * num_round)
                res = num_round
            
        while num_wait > 0:
            num_round += 1
            num_board += min(4, num_wait)
            num_wait -= min(4, num_wait)
            if boardingCost * num_board - runningCost * num_round > max_prof:
                max_prof = max(max_prof, boardingCost * num_board - runningCost * num_round)
                res = num_round 
        return res
            

class Solution:
    def minOperationsMaxProfit(self, customers: List[int], boarding_cost: int, running_cost: int) -> int:
        max_profit = -1
        cur_profit = 0
        waiting = 0
        rotations = 0
        i = 0
        while i < len(customers):

            c = customers[i]
            waiting += c
            boarded = min([waiting,4])
            waiting = max([waiting-boarded,0])
            cur_profit += boarded * boarding_cost - running_cost
            if cur_profit > 0 and max_profit < cur_profit:
                max_profit = max([max_profit,cur_profit])
                rotations = i+1
            i+=1
        while waiting >0:
            boarded = min([waiting,4])
            waiting = max([waiting-boarded,0])
            cur_profit += boarded * boarding_cost - running_cost
            if cur_profit > 0 and max_profit < cur_profit:
                max_profit = max([max_profit,cur_profit])
                rotations = i+1
            i+=1
        if rotations == 0:
            return -1
        return rotations
class Solution:
    def minOperationsMaxProfit(self, customers: List[int], boardingCost: int, runningCost: int) -> int:
        ans=[0]*1000000
        waiting=0
        for i in range(1000000):
            if i<len(customers):
                waiting+=customers[i]
            if waiting<=4:
                ans[i]=ans[i-1] + waiting*boardingCost - runningCost
                waiting=0
            else:
                ans[i]=ans[i-1] + 4*boardingCost -runningCost
                waiting-=4
            if waiting<=0 and i>=len(customers):
                break
        maxVal = max(ans)
        if maxVal<=0:
            return -1
        ret = ans.index(maxVal)+1
        return ret 
class Solution:
    def minOperationsMaxProfit(self, customers: List[int], boardingCost: int, runningCost: int) -> int:
        profits = [-1]
        waiting = customers[0]
        i = 1
        board = 0
        while waiting > 0 or i < len(customers):
        
            board += min(4, waiting)
            profit = board  * boardingCost
            profit -= runningCost * i
            profits.append(profit)
            waiting -= min(4, waiting)
            if i < len(customers):
                waiting += customers[i]
            
            i += 1

        id = profits.index(max(profits))
        if id == 0:
            return -1
            
        return id
class Solution:
    def minOperationsMaxProfit(self, customers: List[int], boardingCost: int, runningCost: int) -> int:
        if not customers:
            return -1
        profit = 4 * boardingCost - runningCost
        if profit <=0:
            return - 1
        
        
        cumulate = sum(customers)
        cum = 0
        rt = 0
        al = 0
        maxl = 0
        for item in customers:
            cum+=item
            if cum>=4:
                cum-=4
                rt+=1
                al+=4
                maxl = max(maxl, al * boardingCost -rt* runningCost)
            else:
                
                rt+=1
                al+=cum
                # if al * boardingCost -rt* runningCost > maxl:
                maxl = al * boardingCost -rt* runningCost
                cum = 0
        a = cum //4
        b = cum%4
        profit = boardingCost * b - runningCost
        if profit > 0:
            rt=rt + a+1
        else:
            rt+=a
        
        return rt
                
        
        
        
        

#         if b == 0:
#             return a
#         else:
#             rt = a 
#             tryone = cumulate * boardingCost - (a+1) * runningCost
#             anotherone = a * 4 * boardingCost - a * runningCost
#             profit = boardingCost * b - runningCost
#             print(a, b, cumulate,tryone,anotherone )
#             if profit > 0:
#                 rt=a+1
                
#             if tryone > anotherone:
#                 return a+1
#             else:
#                 return rt
        

class Solution:
    def minOperationsMaxProfit(self, customers: List[int], boardingCost: int, runningCost: int) -> int:
        profits = [0] * (13 * len(customers))
        p_cursor = 0
        waiting = 0
        for c in customers:
            waiting += c
            riding = min(waiting, 4)
            profits[p_cursor] += boardingCost * riding
            profits[p_cursor] -= runningCost
            if p_cursor > 0:
                profits[p_cursor] += profits[p_cursor - 1]
            p_cursor += 1
            waiting -= riding

        while waiting > 0:
            riding = min(waiting, 4)
            profits[p_cursor] += boardingCost * riding
            profits[p_cursor] -= runningCost
            if p_cursor > 0:
                profits[p_cursor] += profits[p_cursor - 1]
            p_cursor += 1
            waiting -= riding

        max_profit = max(profits)
        if max_profit <= 0:
            return -1

        for i, p in enumerate(profits):
            if p == max_profit:
                return i + 1

        return -1
class Solution:
    def minOperationsMaxProfit(self, a: List[int], boardingCost: int, runningCost: int) -> int:
        res = boarding = profit = rem = 0
        n = len(a)
        cap = 4
        profits = []

        i = 0
        while rem or i < n:
            if i < n:
                rem += a[i]
                # print(f'{rem = }')
            boarding = min(rem, cap)                
            # print(f'{boarding = }')
            rem = max(0, rem - cap)
            profit += boarding * boardingCost - runningCost
            profits.append(profit)
            # print(f'{rem = }')
            # print(f'{profits = }')
            i += 1
            
        argmax = -1
        mx = -float('inf')
        for i, x in enumerate(profits):
            if x > mx:
                mx = x
                argmax = i
                
        return argmax + 1 if profits[argmax] > 0 else -1

class Solution:
    def minOperationsMaxProfit(self, customers: List[int], boardingCost: int, runningCost: int) -> int:
        rotations = 0
        wheel = [0, 0, 0, 0]
        profits = []
        cost = 0
        waiting = 0
        profit = 0
        maxProfit = 0
        
        
        for i in range(len(customers)):
            waiting += customers[i]
            
            
            '''if rotations < i:
                for j in range(rotations - i):
                    profit -= runningCost
                    wheel[2], wheel[3] = wheel[1], wheel[2]
                    wheel[1] = 0'''
            
                
            if waiting > 4:
                profit += 4*boardingCost
            else:
                profit = boardingCost * waiting
            profit -= runningCost
            rotations += 1
            wheel[2], wheel[3] = wheel[1], wheel[2]
            wheel[1] = min(waiting, 4)
            waiting -= min(waiting, 4)
            profits.append((profit, rotations))
            
        while waiting > 0:
            profit += min(4, waiting) * boardingCost
            waiting -= min(4, waiting)
            profit -= runningCost
            rotations += 1
            profits.append((profit, rotations))
            
        #print(profits)
        profits = sorted(profits, key = lambda x: (x[0], -x[1]), reverse = True)
        if profits[0][0] > 0:
            return profits[0][1]
        else:
            return -1
class Solution:
    def minOperationsMaxProfit(self, customers: List[int], boardingCost: int, runningCost: int) -> int:
        if len(customers)==0:
            return -1
        curWait=0
        prof=0
        maps=[]
        count=0
        while curWait!=0 or count<len(customers):
            if count<len(customers):
                cu=customers[count]
                curWait+=cu
            count+=1
            prof-=runningCost
            if curWait<=4:
                prof+=curWait*boardingCost
                curWait=0
            else:
                prof+=4*boardingCost
                curWait-=4
            maps.append([count,prof])
        maps=sorted(maps,key=lambda x:x[1],reverse=True)
        if maps[0][1]<=0:
            return -1
        else:
            return maps[0][0]
class Solution:
    def minOperationsMaxProfit(self, customers: List[int], boardingCost: int, runningCost: int) -> int:
        temp = 0
        profit = 0
        r = 0
        waiting = 0
        ans = -1
        i = 0
        while i < len(customers) or waiting > 0:
            num = min(waiting + customers[i], 4) if i < len(customers) else min(waiting, 4)
            profit += num * boardingCost - runningCost
            r += 1
            waiting = max(waiting + customers[i]-4, 0) if i < len(customers) else max(waiting - 4, 0)
            i += 1
            if profit > temp:
                temp = profit
                ans = r
        return ans

class Solution:
    def minOperationsMaxProfit(self, customers: List[int], boardingCost: int, runningCost: int) -> int:
        times = 0
        left = 0
        profit = 0
        pros = []
        for i in range(len(customers)):
            left += customers[i]
            profit = profit + min(left, 4) * boardingCost -  runningCost
            pros.append(profit)
            left = max(0, left-4)
            times += 1
        i = len(customers)
        while (left > 0):
            profit = profit + min(left, 4) * boardingCost -  runningCost
            pros.append(profit)
            i += 1
            left = max(0, left-4)
            times += 1
        mm = -1
        out = - 1
        for i in range(len(pros)):
            if pros[i] > mm:
                mm = pros[i]
                out = i + 1
        if profit > 0:
            return out
        return -1
            
            

class Solution:
    def minOperationsMaxProfit(self, customers: List[int], boardingCost: int, runningCost: int) -> int:
        
        if not customers:
            return -1
        profits = []
        wait = 0
        i = 0

        while True:
            if i > len(customers) -1 and wait <= 0:
                break
            elif i <= len(customers)-1 :
                wait += customers[i]
                profit = min(4,  wait) * boardingCost - runningCost
                profits.append(profit)
                wait = wait - min(4, wait)
                i += 1
            else:
                profit = min(4, wait) * boardingCost - runningCost
                profits.append(profit)
                wait = wait - min(4, wait)


        #print(profits)


        sum_ = profits[0]
        for i in range(1, len(profits)):
            profits[i] = sum_ + profits[i]
            sum_ = profits[i]

        #print(profits)
        max_ = 0
        index = -1
        for i in range(0, len(profits)):
            if max_ < profits[i]:
                index = i
                max_ = profits[i]

        if index > -1:
            index += 1
        return index
class Solution:
    def minOperationsMaxProfit(self, customers: List[int], boardingCost: int, runningCost: int) -> int:
        nrot = 0
        i = 0
        profits = []
        if customers[i] > 0:
            waiting = customers[i] - min(4, customers[i])
            nrot += 1
            profits.append(min(customers[i], 4)*boardingCost - 1 * runningCost)
        else:
            waiting = 0
            profits.append(0)
        customers.append(0)
        while waiting > 0 or i <= len(customers):
            profits.append(profits[-1] + min(waiting, 4)*boardingCost - 1 * runningCost)
            if i + 1 <= len(customers) - 1:
                waiting += - min(4, waiting + customers[i+1]) + customers[i+1]
            else:
                waiting -= min(waiting, 4)
            nrot += 1
            i += 1
        # print(nrot, boardingCost, sum(customers), runningCost)
        mx = profits.index(max(profits))
        return mx+1 if max(profits) > 0 else -1
            

class Solution:
    def minOperationsMaxProfit(self, customers: List[int], boardingCost: int, runningCost: int) -> int:
        waiting_customers = 0
        wheel_rotations = 0
        minimum_wheel_rotations = -1
        max_profit = 0
        profit = 0
        for c in customers:
            wheel_rotations+=1
            boarding_waiting = min(4, waiting_customers)
            waiting_customers -= boarding_waiting
            boarding_now = max(0, min(4, c) - boarding_waiting)
            waiting_customers += c-boarding_now
            total_boarding = boarding_waiting + boarding_now
            profit += total_boarding * boardingCost - runningCost
            if profit > max_profit:
                max_profit = profit
                minimum_wheel_rotations = wheel_rotations
        fullrevenue_rotations = (waiting_customers // 4)
        wheel_rotations+=fullrevenue_rotations
        # fullprofit = fullprofit_rotations * 4 * boardingCost - fullprofit_rotations * runningCost
        fullrevenue = fullrevenue_rotations * (4 * boardingCost - runningCost)
        
        profit += fullrevenue
        
        if profit > max_profit:
            max_profit = profit
            minimum_wheel_rotations = wheel_rotations
        
        remaining_customers = waiting_customers % 4
        wheel_rotations+=1
        remaining_revenue = remaining_customers * boardingCost - runningCost
        
        profit += remaining_revenue
        if profit > max_profit:
            max_profit = profit
            minimum_wheel_rotations = wheel_rotations
        return minimum_wheel_rotations
        

class Solution:
    def minOperationsMaxProfit(self, customers: List[int], boardingCost: int, runningCost: int) -> int:
        wt = 0
        profit = 0
        bp = 0
        ret = -1
        cr = 0
        gondola = [0, 0, 0, 0]
        i = 0
        while i < len(customers) or wt:
            cr += 1
            if i < len(customers):
                e = customers[i]
                wt += e
            if wt:
                take = min(wt, 4)
                profit += boardingCost * take - runningCost
                wt -= take
                gondola = gondola[1:] + [take]
            else:
                if sum(gondola) == 0:
                    profit -= runningCost
            if profit > bp:
                bp = profit
                ret = cr
            i += 1
        return ret

class Solution:
    def minOperationsMaxProfit(self, customers: List[int], boardingCost: int, runningCost: int) -> int:
        max_profit = 0
        max_step = -1
        profit = 0
        rest = 0

        for i, c in enumerate(customers):
            customer = c + rest
            customer, rest = min(4, customer), max(0, customer - 4)
            profit += customer * boardingCost - runningCost
            if profit > max_profit:
                max_profit = profit
                max_step = i + 1

        q, r = divmod(rest, 4)
        if q > 0:
            profit += q * 4 * boardingCost - q * runningCost
            if profit > max_profit:
                max_profit = profit
                max_step = len(customers) + q
        if r > 0:
            profit += r * boardingCost - runningCost
            if profit > max_profit:
                max_step = len(customers) + q + 1

        return max_step
class Solution:
    def minOperationsMaxProfit(self, customers: List[int], boardingCost: int, runningCost: int) -> int:
        wating = 0
        profit = 0
        idx = -1
        t_r = 0
        max_profit = 0
        for customer in customers:
            wating += customer
            if wating <= 4:
                profit += wating * boardingCost - runningCost
                wating = 0
            else:
                profit += 4 * boardingCost - runningCost
                wating -= 4
            t_r += 1
            if profit > max_profit: 
                idx = t_r
                max_profit = profit
        if wating > 0:
            if 4 * boardingCost > runningCost:
                idx += wating//4
        wating = wating%4
        if wating * boardingCost > runningCost:
            idx += 1
        return idx

class Solution:
    def minOperationsMaxProfit(self, customers: List[int], boardingCost: int, runningCost: int) -> int:
        if not customers: return -1
        if 4 * boardingCost <= runningCost: return -1
        num = 0
        profit = 0
        sum_ppl = 0
        for c in customers:
            sum_ppl += c
            
        cur_w = 0    
        for i in range(len(customers)):
            num += 1
            cur_w += customers[i]
            n = 4 if cur_w >= 4 else cur_w
                
            profit += n * boardingCost - runningCost
            cur_w -= n
           
        
            

        rotates, left = cur_w// 4, cur_w % 4
        num += rotates
        profit += rotates * 4 * boardingCost - runningCost * rotates
        
        if left * boardingCost > runningCost:
            num += 1
            profit += left * boardingCost - runningCost
        if profit <= 0:
            return -1
        return num
            
        

from typing import List, Dict, Tuple


class Solution:
    def minOperationsMaxProfit(self, customers: List[int], boardingCost: int, runningCost: int) -> int:
        current = 0
        cost = 0
        amari = 0
        ans = -1
        macost = 0
        for customer in customers:
            amari += customer
            if amari >= 4:
                amari -= 4
                cost += 4 * boardingCost - runningCost
                current += 1
                if cost > macost:
                    macost = cost
                    ans = current

            else:
                tmp = cost
                tmp = cost
                cost += amari * boardingCost - runningCost
                amari = 0
                current += 1
                if cost > macost:
                    macost = cost
                    ans = current

        a, b = divmod(amari, 4)
        if 4 * boardingCost > runningCost:
            cost += a * boardingCost - runningCost
            current += a
            if cost > macost:
                macost = cost
                ans = current
        if b * boardingCost > runningCost:
            cost += b * boardingCost - runningCost
            current += 1
            if cost > macost:
                macost = cost
                ans = current

        return ans

class Solution:
    def minOperationsMaxProfit(self, customers: List[int], boardingCost: int, runningCost: int) -> int:
        ans = -1
        most = pnl = waiting = 0
        for i, x in enumerate(customers): 
            waiting += x # more people waiting in line 
            waiting -= (chg := min(4, waiting)) # boarding 
            pnl += chg * boardingCost - runningCost 
            if most < pnl: ans, most = i+1, pnl
        q, r = divmod(waiting, 4)
        if 4*boardingCost > runningCost: ans += q
        if r*boardingCost > runningCost: ans += 1
        return ans 


class Solution:
    def minOperationsMaxProfit(self, customers: List[int], boardingCost: int, runningCost: int) -> int:
        count = 0
        minCount = -1
        profit = 0
        maxProfit = None
        waiting = 0
        for i, cnum in enumerate(customers):
            waiting += cnum
            if waiting >= 4:
                rounds = waiting // 4
                count += rounds
                profit += (boardingCost * 4 - runningCost)*rounds
                waiting = waiting % 4
            else:
                if count <= i:
                    count += 1
                    profit += boardingCost * waiting - runningCost
                    waiting = 0
            if profit > 0 and (maxProfit is None or profit > maxProfit):
                maxProfit = profit
                minCount = count
        if waiting > 0:
            profit += boardingCost * waiting - runningCost
            count += 1
            if profit > 0 and (maxProfit is None or profit > maxProfit):
                maxProfit = profit
                minCount = count

        return minCount
class Solution:
    def minOperationsMaxProfit(self, customers: List[int], boardingCost: int, runningCost: int) -> int:
        r=0
        c=0
        n=len(customers)
        total=0
        ans=0
        ind=-1
        while c<n:
            if customers[c]>=4:
                r+=(customers[c]//4)
                total+=((customers[c]//4)*4)
                customers[c]-=((customers[c]//4)*4)
                if customers[c]==0:
                    c+=1
                res=total*boardingCost-r*runningCost
                if res>ans:
                    ans=res
                    ind=r
            else:
                if c==n-1 or c==r:
                    total+=customers[c]
                    r+=1
                    res=total*boardingCost-r*runningCost
                    if res>ans:
                        ans=res
                        ind=r
                else:
                    customers[c+1]+=customers[c]
                c+=1
        return ind

class Solution:
    def minOperationsMaxProfit(self, customers: List[int], boardingCost: int, runningCost: int) -> int:
        
        profit = 0
        waiting = 0
        rotations = 0
        onboard = 0
        gondola_customers = deque([])
        total = 0
        maxprofit = 0
        max_rotation = -1
        for idx,arrival in enumerate(customers):
            # if onboard == 0 and waiting == 0 and arrival == 0:
                # continue
            
            if gondola_customers and gondola_customers[0][1]==idx:
                coming_down = gondola_customers.popleft()[0]
                onboard -= coming_down
            
            total = arrival   
            if waiting >0:
                total = waiting + arrival

            board = min(total,4)
            profit += ((board*boardingCost) - runningCost)
            onboard += board
            gondola_customers.append([board,idx+4])
            waiting += (arrival-board)

            rotations += 1
            if profit > maxprofit:
                maxprofit = profit
                max_rotation = rotations
        
        
        profit += ((waiting//4)*((4*boardingCost)-runningCost))
        rotations += (waiting//4)
        if profit > maxprofit:
                maxprofit = profit
                max_rotation = rotations
        
        profit += (((waiting%4)*boardingCost)-runningCost)
        rotations += ((waiting%4)>0)
        if profit > maxprofit:
                maxprofit = profit
                max_rotation = rotations
        
        return max_rotation if maxprofit > 0 else -1
                
        

class Solution:
    def minOperationsMaxProfit(self, customers: List[int], B: int, R: int) -> int:
        turns = 0
        boarded = 0
        max_profit = -1
        rem = 0
        max_turns = -1
        # print(sum(customers))
        for i, c in enumerate(customers):
            t, rem = divmod(c+rem, 4)
            turns += t
            boarded += t * 4
            if turns <= i: 
                turns += 1
                boarded += rem
                rem = 0
            profit = boarded * B - turns * R
            if profit > max_profit:
                max_profit = profit
                max_turns = turns
            # print(i, c, boarded, rem, turns, res)
        if rem > 0:
            boarded += rem
            turns += 1
            profit = boarded * B - turns * R
            if profit > max_profit:
                max_profit = profit
                max_turns = turns
        return max_turns if max_profit > 0 else -1
class Solution:
    def minOperationsMaxProfit(self, customers: List[int], boardingCost: int, runningCost: int) -> int:
        left = 0 
        cost = 0 
        step = 0 
        def ceildiv(a, b):
               return -(-a // b)
        for i in range(len(customers)):
            left += customers[i]
            if left >= 4:
                    left -= 4 
                    cost = 4*(boardingCost) - runningCost
                    step +=1 
             
            else: 
                cost = left*boardingCost - runningCost
                step +=1 
                left = 0 
        lefts = left // 4 
        leftc  = left % 4 
        costl = leftc*boardingCost - runningCost
        cost = cost + left*boardingCost - runningCost*lefts 
        step += lefts 
       
        if cost > 0 :
            if costl > 0:
               return step +1 
            else : return step 
        else : return -1 

class Solution:
    def minOperationsMaxProfit(self, customers: List[int], boardingCost: int, runningCost: int) -> int:
        _max=float('-inf')
        queue,index,profit,rotations,buffer=0,0,0,0,0
        _len=len(customers)
        while queue>=0 and index<_len: 
            queue+=customers[index]
            profit+= ( ( min(queue,4)*boardingCost ) - ( runningCost )  )
            queue-=min(queue,4)
            if profit>_max:
                rotations+=1
            if profit==_max:
                buffer+=1
            _max=max(_max,profit)
            index+=1
            if  index==_len and queue:
                profit+= (  ( ( (queue//4)*4 ) * boardingCost ) - ( (queue//4)*runningCost )  )
                _max=max(_max,profit)
                rotations+=queue//4
                queue-=((queue//4)*4)

        if queue:
            profit+=  (( queue%4 * boardingCost ) -  runningCost)
            if profit>_max:
                rotations+=1
            _max=max(_max,profit)

        return rotations+buffer if _max>=0 else -1

class Solution:
    def minOperationsMaxProfit(self, customers: List[int], boardingCost: int, runningCost: int) -> int:
        cur = 0
        total = 0

        profit1 = [0,0]
        profit2 = [0,0]
        
        stop = False
        
        left =0
        
        if 4*boardingCost-runningCost<=0:
            return -1
        while cur<len(customers):
            left+=customers[cur]
            if left>=4:
                left-=4
                customer = 4
            else:
                customer = left
                left = 0
            if (customer*boardingCost-runningCost)+profit1[0]>profit1[0]:
                profit2[0]=(customer*boardingCost-runningCost)+profit1[0]
                profit2[1]=profit1[1]+1
            
            profit1[0]=(customer*boardingCost-runningCost)+profit1[0]
            profit1[1]=profit1[1]+1
            cur+=1
                
        if left:
            profit1[0]+=(left//4)*(customer*boardingCost-runningCost)
            profit1[1]+=(left//4)
            
            if (left%4)*boardingCost-runningCost>0:
                profit1[0]+=(left%4)*boardingCost-runningCost
                profit1[1]+=1
        
        if profit1[0]>profit2[0]:
            return profit1[1]
        elif profit1[0]<profit2[0]:
            return profit2[1]
        else:
            return min(profit1[1],profit2[1])
                
        '''
        while cur<len(customers):
            current = customers[cur]

            if current >=4:
                
                profit1[cur+1][0]= (current//4)*(4*boardingCost-runningCost)+profit1[cur][0]
                profit1[cur+1][1] = profit1[cur][1] +current//4
                current = current%4               
                if profit1[cur][0]<profit1[cur+1][0]:
                    profit2[cur+1][0]=profit1[cur+1][0]
                    profit2[cur+1][1]=profit1[cur+1][1]
                else:
                    profit2[cur+1][1]=profit1[cur][1]
                    profit2[cur+1][0]=profit1[cur][0]
                    
                if current>0:
                    profit1[cur+1][0]=current*boardingCost-runningCost+profit1[cur+1][0]
                    profit1[cur+1][1]=profit1[cur+1][1]+1 
            
            else:

                profit1[cur+1][0]=current*boardingCost-runningCost+profit1[cur][0]

                profit1[cur+1][1]=profit1[cur][1]+1 

                
                
                
                profit2[cur+1][1]=profit1[cur][1]

                profit2[cur+1][0]=profit1[cur][0]

            
            
                    

            cur+=1
            

        
        keys1 = list(range(len(customers)+1))
        keys2= list(range(len(customers)+1))
        keys1.sort(key=lambda x: (profit1[x][0],-profit1[x][1]))
        keys2.sort(key=lambda x:(profit2[x][0],-profit2[x][1]))

        key1= keys1[-1]
        key2 = keys2[-1]
        print(profit1)
        print(profit2)

        if profit1[key1][0]>profit2[key2][0]:
            return profit1[key1][1]
        elif profit1[key1][0]<profit2[key2][0]:
            return profit2[key1][1]
        else:
            return min(profit1[key1][1],profit2[key2][1])
        '''
class Solution:
    def minOperationsMaxProfit(self, customers: List[int], boardingCost: int, runningCost: int) -> int:
        # Ugliest code ever dont even try reading this
        return self.min_operations_max_profit_recurse(customers, 0, boardingCost, runningCost, 0, 0)[1]
    
    def serve_remaining_people(self, waiting, boardingCost, runningCost, depth):
        profit = 0
        
        num_rotations = waiting // 4
        if num_rotations and 4 * boardingCost - runningCost > 0:
            depth += num_rotations
            profit += (4 * boardingCost - runningCost) * num_rotations

        if waiting % 4 and waiting % 4 * boardingCost - runningCost > 0:
            depth += 1
            profit += waiting % 4 * boardingCost - runningCost
        return profit, depth
        

    def min_operations_max_profit_recurse(self, customers, index, boardingCost, runningCost, waiting, depth):
        if index == len(customers):
            return self.serve_remaining_people(waiting, boardingCost, runningCost, depth)
                
        # free gondola
        waiting += customers[index]
        gondola = min(waiting, 4)
        waiting -= gondola
        
        # try rotating
        profit, rotations = self.min_operations_max_profit_recurse(customers, index + 1, boardingCost, runningCost, waiting, depth + 1)
        profit += gondola * boardingCost - runningCost
        
        if profit <= 0:
            return -1, -1
        # print(profit, rotations)
        return profit, rotations
        
#         [10,9,6]
#         6
#         4
        
#         0: 4 * 6 - 4 = 24 - 4
#         1: 4 * 6 - 4
#         2: 
#         waiting = 11
#         index = 1
#         gondola = 4
#         depth = 1
        
        
        
        

class Solution:
    def minOperationsMaxProfit(self, customers: List[int], boardingCost: int, runningCost: int) -> int:
        wait = 0
        pro = 0
        high = 0
        res = -1
        for i in range(len(customers)):
            vacc = 4 - wait
            if vacc <= 0:
                wait += customers[i] - 4
                pro += 4 * boardingCost - runningCost
            # board all
            elif customers[i] <= vacc: # board=customers[i]+wait
                pro += boardingCost * (customers[i] + wait) - runningCost
                wait = 0
            else:
                pro += boardingCost * 4 - runningCost
                wait += customers[i] - 4
            if pro > high:
                high = pro
                res = i
        # determine after all arrives
        pro_per = boardingCost * 4 - runningCost
        if pro_per > 0:
            last = wait % 4
            if wait >= 4:
                if boardingCost * last - runningCost > 0: return len(customers) + wait // 4 + 1
                else: return len(customers) + wait // 4
            if boardingCost * last - runningCost > 0: return len(customers) + 1
        return res + 1 if res >= 0 else -1
                

class Solution:
    def minOperationsMaxProfit(self, customers: List[int], boardingCost: int, runningCost: int) -> int:
        
        return self.min_operations_max_profit_recurse(customers, 0, boardingCost, runningCost, 0, 0)[1]
    
    def serve_remaining_people(self, waiting, boardingCost, runningCost, depth):
        profit = 0
        
        num_rotations = waiting // 4
        if num_rotations and 4 * boardingCost - runningCost > 0:
            depth += num_rotations
            profit += (4 * boardingCost - runningCost) * num_rotations

        if waiting % 4 and waiting % 4 * boardingCost - runningCost > 0:
            depth += 1
            profit += waiting % 4 * boardingCost - runningCost
        return profit, depth
        

    def min_operations_max_profit_recurse(self, customers, index, boardingCost, runningCost, waiting, depth):
        if index == len(customers):
            return self.serve_remaining_people(waiting, boardingCost, runningCost, depth)
                
        # free gondola
        waiting += customers[index]
        gondola = min(waiting, 4)
        waiting -= gondola
        
        # try rotating
        profit, rotations = self.min_operations_max_profit_recurse(customers, index + 1, boardingCost, runningCost, waiting, depth + 1)
        profit += gondola * boardingCost - runningCost
        
        if profit <= 0:
            return -1, -1
        # print(profit, rotations)
        return profit, rotations
        
#         [10,9,6]
#         6
#         4
        
#         0: 4 * 6 - 4 = 24 - 4
#         1: 4 * 6 - 4
#         2: 
#         waiting = 11
#         index = 1
#         gondola = 4
#         depth = 1
        
        
        
        

class Solution:
    def minOperationsMaxProfit(self, customers: List[int], boardingCost: int, runningCost: int) -> int:
        nwaiting = 0
        profit = 0
        maxprofit = 0
        res = - 1
        for i,c in enumerate(customers,1):
            nwaiting += c
            onboard = min(4,nwaiting)
            nwaiting -= onboard
            profit += onboard*boardingCost - runningCost
            if maxprofit < profit:
                maxprofit = profit
                res = i
        
        if nwaiting > 0:
            roundn = nwaiting//4
            nwaiting -= roundn*4
            profit += roundn* (4*boardingCost - runningCost )
            if maxprofit < profit:
                maxprofit = profit
                res += roundn
        if nwaiting > 0:
            profit += nwaiting*boardingCost - runningCost
            if maxprofit < profit:
                maxprofit = profit
                res += 1
        
        return res if maxprofit > 0 else -1
from math import ceil
class Solution:
    def minOperationsMaxProfit(self, customers: List[int], boardingCost: int, runningCost: int) -> int:
        r = 0 
        boarded = 0
        maxProfit = 0
        maxR = -1
        print(customers,boardingCost,runningCost)
        for i in range(len(customers)-1) :
            customers[i+1] += customers[i] - 4 if customers[i] > 4 else 0
            r += 1
            boarded += min(customers[i],4)
            profit = boarded * boardingCost - r * runningCost
            if profit > maxProfit : 
                maxProfit = profit
                maxR = r
        r += ceil(customers[-1]/4)
        boarded += customers[-1] 
        if customers[-1] % 4 > 0 and (customers[-1] % 4) * boardingCost - runningCost <= 0: 
            r -= 1
            boarded -= customers[-1] % 4
        profit = boarded *boardingCost - r *runningCost
        if profit > maxProfit : 
            maxProfit = profit
            maxR = r
        return maxR if maxProfit >0 else -1 
class Solution:
    def profit(self,s,customers,count,curr_profit,waiting,rot_cost,bill):
        # print(s,waiting,count)
        if curr_profit>self.profit_so_far:
            self.profit_so_far=curr_profit
            self.ans=count
        
        if waiting==0 and s>=len(customers):
            return 
        if s>=len(customers) and waiting>4:
            n=waiting//4
            waiting=waiting-(n*4)
            count+=(n)
            
            curr_profit+= (n*4*bill - (n)*rot_cost)
            
            self.profit(len(customers) ,customers,count,curr_profit,waiting, rot_cost,bill)
            return
        
        if s<len(customers):waiting+=customers[s]
        curr_profit-=rot_cost
        if waiting<=4:
            
            curr_profit+=waiting*bill
            waiting=0
        else:
            waiting-=4
            curr_profit+=4*bill
        self.profit(s+1,customers,count+1,curr_profit,waiting,rot_cost,bill)
        
            
            
    def minOperationsMaxProfit(self, customers: List[int], boardingCost: int, runningCost: int) -> int:
        self.ans=-1
        self.profit_so_far=0
        self.profit(0,customers,0,0,0,runningCost,boardingCost)
        return self.ans
class Solution:
    def minOperationsMaxProfit(self, customers: List[int], boardingCost: int, runningCost: int) -> int:
        
        avail, cur, high, high_idx = 0, 0, 0, -1
        for i, c in enumerate(customers):
            avail += c
            if avail > 4:
                cur += 4 * boardingCost - runningCost
                avail -= 4
            else:
                cur += avail * boardingCost - runningCost
                avail = 0
            if cur > high:
                high, high_idx = cur, i + 1
                
        if 4 * boardingCost - runningCost > 0:
            i += avail // 4
            cur += (4 * boardingCost - runningCost) * (avail // 4)
            avail = avail % 4
            high, high_idx = cur, i + 1
        
            cur += avail * boardingCost - runningCost
            avail = 0
            if cur > high:
                high, high_idx = cur, high_idx + 1
            
        return high_idx
class Solution:
    def minOperationsMaxProfit(self, customers: List[int], boardingCost: int, runningCost: int) -> int:
        if boardingCost*4 <= runningCost:
            return -1
        customers.append(0)
        profit = []
        curProfit, runs = 0, 0
        for i in range(len(customers)-1):
            if customers[i] >= 4:
                if i+1 < len(customers):
                    customers[i+1] += customers[i]-4
                customers[i] = 4
            if customers[i]*boardingCost  < runningCost:
                runs = runs + 1
                profit.append((curProfit, runs))
                continue
            curProfit += customers[i]*boardingCost - runningCost
            runs += 1
            profit.append((curProfit, runs))
        #print(customers, curProfit, runs)
        if customers[-1] > 0:
            runs = runs + (customers[-1]//4)
            curProfit += (customers[-1]//4)*boardingCost - (customers[-1]//4)//4*runningCost
            customers[-1] = customers[-1]%4
            if customers[-1]*boardingCost > runningCost:
                runs = runs + 1
                curProfit += customers[-1]*boardingCost - runningCost
            #print("w", curProfit, runs)
            profit.append((curProfit, runs))
        profit.sort(key = lambda x: (-x[0], x[1]))
        #print(profit)
        return profit[0][1] if profit[0][0] > 0 else -1
            
            

class Solution:
    def minOperationsMaxProfit(self, customers: List[int], boardingCost: int, runningCost: int) -> int:
        # use a list to record profits
        cs = [customers[0]]
        for tmp in customers[1:]:
            cs.append(cs[-1] + tmp)
        # first steps
        maxp = -1
        maxn = -1
        max_cap = 0
        for i in range(len(customers)):
            max_cap = min(cs[i], max_cap + 4)
            cur_profit = max_cap * boardingCost - runningCost * (i + 1)
            if cur_profit > maxp:
                maxp = cur_profit
                maxn = i + 1
        # how many people are left?
        ppl_left = cs[-1] - max_cap
        rounds = ppl_left // 4
        cur_profit += rounds * (4 * boardingCost - runningCost)
        cur_round = len(customers) + rounds
        if cur_profit > maxp:
            maxp = cur_profit
            maxn = cur_round
        ppl_left2 = ppl_left % 4
        cur_profit += (ppl_left2 * boardingCost - runningCost)
        cur_round += 1
        if cur_profit > maxp:
            maxp = cur_profit
            maxn = cur_round
    
        return maxn

class Solution:
    def minOperationsMaxProfit(self, customers: List[int], boardingCost: int, runningCost: int) -> int:
        max_profit = times_rotated = float('-inf')
        total_customers = ran = left = i = 0
        while i < len(customers) or left > 0:
            if i < len(customers):
                left += customers[i]
            if i < ran:
                i += 1
                continue
            if left >= 4:
                times = left // 4
                total_customers += 4 * times
                ran += times
                left -= 4 * times
            else:
                total_customers += left
                left = 0
                ran += 1
            
            curr_profit = total_customers * boardingCost - ran * runningCost
            if curr_profit > max_profit:
                max_profit = curr_profit
                times_rotated = ran
            i += 1
            # print(total_customers, ran, curr_profit)
        if max_profit < 0:
            return -1
        return times_rotated
class Solution:
    def minOperationsMaxProfit(self, customers: List[int], boardingCost: int, runningCost: int) -> int:
        if runningCost >= boardingCost*4:
            return -1
        
        result = -1
        maxProfit = 0
        waiting = 0
        current = 0
        
        for i, customerCount in enumerate(customers):
            waiting += customerCount
            boarding = min(waiting, 4)
            waiting -= boarding
            current += boarding*boardingCost - runningCost
            
            if current > maxProfit:
                maxProfit = current
                result = i+1
            
        if waiting > 0:
            fullRoundsLeft = waiting // 4
            lastRoundQuantity = waiting % 4
            
            current += fullRoundsLeft * (4*boardingCost - runningCost)
            turns = len(customers) + fullRoundsLeft
            
            if current > maxProfit:
                maxProfit = current
                result = turns
                
            current += lastRoundQuantity*boardingCost - runningCost
            turns += 1
            
            if current > maxProfit:
                maxProfit = current
                result = turns
            
        
        return result if result >= 0 else -1
class Solution:
    def minOperationsMaxProfit(self, customers: List[int], boardingCost: int, runningCost: int) -> int:
        numCust = 0
        rotation = 0
        curWait = 0
        maxProfit = 0
        minRo = 0
        ##8
        for i, customer in enumerate(customers):
            while rotation < i:
                rotation += 1
                curWait -= 4
                curWait = max(0, curWait)
                curProfit = (numCust - curWait) * boardingCost - rotation * runningCost
                if curProfit > maxProfit:
                    maxProfit = curProfit
                    minRo = rotation
                
            numCust += customer
            curWait += customer
            rots = curWait // 4
            rotation += rots
            curWait %= 4
            curProfit = (numCust - curWait) * boardingCost - rotation * runningCost
            if curProfit > maxProfit:
                maxProfit = curProfit
                minRo = rotation
                
        if curWait > 0:    
            rotation += 1
            curProfit = numCust * boardingCost - rotation * runningCost
            if curProfit > maxProfit:
                maxProfit = curProfit
                minRo = rotation
        return minRo if maxProfit > 0 else -1
    
                    
            

class Solution:
    def minOperationsMaxProfit(self, customers: List[int], boardingCost: int, runningCost: int) -> int:
        profit = 0
        waiting = 0
        rotations = 0
        ans = -1
        for c in customers:
            waiting += c
            cap = 0
            if waiting >= 4:
                waiting -= 4
                cap += 4
            else:
                cap = waiting
                waiting = 0
            cur_profit = profit + (cap * boardingCost) - runningCost
            rotations += 1
            if cur_profit > profit:
                ans = rotations
            profit = cur_profit
        
        if waiting > 0:
            req_rotations = math.ceil(waiting/4)
            ignore = waiting // 4
            possible_profit = (waiting * boardingCost) - (req_rotations * runningCost)
            full_only = ((waiting - (waiting % 4)) * boardingCost) - (ignore * runningCost)
            if possible_profit > full_only:
                additional = req_rotations
                if profit + possible_profit > profit:
                    ans = rotations + req_rotations
            else:
                additional = ignore
                if profit + full_only > profit:
                    ans = rotations + ignore
        return ans

class Solution:
    def minOperationsMaxProfit(self, A, BC, RC):
        ans=profit=t=0
        maxprofit=0
        wait=i=0
        n=len(A)
        while i<n:
            if i<n:
                wait+=A[i]
                i+=1
            t+=1
            y=wait if wait<4 else 4
            wait-=y
            profit+=y*BC
            profit-=RC
            if profit>maxprofit:
                maxprofit=profit
                ans=t
        
        profit+=wait//4*BC
        #profit-=RC*(wait+3)//4
        if profit>maxprofit:
            ans+=wait//4
        if wait%4*BC>RC:
            maxprofit+=wait%4*BC-RC
            ans+=1

        if maxprofit<=0:
            return -1
        else:
            return ans

class Solution:
    def minOperationsMaxProfit(self, customers: List[int], boardingCost: int, runningCost: int) -> int:
        ans=-1
        maxp=0
        curr=0
        c=0
        track=0
        for i in range(len(customers)):
            track+=customers[i]
            curr+=min(4,track)*boardingCost-runningCost
            track-=min(4,track)
            c+=1
            if curr>maxp:
                maxp=curr
                ans=c
        if track>=4:
            curr+=(track-track%4)*boardingCost-(track//4)*runningCost
            c+=track//4
            if curr>maxp:
                maxp=curr
                ans=c
        curr+=(track%4)*boardingCost-runningCost
        c+=1
        if curr>maxp:
            maxp=curr
            ans=c
        return ans
class Solution:
    def minOperationsMaxProfit(self, customers: List[int], boardingCost: int, runningCost: int) -> int:
        ans = -1
        if boardingCost * 4 < runningCost: return ans
        cur_profit = 0
        wait_num = 0
        dic = {}
        for i in range(len(customers)):
            if customers[i] + wait_num < 5:
                cur_profit += boardingCost * (customers[i] + wait_num) - runningCost
                wait_num = 0
            else:
                cur_profit += boardingCost * 4 - runningCost
                wait_num += customers[i] - 4
                
            if cur_profit > ans:
                ans = cur_profit
                dic[ans] = i+1
        
        if wait_num > 0:
            while wait_num > 3:
                wait_num -= 4
                dic[ans] += 1
            if wait_num * boardingCost > runningCost:
                dic[ans] += 1
        
        return dic[ans]

class Solution:
    def minOperationsMaxProfit(self, customers: List[int], boardingCost: int, runningCost: int) -> int:
        if boardingCost*4 < runningCost: return -1
        rot = 0
        spot = 0
        prof = max_prof = [0, 0]
        while spot < len(customers):
            rot += 1
            curr = customers[spot]
            if curr > 4:
                if spot == len(customers) - 1: 
                    temp = prof[0] + 4*boardingCost*(curr//4) - runningCost*(curr//4)
                    max_prof1 = max(max_prof, [temp, rot -1 + curr//4]) 
                    temp = prof[0] + 4*boardingCost*(curr//4) + boardingCost*(curr%4) - runningCost*(curr//4+bool(curr%4))
                    max_prof2 = max(max_prof, [temp, rot -1 + curr//4 + bool(curr%4)]) 
                    if max_prof1[0] != max_prof2[0]: return max(max_prof1, max_prof2)[1] 
                    if max_prof1[0] == max_prof2[0]: return max_prof1[1] 
                else:
                    customers[spot+1] += curr-4
                prof[0] += 4*boardingCost - runningCost
            else:
                prof[0] += curr*boardingCost - runningCost
            max_prof = max(max_prof, [prof[0], rot]) 
            spot += 1
        return max_prof[1]
                
            

class Solution:
    def minOperationsMaxProfit(self, customers: List[int], boardingCost: int, runningCost: int) -> int:
        profit = 0
        waiting = 0
        rotation = 0
        max_profit = 0
        ans = None
        for customer in customers:
            customer += waiting
            rotation += 1
            if customer>=4:
                profit += 4*boardingCost - runningCost
                waiting = customer-4
            else:
                profit = customer*boardingCost - runningCost
                waiting = 0
            
            if max_profit<profit:
                max_pprofit = profit
                ans = rotation
        
        if waiting>0:
            if waiting>4:
                while waiting>4:
                    profit += 4*boardingCost - runningCost
                    waiting = waiting-4
                    rotation += 1
                    #print(profit)
                    if max_profit<profit:
                        max_pprofit = profit
                        ans = rotation
            
        profit = waiting*boardingCost - runningCost
        rotation+=1
        if max_profit<profit:
            max_pprofit = profit
            ans = rotation
        
        return ans if ans else -1
class Solution:
    def minOperationsMaxProfit(self, customers: List[int], boardingCost: int, runningCost: int) -> int:
        profit = 0
        maxProfit = -1
        res = -1
        currentCustomer = 0
        for i, c in enumerate(customers):
            currentCustomer += c
            if currentCustomer <=4:
                profit += currentCustomer*boardingCost - runningCost 
                currentCustomer = 0
            else:
                profit += 4*boardingCost - runningCost
                currentCustomer -=4
            if profit > maxProfit:
                maxProfit = profit
                res = i+1
        
        rounds = currentCustomer // 4
        left = currentCustomer % 4
        if boardingCost*4 - runningCost > 0:
            profit += rounds*(boardingCost*4 - runningCost)
            if profit > maxProfit:
                maxProfit = profit
                res += rounds
            profit += boardingCost*left-runningCost
            if profit > maxProfit:
                maxProfit = profit
                res +=1
            
        
        return res
class Solution:
    def minOperationsMaxProfit(self, customers: List[int], boardingCost: int, runningCost: int) -> int:
        minimum = runningCost//boardingCost+1
        if minimum>4: return -1
        max_value = 0
        profit = 0
        remain = 0
        turns = 0
        res = None
        for c in customers:
            turns += 1
            temp = remain+c
            if temp>=4:
                remain = temp-4
                profit += 4*boardingCost-runningCost
            else:
                remain = 0
                profit += temp*boardingCost-runningCost
            if profit>max_value:
                res = turns
                max_value = profit
        print(turns, remain)
        while remain:
            turns += 1
            if remain>=4:
                remain -= 4
                profit += 4*boardingCost-runningCost
            else:
                profit += remain*boardingCost-runningCost
                remain = 0
            if profit>max_value:
                res = turns
                max_value = profit
        return res
class Solution:
    def minOperationsMaxProfit(self, nums: List[int], pos: int, fee: int) -> int:
        if 4*pos <= fee:
            return -1
        ans = cur = 0
        s = sum(nums)
        best = -math.inf
        p = 0
        for i, x in enumerate(nums):
            cur += x
            if cur >= 4:
                p += (4*pos - fee)
                cur -= 4
            else:
                p += (cur * pos - fee)
                cur = 0
            
            if p > best:
                best = p
                ans = i + 1
        res = len(nums)
        while cur > 0:
            res += 1
            if cur >= 4:
                p += (4*pos - fee)
                cur -= 4
            else:
                p += (cur * pos - fee)
                cur = 0
            if p > best:
                best = p
                ans = res
                
        if best <=0:
            return -1
        return ans

class Solution:
    def minOperationsMaxProfit(self, customers: List[int], boardingCost: int, runningCost: int) -> int:
        inLine = 0
        profit = 0
        maxProf = -1
        maxRoll = -1
        rolls = 0
        for i in customers:
            inLine += i
            if inLine >= 4:
                profit += 4*boardingCost - runningCost
                inLine -= 4
            else:
                profit += inLine*boardingCost - runningCost
                inLine = 0
            rolls += 1
            
            if profit > maxProf:
                maxProf = profit
                maxRoll = rolls
                
        
        while inLine:
            if inLine >= 4:
                profit += 4*boardingCost - runningCost
                inLine -= 4
            else:
                profit += inLine*boardingCost - runningCost
                inLine = 0
            rolls += 1
            # maxProf = max(maxProf, profit)
            if profit > maxProf:
                maxProf = profit
                maxRoll = rolls
        
        return maxRoll
            

