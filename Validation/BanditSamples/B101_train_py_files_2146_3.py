
n, d, m = [int(x) for x in input().split()]
a = [int(x) for x in input().split()]
a = list(reversed(sorted(a)))
nl = [x for x in a if x <= m]
ml = [x for x in a if x > m]
aml = [0]
for x in ml:
    aml.append(aml[-1] + x)
anl = [0]
for x in nl:
    anl.append(anl[-1] + x)

if len(ml) == 0:
    print(sum(nl))
    return

result = []

best = 0

for i in range(1, len(ml) + 1):
    # Is it possible to have i muzzles?
    if (i-1)*(d+1) + 1 > n:
        continue
    if i*d < len(ml) - i:
        continue

    # What is my score if I cause i muzzles?
    # Then it is: the top i muzzling elements.
    # Plus the top how many nmes I have left after filling

    cur = aml[i]
    need_nmes = max(0, (i-1)*(d+1) + 1 - len(ml))
    rem_nmes = len(nl) - need_nmes
    assert rem_nmes >= 0
    cur += anl[rem_nmes]
    
    if cur > best:
        #print("Doing better with", i, "muzzles:", cur)
        best = cur

print(best)

