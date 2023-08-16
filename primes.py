import numpy as np
import time 
def prime(num):
    if num<=1:
        return False
    if num <=3:
        return True
    if num % 2==0 or num%3==0:
        return False
    i=5
    while i*i<=num:
        if num%i==0 or num%(i+2)==0:
            return False
        i=i+6
    return True
start=time.time()
for i in range(1000000000):
    booll=prime(i)
    if booll==True:
      print(f"{i} is prime")
print(time.time()-start)
