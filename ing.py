# st =3.9
# # print(st)
# # print(type(st))
# num = round(st)
# print(type(num))
# print (num)
def is_prime(n):
    if n<2:
        return False
    
    for i in range(2,n):
        if n%i ==0:
            return False
    return True
   # 2 
i =[]
print("consecutive numbers are:")  
for j in range(2,1000):
    if is_prime(j) and is_prime(j+2):
        print(j , "and", j+2)   
        i.append(j) 
print("total number of pairs are",len(i))        
        
# 4            
for j in range(2,1000):
    if is_prime(j) and is_prime(j+2):
        
        if((j+j+2)%5 ==0):
            print("pairs Whose Sum is divisible by 5", j, "and",j+2)
