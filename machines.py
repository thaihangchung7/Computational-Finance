import numpy as np


n_mach = []
machine_true = [1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
start_machine = 0
for i in machine_true:
    if i == 1:
        start_machine += 1
        n_mach.append(start_machine)
    elif i == 0:
        n_mach.append(5)

#print(n_mach)

index = 0
for i in n_mach:
   index += 1
   print("index:", index)
   if index > 10 and machine_true[index] == 0:
      start_machine -= 1
      print("start_machine: ", start_machine)
      
      #print(i)
      n_mach.pop(n_mach[index])
      n_mach.append(start_machine)
      if start_machine == 0:
         break

print(n_mach)
         
      
