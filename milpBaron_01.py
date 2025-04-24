from pyomo.environ import *
import pyomo.environ as pyo
import numpy as np
from numpy.linalg import solve
import matplotlib.pyplot as plt
import sympy as sp
from sympy import sqrt, atan, Function, lambdify, symbols, Matrix
import math
import re
# Create a Pyomo model
model = ConcreteModel()

PL = 1800000
QL = (PL/0.9)*math.sin(math.acos(0.9))
St = 6000000
VoltageH = 12470/np.sqrt(3)
VoltageL = 4160
kVAt = 6000000
nt = 12470/4160

v1ra = VoltageH * math.cos(math.radians(0))
v1rb = VoltageH * math.cos(math.radians(-120))
v1rc = VoltageH * math.cos(math.radians(120))

v1ia = VoltageH * math.sin(math.radians(0))
v1ib = VoltageH * math.sin(math.radians(-120))
v1ic = VoltageH * math.sin(math.radians(120))
Vsr = np.array([
    [v1ra],
    [v1rb],
    [v1rc] 
])
Vsi = np.array([
    [v1ia],
    [v1ib],
    [v1ic]
])
InitI = np.ones((3,1))
Xinit = np.vstack((Vsr, Vsi, Vsr, Vsi, (1/nt)*Vsr, (1/nt)*Vsi, (1/nt)*Vsr, (1/nt)*Vsi, InitI, InitI, InitI, InitI, InitI, InitI))
#Setting up the equations for transformer admittance matrix
ztlow = (VoltageL**2)/kVAt
ztpu = 0.01+0.06j
zt = ztpu*ztlow  
zphase = np.array([ 
    [zt, 0, 0],
    [0, zt, 0],
    [0, 0, zt]
])
Yt = np.linalg.inv(zphase)

Gtr = Yt.real
Bti = Yt.imag 

Zline = np.array([
 [0.4576+1.078j, 0.1559 +0.5017j, 0.1535+0.3849j],
 [0.1559+0.5017j, 0.4666+1.0482j, 0.158+0.4236j],
 [0.1535+0.3849j, 0.158+0.4236j, 0.4615+1.0651j]
])

lineOne = 2000/5280
lineTwo = 2500/5280

Zline12 = Zline*lineOne
Zline34 = Zline*lineTwo

Yline12 = np.linalg.inv(Zline12)
Yline34 = np.linalg.inv(Zline34)

Gl12 = Yline12.real
Bl12 = Yline12.imag

Gl34 = Yline34.real
Bl34 = Yline34.imag

# 3-phase vector
n=3

model.n = pyo.RangeSet(0, n-1)
model.V1r = Var(range(n), bounds=(-20000, 20000), initialize=12470)
model.V1i = Var(range(n), bounds=(-20000, 20000), initialize=0)
model.V2r = Var(range(n), bounds=(-20000, 20000), initialize=12000)
model.V2i = Var(range(n), bounds=(-20000, 20000), initialize = 0)
model.V3r = Var(range(n), bounds=(-20000, 20000), initialize=12000)
model.V3i = Var(range(n), bounds=(-20000, 20000), initialize = 0)
model.V4r = Var(range(n), bounds=(-20000, 20000), initialize=12000)
model.V4i = Var(range(n), bounds=(-20000, 20000), initialize = 0)
model.Islackr = Var(range(n), bounds=(-20000, 20000), initialize=0)
model.Islacki = Var(range(n), bounds=(-20000, 20000), initialize=0)
model.Ixr = Var(range(n), bounds=(-20000, 20000), initialize=0)
model.Ixi = Var(range(n), bounds=(-20000, 20000), initialize=0)
model.I2xr = Var(range(n), bounds=(-20000, 20000), initialize=0)
model.I2xi = Var(range(n), bounds=(-20000, 20000), initialize=0)
# model.St = Var(range(n), bounds=(-20000000, 20000000), initialize = 0)

aj = [5000000, 6000000, 7000000, 8000000, 9000000, 10000000]
bj = [1, 2, 3, 4, 5, 6]
# bj = [-6,-5,-4,-3,-2,-1]
sizeSj = len(aj)
model.sj = Var(range(sizeSj), within = pyo.Binary)



# Define objective function
# model.obj = Objective(expr = 1) #when i chose this, it always chose the largest value
# model.obj = Objective(expr = sum((bj[j])*model.sj[j] for j in range(sizeSj))) #when this was constraint, it wouldn't stop iterating

model.obj = Objective(expr = bj[0]*model.sj[0] + bj[1]*model.sj[1] + bj[2]*model.sj[2] + bj[3]*model.sj[3] + bj[4]*model.sj[4] + bj[5]*model.sj[5])

def equality_constraint1(model, i):
    return -model.Islackr[i] + sum(Gl12[i,j]*(model.V1r[j]-model.V2r[j]) for j in range(n)) - sum(Bl12[i,j]*(model.V1i[j]-model.V2i[j]) for j in range(n))==0
def equality_constraint2(model, i):
    return -model.Islacki[i] + sum(Gl12[i,j]*(model.V1i[j]-model.V2i[j]) for j in range(n)) + sum(Bl12[i,j]*(model.V1r[j]-model.V2r[j]) for j in range(n))==0

def equality_constraint3(model, i):
    return Vsr[i] - model.V1r[i] == 0
def equality_constraint4(model, i):
    return Vsi[i] - model.V1i[i] == 0

def equality_constraint5(model, i):
    return model.Ixr[i]+sum(Gl12[i,j]*(model.V2r[j]-model.V1r[j]) for j in range(n)) - sum(Bl12[i,j]*(model.V2i[j]-model.V1i[j]) for j in range(n))==0
def equality_constraint6(model, i):
    return model.Ixi[i]+sum(Gl12[i,j]*(model.V2i[j]-model.V1i[j]) for j in range(n)) + sum(Bl12[i,j]*(model.V2r[j]-model.V1r[j]) for j in range(n))==0

def equality_constraint7(model, i):
    return nt*model.Ixr[i] - model.I2xr[i] == 0
def equality_constraint8(model, i):
    return nt*model.Ixi[i] - model.I2xi[i] == 0

def equality_constraint9(model, i):
    return -model.I2xr[i] + sum(Gtr[i,j]*((1/nt)*model.V2r[j]-model.V3r[j]) for j in range(n)) - sum(Bti[i,j]*((1/nt)*model.V2i[j]-model.V3i[j]) for j in range(n))==0
def equality_constraint10(model, i):
    return -model.I2xi[i] + sum(Gtr[i,j]*((1/nt)*model.V2i[j]-model.V3i[j]) for j in range(n)) + sum(Bti[i,j]*((1/nt)*model.V2r[j]-model.V3r[j]) for j in range(n))==0

def equality_constraint11(model, i):
    return sum(Gtr[i,j]*(model.V3r[j] - (1/nt)*model.V2r[j]) for j in range(n)) - sum(Bti[i,j]*(model.V3i[j] - (1/nt)*model.V2i[j]) for j in range(n)) + \
        sum(Gl34[i,j]*(model.V3r[j]-model.V4r[j]) for j in range(n)) - sum(Bl34[i,j]*(model.V3i[j]-model.V4i[j]) for j in range(n))==0
        
def equality_constraint12(model, i):
    return sum(Gtr[i,j]*(model.V3i[j] - (1/nt)*model.V2i[j]) for j in range(n)) + sum(Bti[i,j]*(model.V3r[j] - (1/nt)*model.V2r[j]) for j in range(n)) + \
        sum(Gl34[i,j]*(model.V3i[j]-model.V4i[j]) for j in range(n)) + sum(Bl34[i,j]*(model.V3r[j]-model.V4r[j]) for j in range(n))==0

def equality_constraint13(model, i):
    return sum(Gl34[i,j]*(model.V4r[j]-model.V3r[j]) for j in range(n)) -sum(Bl34[i,j]*(model.V4i[j] - model.V3i[j]) for j in range(n)) + \
        (PL*model.V4r[i] + QL*model.V4i[i])/(model.V4r[i]**2 + model.V4i[i]**2)==0
        
def equality_constraint14(model, i):
    return sum(Gl34[i,j]*(model.V4i[j]-model.V3i[j]) for j in range(n)) + sum(Bl34[i,j]*(model.V4r[j] - model.V3r[j]) for j in range(n)) + \
        (PL*model.V4i[i] - QL*model.V4r[i])/(model.V4r[i]**2 + model.V4i[i]**2)==0
        
        
#make sure we only select one transformer
def equality_constraint15(model):
    return sum(model.sj[j] for j in range(sizeSj)) == 1

#select the right transformer
# def ineq_constr1(model):
#     return 7100000 <= (sum((aj[j])*model.sj[j] for j in range(sizeSj))) #change this so it does it iteratively

# def ineq_constr2(model,i):
#     return model.St[i] >=0
# #Power flow constraint Sabc = Sa + Sb + Sc

# def ineq_constr3(model, i):
#     realP = (sum(model.V2r[j]*model.Ixr[j]+model.V2i[j]*model.Ixi[j]) for j in range(n))
#     imagQ = (sum(model.V2i[j]*model.Ixr[j]-model.V2r[j]*model.Ixi[j]) for j in range(n))
#     rhs = (sum(((aj[j])**2)*model.sj[j] for j in range(sizeSj)))
#     return realP**2 + imagQ**2 <= rhs
# did not like this, threw error for unsupported operand type
def ineq_constr3(model, i): #tried dividing by 1e3 to get less iterations in making these numbers smaller
    realP = (model.V2r[0]*model.Ixr[0] + model.V2i[0]*model.Ixi[0] + model.V2r[1]*model.Ixr[1] + model.V2i[1]*model.Ixi[1] + \
        model.V2r[2]*model.Ixr[2] + model.V2i[2]*model.Ixi[2])/1e3
    
    imagQ = ((model.V2i[0]*model.Ixr[0] - model.V2r[0]*model.Ixi[0]) + (model.V2i[1]*model.Ixr[1] - model.V2r[1]*model.Ixi[1]) + \
        (model.V2i[2]*model.Ixr[2] - model.V2r[2]*model.Ixi[2]))/1e3
        
    rhs = (sum(((aj[j]/1e3)**2)*model.sj[j] for j in range(sizeSj)))
    
    return realP**2 + imagQ**2 <= rhs

#Sabc = Sa + Sb + Sc find magnitude 

model.constraint1 = Constraint(model.n, rule=equality_constraint1)
model.constraint2 = Constraint(model.n, rule=equality_constraint2)
model.constraint3 = Constraint(model.n, rule=equality_constraint3)
model.constraint4 = Constraint(model.n, rule=equality_constraint4)
model.constraint5 = Constraint(model.n, rule=equality_constraint5)
model.constraint6 = Constraint(model.n, rule=equality_constraint6)
model.constraint7 = Constraint(model.n, rule=equality_constraint7)
model.constraint8 = Constraint(model.n, rule=equality_constraint8)
model.constraint9 = Constraint(model.n, rule=equality_constraint9)
model.constraint10 = Constraint(model.n, rule=equality_constraint10)
model.constraint11 = Constraint(model.n, rule=equality_constraint11)
model.constraint12 = Constraint(model.n, rule=equality_constraint12)
model.constraint13 = Constraint(model.n, rule=equality_constraint13)
model.constraint14 = Constraint(model.n, rule=equality_constraint14)
model.constraint15 = Constraint(rule=equality_constraint15)

# model.ineq_constr1 = Constraint( rule=ineq_constr1)
# model.ineq_constr2 = Constraint(model.n, rule=ineq_constr2)
model.ineq_constr3 = Constraint(model.n, rule=ineq_constr3)
# model.ineq_constr2 = Constraint(rule=ineq_constr2)
solver = SolverFactory('baron')
solver.options['MaxIter'] = 1000
solver.options['PrLevel'] = 5 
solver.options['TolRel'] = 1e-6  

result = solver.solve(model, tee=True, logfile="baron_prac_data.txt")  # 'tee=True' will display solver output in the terminal

# Display results
model.display()

V1r_vals = np.array([pyo.value(model.V1r[i]) for i in range(n)]).reshape(-1,1)
V1i_vals = np.array([pyo.value(model.V1i[i]) for i in range(n)]).reshape(-1,1)
V2r_vals = np.array([pyo.value(model.V2r[i]) for i in range(n)]).reshape(-1,1)
V2i_vals = np.array([pyo.value(model.V2i[i]) for i in range(n)]).reshape(-1,1)
V3r_vals = np.array([pyo.value(model.V3r[i]) for i in range(n)]).reshape(-1,1)
V3i_vals = np.array([pyo.value(model.V3i[i]) for i in range(n)]).reshape(-1,1)
V4r_vals = np.array([pyo.value(model.V4r[i]) for i in range(n)]).reshape(-1,1)
V4i_vals = np.array([pyo.value(model.V4i[i]) for i in range(n)]).reshape(-1,1)
Islackr_vals = np.array([pyo.value(model.Islackr[i]) for i in range(n)]).reshape(-1,1)
Islacki_vals = np.array([pyo.value(model.Islacki[i]) for i in range(n)]).reshape(-1,1)
Ixr_vals = np.array([pyo.value(model.Ixr[i]) for i in range(n)]).reshape(-1,1)
Ixi_vals = np.array([pyo.value(model.Ixi[i]) for i in range(n)]).reshape(-1,1)
I2xr_vals = np.array([pyo.value(model.I2xr[i]) for i in range(n)]).reshape(-1,1)
I2xi_vals = np.array([pyo.value(model.I2xi[i]) for i in range(n)]).reshape(-1,1)
# St_vals = np.array([pyo.value(model.St[i]) for i in range(n)]).reshape(-1,1)
sj_var = np.array([pyo.value(model.sj[i]) for i in range(sizeSj)]).reshape(-1,1)

# Stot_val = np.sum(St_vals)
# print(Stot_val)






# BinaryS = np.array([pyo.value(model.sj[i]) for i in range(sizeSj)]).reshape(-1,1)


Xn = np.vstack([V1r_vals, V1i_vals, V2r_vals, V2i_vals, V3r_vals, V3i_vals, 
                             V4r_vals, V4i_vals, Islackr_vals, Islacki_vals, 
                             Ixr_vals, Ixi_vals, I2xr_vals, I2xi_vals])

print("Solution Vector:")
names = ["V1", "V2", "V3", "V4", "Islack", "Ix", "I2r" ]
iterate = 0
while iterate <=6:
    iter_while1 = iterate*6
    a =np.sqrt(Xn[iter_while1+0,0]**2 + Xn[iter_while1+3,0]**2)
    b =np.sqrt(Xn[iter_while1+1,0]**2 + Xn[iter_while1+4,0]**2)
    c =np.sqrt(Xn[iter_while1+2,0]**2 + Xn[iter_while1+5,0]**2)
    d = np.degrees(np.arctan2(Xn[iter_while1+3,0], Xn[iter_while1+0,0]))
    e = np.degrees(np.arctan2(Xn[iter_while1+4,0], Xn[iter_while1+1,0]))
    f = np.degrees(np.arctan2(Xn[iter_while1+5,0], Xn[iter_while1+2,0]))
    print(names[iterate])
    print("a-phase magnitude: ", a, "angle: ", d)
    print("b-phase magnitude: ", b, "angle: ", e)
    print("c-phase magnitude: ", c, "angle: ", f)
    iterate +=1
    
Sa1 = pyomo.environ.value((model.V2r[0]*model.Ixr[0]+model.V2i[0]*model.Ixi[0])**2 + (model.V2i[0]*model.Ixr[0]-model.V2r[0]*model.Ixi[0])**2)
Sb1 = pyomo.environ.value((model.V2r[1]*model.Ixr[1]+model.V2i[1]*model.Ixi[1])**2 + (model.V2i[1]*model.Ixr[1]-model.V2r[1]*model.Ixi[1])**2)
Sc1= pyomo.environ.value((model.V2r[2]*model.Ixr[2]+model.V2i[2]*model.Ixi[2])**2 + (model.V2i[2]*model.Ixr[2]-model.V2r[2]*model.Ixi[2])**2)
Sa = pyomo.environ.sqrt(Sa1)
Sb = pyomo.environ.sqrt(Sb1)
Sc = pyomo.environ.sqrt(Sc1)
Stot = Sa + Sb + Sc
print("Apparent per phase power at Transformer: ", Sa, "and", Sb, "and", Sc)
print("Apparent power at Transformer: ", Stot)

# with open("ipopt_prac_data.txt", "r") as f:
#     ipoptData = f.read()
    
# pattern = re.compile(r"^\s*(\d+)\s+([-+]?\d*\.\d+e[+-]?\d+)\s+([-+]?\d*\.\d+e[+-]?\d+)\s+([-+]?\d*\.\d+e[+-]?\d+)", re.MULTILINE)


# iterations, objectives, inf_prs, inf_dus = [], [], [], []

# # Search for all matches in the log
# for match in pattern.finditer(ipoptData):
#     iterations.append(int(match.group(1)))        # Iteration number
#     objectives.append(float(match.group(2)))      # Objective value
#     inf_prs.append(float(match.group(3)))         # Primal infeasibility
#     inf_dus.append(float(match.group(4))) 
            
# plt.plot(iterations, inf_prs, marker='o', linestyle='-', label="Primal Infeasibility")
# plt.plot(iterations, inf_dus, marker='s', linestyle='--', label="Dual Infeasibility")
# plt.yscale("log")  # Log scale for better visualization
# plt.xlabel("Iterations")
# plt.ylabel("Error (log scale)")
# plt.title("IPOPT Convergence of IEEE Four")
# plt.legend()
# plt.grid()
# plt.show()