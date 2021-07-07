# import solver
import gurobipy as gp #import gurobipy library in Python as gp
from gurobipy import GRB
import pandas as pd #import pandas library as pd
import numpy as np #import numpy library
import math
import xlrd

data_header=np.loadtxt('Instances/Data6_a.txt',max_rows=1,dtype=int)
data_header_decimal=np.loadtxt('Instances/Data6_a.txt',max_rows=1,dtype=float)
print(data_header)
n=data_header[0]
no_vehicles=data_header[7]; K=[] #vehicle number
for i in range(1,no_vehicles+1): K.append(i)
Qk = data_header[1] #vehicle capacity
LL = data_header[9] #maximum ride time of every user
max_route_duration = data_header[8]
TT={i:max_route_duration for i in K} #maximum route duration
a_par=data_header[5] #fixed time for unloading and reloading at the depot
b_par=data_header[6] #fixed time required for uploading and loading a customer
rho = data_header_decimal[10]
gamma = data_header_decimal[11]
print('rho',rho,'gamma',gamma)
Mbig=10000 #a large positive number
print(n,no_vehicles,Qk,TT,a_par,b_par)
e_bound={}  # lower bound of the time window
l_bound={} # upper bound of the time window
latitude = {}; longitude ={}
q={}; d={}
data_main_body=np.loadtxt('Instances/Data6_a.txt',skiprows=1,dtype=int)
data_tw=np.loadtxt('Instances/Data6_a_tw.txt',dtype=int)
print(data_main_body,data_tw,'length',len(data_tw))
for i in range(1,n+1):
    latitude[i]=data_main_body[i,3]
    longitude[i]=data_main_body[i,4]
    q[i]=data_main_body[i,5]
    e_bound[i]=data_tw[i-1,2]
    l_bound[i]=data_tw[i-1,3]
for i in range(n+1,2*n+1):
    latitude[i]=data_main_body[i-n,1]
    longitude[i]=data_main_body[i-n,2]
    q[i]=data_main_body[i-n,5]
    e_bound[i]=data_tw[i-n-1,0]
    l_bound[i]=data_tw[i-n-1,1]
for i in [0,2*n+1,2*n+2,2*n+3]:
    latitude[i]=data_main_body[0,1]
    longitude[i] = data_main_body[0,2]
    q[i]=data_main_body[0,5]
    e_bound[i] = data_header[2]*60
    l_bound[i] = data_header[3]*60

print(latitude)
print(longitude)
print(q)
print('e_bound',e_bound)
print(l_bound)

c={};t={}
from scipy.spatial import distance
for i in range(2*n+4):
   for j in range(2*n+4):
      c[(i,j)] = (1/1000)*distance.euclidean([latitude[i],longitude[i]],[latitude[j],longitude[j]]) * (60/data_header[4])
      t[(i,j)] = (1/1000)*distance.euclidean([latitude[i],longitude[i]],[latitude[j],longitude[j]]) * (60/data_header[4])

print(c,t)



P=[];D=[];V=[]
for i in range(1,n+1): P.append(i)
print('P',P)
for i in range(n+1,2*n+1): D.append(i)
print('D',D)
for i in range(0,2*n+4): V.append(i)
print('V',V)
PcupD=[]
for i in range(1,2*n+1): PcupD.append(i)
print('PcupD',PcupD)
PplusDepot=P+[0,2*n+1]
DplusDepot=D+[2*n+2,2*n+3]

A0=[(0,j) for j in P]
A00=[(j,2*n+1) for j in P]
A000=[(i,j) for i in P for j in P if j!=i]# if e_bound[i]+t[i,j]<=l_bound[j]]
A1=[(2*n+2,j) for j in D]
A11=[(j,2*n+3) for j in D]
A111=[(i,j) for i in D for j in D if j!=i]# if e_bound[i]+t[i,j]<=l_bound[j]]
A=A0+A00+A000+A1+A11+A111
print('A',A)

active_t=np.zeros((2*n+4,2*n+4))
for i in range(2*n+4):
   for j in range(2*n+4):
      if (i,j) in A:
          active_t[i,j]=t[(i,j)]
print('active_t',active_t)
np.savetxt('active_t.txt',active_t,fmt='%.3f')

#Initialize the Gurobi model
model = gp.Model()
#VARIABLES
x = model.addVars(A,K,vtype=gp.GRB.BINARY, name='x')
w = model.addVars(K,lb=0.0,vtype=gp.GRB.CONTINUOUS, name='w')
r = model.addVars(V,lb=0.0,vtype=gp.GRB.CONTINUOUS, name='r')
u = model.addVars(V,K,lb=0.0,vtype=gp.GRB.CONTINUOUS, name='u')
eta = model.addVars(P,K,vtype=gp.GRB.BINARY, name='eta')
theta = model.addVars(P,K,vtype=gp.GRB.BINARY, name='theta')
eta_k = model.addVars(K,vtype=gp.GRB.BINARY, name='eta_k')
theta_k = model.addVars(K,vtype=gp.GRB.BINARY, name='theta_k')
tau = model.addVars(K,lb=0.0,vtype=gp.GRB.CONTINUOUS, name='tau')
z = model.addVars(P,lb=0.0,vtype=gp.GRB.CONTINUOUS, name='z')
v = model.addVars(V,K,vtype=gp.GRB.CONTINUOUS, name='v')
g = model.addVars(V,K,vtype=gp.GRB.CONTINUOUS, name='g')
d = model.addVars(V,K,vtype=gp.GRB.CONTINUOUS, name='d')
delta = model.addVars(V,K,vtype=gp.GRB.BINARY, name='delta')
p = model.addVars(V,K,vtype=gp.GRB.CONTINUOUS, name='p')
s = model.addVars(A,K,lb=-math.inf,vtype=gp.GRB.CONTINUOUS, name='s')
s_tilde = model.addVars(A,K,lb=-math.inf,vtype=gp.GRB.CONTINUOUS, name='s_tilde')
sigma = model.addVars(A,K,lb=-math.inf,vtype=gp.GRB.CONTINUOUS, name='sigma')
p_active = model.addVars(V,lb=-math.inf,vtype=gp.GRB.CONTINUOUS, name='p_active')

#KPI Variables
vehiclecosts = model.addVar(vtype=gp.GRB.CONTINUOUS, name='vehiclecosts')
route_duration_pickup = model.addVars(K,vtype=gp.GRB.CONTINUOUS, name='route_duration_pickup')
route_duration_delivery = model.addVars(K,vtype=gp.GRB.CONTINUOUS, name='route_duration_delivery')

#CONSTRAINTS
print('q',q)

#Constraint 21
model.addConstrs(p_active[i]+sigma[j2,i,k2]==p[i,k2] for (j2,i) in A if i in P+D for k2 in K)
model.addConstrs(sigma[j,i,k]<=Mbig*(1-x[j,i,k]) for (j,i) in A if i in P+D for k in K)
model.addConstrs(sigma[j,i,k]>=-Mbig*(1-x[j,i,k]) for (j,i) in A if i in P+D for k in K)
#Constraint 22
model.addConstrs(r[i]==p_active[n+i]-p_active[i] for i in P for kei in K)

#Constraint 2
model.addConstrs( sum(sum(x[i,j,kei] for (i,j) in A if i==ii) for kei in K) == 1 for ii in PcupD)

#Constraints 27-30
model.addConstrs(v[0,kei]==0 for kei in K)
model.addConstrs(v[2*n+2,kei]==sum(sum(q[i]*x[i,j,kei] for (i,j) in A if i==ii) for ii in D) for kei in K)
#model.addConstrs((x[i,j,kei] == 1) >> (v[j,kei]==v[i,kei]+q[j]) for i in P+[0] for j in P if j!=i for kei in K)

model.addConstrs(v[j,kei]+s[i,j,kei]==v[i,kei]+q[j] for (i,j) in A0+A00+A000 if j!=i for kei in K)
model.addConstrs(s[i,j,kei]<=Mbig*(1-x[i,j,kei]) for (i,j) in A0+A00+A000 if j!=i for kei in K)
model.addConstrs(s[i,j,kei]>=-Mbig*(1-x[i,j,kei]) for (i,j) in A0+A00+A000 if j!=i for kei in K)

#model.addConstrs((x[i,j,kei] == 1) >> (v[j,kei]==v[i,kei]-q[j]) for i in D+[2*n+2] for j in D if j!=i for kei in K)
model.addConstrs(v[j,kei]+s[i,j,kei]==v[i,kei]-q[j] for i in D+[2*n+2] for j in D if j!=i for kei in K)
model.addConstrs(s[i,j,kei]<=Mbig*(1-x[i,j,kei]) for i in D+[2*n+2] for j in D if j!=i for kei in K)
model.addConstrs(s[i,j,kei]>=-Mbig*(1-x[i,j,kei]) for i in D+[2*n+2] for j in D if j!=i for kei in K)

model.addConstrs(v[i,kei]<=Qk for i in V for kei in K)
model.addConstrs(0<=v[i,kei] for i in V for kei in K)

#Constraints 33-38
model.addConstrs(d[i,kei]==1+gamma*g[i,kei] for i in V for kei in K)
model.addConstrs(g[i,kei]>=v[i,kei]-rho*Qk for i in V for kei in K)
model.addConstrs(g[i,kei]>=0 for i in V for kei in K)
model.addConstrs(g[i,kei]<=v[i,kei]-rho*Qk+Mbig*delta[i,kei] for i in V for kei in K)
model.addConstrs(g[i,kei]<=Mbig*(1-delta[i,kei]) for i in V for kei in K)

#Constraint 42-44
model.addConstrs(p[0,kei]==u[0,kei] for kei in K)
model.addConstrs(p[2*n+2,kei]==u[2*n+2,kei]+(p[2*n+1,kei]-u[2*n+1,kei]) for kei in K)
#model.addConstrs((x[i,j,kei] == 1) >> (p[j,kei]==u[j,kei]+(p[i,kei]-u[i,kei])+t[i,j]*d[i,kei]-t[i,j]) for (i,j) in A if j!=i for kei in K)
model.addConstrs(p[j,kei]+s_tilde[i,j,kei]==u[j,kei]+(p[i,kei]-u[i,kei])+t[i,j]*d[i,kei]-t[i,j] for (i,j) in A if j!=i for kei in K)
model.addConstrs(s_tilde[i,j,kei]<=Mbig*(1-x[i,j,kei]) for (i,j) in A for kei in K)
model.addConstrs(s_tilde[i,j,kei]>=-Mbig*(1-x[i,j,kei]) for (i,j) in A for kei in K)

#Constraints 5-6
model.addConstrs(sum(x[h,j,kei] for (h,j) in A if h==0 )==1 for kei in K)
model.addConstrs(sum(x[h,j,kei] for (h,j) in A if h==2*n+2 )==1 for kei in K)
model.addConstrs(sum(x[j,h,kei] for (j,h) in A if h==2*n+1 )==1 for kei in K)
model.addConstrs(sum(x[j,h,kei] for (j,h) in A if h==2*n+3 )==1 for kei in K)
#Constraint 7
model.addConstrs(sum(x[i,h,kei] for (i,h) in A if h==hh) - sum(x[h,j,kei] for (h,j) in A if h==hh)==0 for hh in PcupD for kei in K)
#constraint 8
model.addConstrs(u[j,kei]>=u[i,kei]+t[i,j]-Mbig*(1-x[i,j,kei]) for (i,j) in A for kei in K)
#Constraint 9
model.addConstrs(int(e_bound[i])<=u[i,kei] for i in V for kei in K)
model.addConstrs(u[i,kei]<=l_bound[i] for i in V for kei in K)
#Constraint 10
model.addConstrs(eta[i,kei]-theta[i,kei]== sum(x[i,j,kei] for j in P+[2*n+1] if (i,j) in A and j!=i ) - sum(x[i+n,j,kei] for j in D+[2*n+3] if (i+n,j) in A and j!=i+n) for i in P for kei in K)
###wrong##model.addConstrs(eta[i,kei]-theta[i,kei]== sum(x[i,j,kei] for j in P+[2*n+1] if (i,j) in A ) - sum(x[i+n,j,kei] for j in D+[2*n+3] if (i,j) in A ) for i in P for kei in K)
#Constraint 11
model.addConstrs(eta[i,kei]+theta[i,kei]<=1 for i in P for kei in K)
#Constraint 12
model.addConstrs( (1/Mbig) * sum(eta[i,kei] for i in P) <= eta_k[kei] for kei in K )
model.addConstrs( eta_k[kei] <= sum(eta[i,kei] for i in P) for kei in K )
#Constraint 13
model.addConstrs( (1/Mbig) * sum(theta[i,kei] for i in P) <= theta_k[kei] for kei in K )
model.addConstrs( theta_k[kei] <= sum(theta[i,kei] for i in P) for kei in K )
#Constraint 14
model.addConstrs( tau[kei] == u[2*n+1,kei]+a_par*eta_k[kei]+b_par*sum(q[i]*eta[i,kei] for i in P) for kei in K)
#Constraint 15
model.addConstrs(w[kei]>=tau[kei] for kei in K)
#Constraint 16
model.addConstrs(u[2*n+2,kei]==w[kei]+a_par*theta_k[kei]+b_par*sum(q[i]*theta[i,kei] for i in P) for kei in K)
#Constraint 17
model.addConstrs(w[kei]>=z[i]-Mbig*(1-theta[i,kei]) for i in P for kei in K)
#Constraint 18
model.addConstrs(z[i]>=tau[kei]-Mbig*(1-eta[i,kei]) for i in P for kei in K)
#Constraint 19
model.addConstrs(u[2*n+1,kei]-u[0,kei] <= TT[kei] for kei in K)
#Constraint 20
model.addConstrs(u[2*n+3,kei]-u[2*n+2,kei] <= TT[kei] for kei in K)
#Constraint 22
model.addConstrs(r[i]<=LL for i in P)

#Valid Inequality constraints
model.addConstrs(u[i,kei]>=e_bound[i]+sum(max(0,e_bound[j]-e_bound[i]+t[i,j])*x[j,i,kei] for (j,i) in A if
                                          i==ii) for ii in PcupD for kei in K)
model.addConstrs(u[i,kei]<=l_bound[i]+sum(max(0,l_bound[i]-l_bound[j]+t[i,j])*x[j,i,kei] for (j,i) in A if
                                          i==ii) for ii in PcupD for kei in K)
model.addConstrs(r[i]>=t[i,2*n+1]+(a_par*theta_k[kei]+b_par*sum(q[i]*theta[i,kei] for i in P)) +t[2*n+2,n+i]
                 for i in P for kei in K)

#performance measuring
model.addConstr(vehiclecosts==sum(sum(c[i,j]*x[i,j,kei] for (i,j) in A) for kei in K))
model.addConstrs(route_duration_pickup[kei]==sum(c[i,j]*x[i,j,kei] for (i,j) in A0+A00+A000) for kei in K)
model.addConstrs(route_duration_delivery[kei]==sum(c[i,j]*x[i,j,kei] for (i,j) in A1+A11+A111) for kei in K)

#Declare objective function
obj = sum(sum(c[i,j]*x[i,j,kei] for (i,j) in A) for kei in K)

#Add objective function to model and declare that we solve a minimization problem
model.setObjective(obj,GRB.MINIMIZE)
model.Params.NodefileStart = 0.5
#model.Params.timeLimit = 900.0


#model.computeIIS()
#model.write("model.ilp")

#model.params.NonConvex = 2  # allow to handle quadratic equality constraints - which are always non-convex
model.optimize()
if model.status == GRB.OPTIMAL:  # check if the solver is capable of finding an optimal solution
    model.printAttr('X')
    print(model.status, 'optimal')
    print('Obj: %g' % model.objVal)
else:
    print(model.status, 'not optimal')

# print results
for v in model.getVars():
    if v.x > 0:
        print('%s %g' % (v.varName, v.x))


for k in K:
    route=[]
    for (i, j) in A0:
        if x[i,j,k].x>0.1:
            route.append(i); route.append(j)
            j_current=j
    while j_current!=2*n+1:
        for (i,j) in A00+A000:
            if i==j_current and x[i,j,k].x>0.1:
                route.append(j)
                j_current=j
    print('vehicle',k,'pick-up route',route)
for k in K:
    route=[]
    for (i, j) in A1:
        if x[i,j,k].x>0.1:
            route.append(i); route.append(j)
            j_current=j
    while j_current!=2*n+3:
        for (i,j) in A11+A111:
            if i==j_current and x[i,j,k].x>0.1:
                route.append(j)
                j_current=j
    print('vehicle',k,'delivery route',route)
