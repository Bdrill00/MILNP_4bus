from pyomo.environ import *

# Create a concrete model
model = ConcreteModel()

# Define variables
model.x = Var(initialize=1.5)
model.y = Var(initialize=1.5)

# Define objective function
def rosenbrock(model):
    return (1.0 - model.x)**2 + 100.0 * (model.y - model.x**2)**2
model.obj = Objective(rule=rosenbrock, sense=minimize)

# Solve the model using BARON
opt = SolverFactory('baron')
results = opt.solve(model, tee=True)

# Print the results
model.pprint()