from pyscipopt import Model

# Create a new SCIP model
model = Model("MyMIPModel")

# Define variables, objective, and constraints for your MIP problem here
# model.addVar("x", vtype="INTEGER", ...)
# model.setObjective(...)
# model.addCons(...)

# Step 1: Run with an initial time limit
model.setParam('limits/time', 600)  # 600 seconds = 10 minutes
model.optimize()

# Step 2: Check if the problem is solved to optimality or within 1% gap
status = model.getStatus()
gap = model.getGap()

if status == 'optimal' or gap <= 0.01:
    print("Solution is optimal or within 1% gap.")
else:
    # Step 3: Warm start with the solution from the last run
    # Get the current best solution
    current_solution = model.getBestSol()

    # Reset the model if you want to change parameters or re-solve from scratch
    model.freeTransform()

    # Optionally adjust the model (e.g., change parameters)
    model.setParam('limits/time', additional_time)  # Set a new time limit for the next run

    # Add the previous solution as a starting point
    model.addSol(current_solution)

    # Re-optimize the model
    model.optimize()

    # Check the new solution status and gap
    new_status = model.getStatus()
    new_gap = model.getGap()
    print(f"New status: {new_status}, New gap: {new_gap}")

    # You can repeat this process as needed, adjusting parameters or conditions for re-solving.
