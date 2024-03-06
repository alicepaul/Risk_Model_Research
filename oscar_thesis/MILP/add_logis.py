# Run Logistic Regression
    ## add cord
    log_reg = LogisticRegression()
    log_reg.fit(X, y)
    
    # Extract and round coefficients
    beta_start = np.round(log_reg.coef_[0]).astype(int)

    # Create a partial solution
    partial_solution = model.createPartialSol()

    # Set initial values for beta using beta_start
    for j, val in enumerate(beta_start):
        model.setSolVal(partial_solution, beta[j], val)

    # Calculate s values using X and beta_start
    for i in range(n):
        s_val = sum(beta_start[j] * X[i, j] for j in range(p))
        model.setSolVal(partial_solution, s[i], s_val)
    
    # Add the partial solution to the model
    model.addSol(partial_solution)

    # Set time limit 
    model.setRealParam('limits/time', 100)