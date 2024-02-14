# detect anormal seed
# Define the range of seed values to test
seed_range <- 1:100000000

# Loop over seed values
for (seed in seed_range) {
  # Set the seed
  set.seed(seed)
  
  # Run your function
  result <- tryCatch({
    print(seed)
    run_experiments("/Users/oscar/Documents/GitHub/Risk_Model_Research/ncd_milp/sim_newalg_1/")
  }, error = function(e) {
    # Print an error message if the function fails
    message(paste("Seed", seed, "causes an error:", conditionMessage(e)))
    NULL  # Return NULL to indicate failure
  })
}

# Or add the following somewhere in the function
# Get the current seed number
current_seed <- .Random.seed

# Print the current seed number
print(current_seed)

# then copy paste the seed number
.Random.seed <- copy.seed
