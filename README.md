Welcome to the README!

An overview:
- rddl_loader.py is used to load and run Jax-based planners on the domain/problem instances used in the report (and some extras).
- RDDL files in the MDP directory are fully-observable, continuous planning problems. 
- RDDL files in the POMDP directory are identical to those in the MDP directory, but have been modified to (theoretically) work with partial observability. Jax runs into issues with PO domains currently, so be aware this may not work yet!
- Details on the domain/instance files can be found in CISC_813_Project_Paper.pdf. It's not currently at the standard of a full report, but I got wrapped up in the implementation and ran out of time. The most important information is there though (rationale for model decisions, definitions, results, etc).

To reproduce results from the report:
- In rddl_loader.py, change instance_name to each of 'instance0', 'instance1', 'instance11', 'instance2' and 'instance3'.
- Run rddl_loader.py for each instance_name.
