Automating Turbulence Modeling by Multi-Agent Reinforcement Learning
********************************************************************

This folder contains all training and post-processing scripts, as well as settings file, used to obtain the results in the paper 
`Automating Turbulence Modeling by Multi-Agent Reinforcement Learning <https://arxiv.org/pdf/2005.09023.pdf>`_. 
Because ``smarties`` is a library, the main executable that simulates the flow is produced by 
`CubismUP 3D <https://github.com/cselab/CubismUP_3D>`_.
We refer to that page for instructions regarding install and dependencies. 
The dependencies of ``CubismUP 3D`` are a superset of those of ``smarties``.

All the scripts in this folder assume that:  

- The directories of ``smarties`` and ``CubismUP 3D`` are placed at the same path (e.g. ``${HOME}/``).
- The repository ``MARL_LES`` has been placed in ``smarties/apps/``. E.g.:

.. code:: shell

    git clone --recursive https://github.com/cselab/CubismUP_3D.git
    git clone --recursive https://github.com/cselab/smarties.git
    cd smarties/apps/
    git clone --recursive https://github.com/cselab/MARL_LES.git

- ``smarties`` is installed and ``CubismUP 3D`` can be compiled without issues.

Core task description
=====================
The ``CubismUP 3D`` file ``source/main_RL_HIT.cpp`` produces the main executable.
It describes the environment loop and parses all hyper-parameters.
This file interacts with 3 objects:

- The class ``smarties::Communicator`` receives a description of the RL problem and handles the state-action loop for all the agents.   
- The class ``cubismup3d::Simulation`` comprises the solver and defines the operations performed on each (simulation) time-step.   
- The class ``cubismup3d::SGS_RL`` describes the operations performed to update the Smagorinsky coefficients. This class describes both the interpolation of the actions onto the grid and the calculation of the state components.


In this folder, the file ``setup.sh`` is read by ``smarties.py`` and prepares all hyper-parameters, and simulation description.

Training script
===============
Training can be easily started, with default hyper-parameters, as:

.. code:: shell

    smarties.py MARL_LES -r training_directory_name

In order to reproduce the number of gradient steps of the paper on a personal computer, the script may run for days, 
with large uncertainty due to the specific processor and software stack. We relied on the computational resources provided by
the Swiss National Supercomputing Centre (CSCS) (on the Piz Daint supercomputer).
We provide a set of trained policy parameters and restart folder.
The helper file ``launch_all_les_blocksizes.sh`` was used to evaluate  multiple hyper-parameter choices.

By default, ``smarties.py`` will place all run directories in ``${SMARTIES_ROOT}/runs``, but can be changed with
the argument ``--runprefix``.

When training, the terminal output will be that of ``smarties.py``, which tracks training progress, not of ``CubismUP 3D``.
The terminal output of the simulations is redirected to, for example, ``training_directory_name/simulation_000_00000/output`` and 
all the simulation snapshots and post-processing to, for example, ``training_directory_name/simulation_000_00000/run_00000000/``.
During training, no post-processing (e.g. energy spectra, dissipation, other integral statistics) are stored to file.

Running the trained model
==========================
Once trained, the policy can be used to perform any simulation. This can be done for example as:

.. code:: shell

    smarties.py MARL_LES -r evaluation_directory_name --restart training_directory_name --nEvalEpisodes 1

This process should take few minutes. Again, the terminal output will be that of ``smarties.py``,
which, if everything works correctly, will not be very informative.
To see the terminal output of the simulation itself prepend ``--printAppStdout`` to the run command.

Because we specificed that we evaluate the policy for 1 episode (or N), training is disabled and the policy is fixed.
However, the ``CubismUP_3D`` side will run identically as for training, which means that it will simulate a random Reynolds number.
Using the script

.. code:: shell

    python3 eval_all_train.py training_directory_name

will evaluate the policy saved in ``training_directory_name`` at Reynolds in log-intervals from 60 to 205, each in a separate directory.
Each evaluation directory will be named according to the Reynolds like: ``training_directory_name_RE%03d``.
To evaluate the trained policy provided with this repository, you can use the command:

.. code:: shell

    python3 eval_all_train.py trained_BlockAgents_FFNN_4blocks_act08_sim20 --restartsPath ./

This problem should take several minutes per Reynolds number, again depending on the software stack and CPU (should be under one hour with a modern laptop per simulation). To speed things up, you may evaluate on a subset of Reynolds by modidfying the ``eval_all_train.py`` script itself.

Evaluating the trained policy
==============================
From the list of directories, the energy spectra can be plotted as

.. code:: shell

    python3 plot_ll_les_error.py training_directory_name --runspath /rel/path/to/runs/ --res 65, 76, 88, 103, 120, 140, 163

Here we need to write the relative path to where ``smarties.py`` has created the evaluation runs.
By default, all the directories were placed in ``${SMARTIES_ROOT}/runs``.

Reproducing the plots on the paper
==================================
To produce the figures 3a and 3b:

.. code:: shell

    ./plot_ll_les_error.py SSM DSM GERMANO LL SINGLE

To produce figure 3c:

.. code:: shell

    ./plot_histograms_CSS.py LL GERMANO DSM DNS

To produce figures 3d and 3e:

.. code:: shell

    ./plot_compute_structure.py DNS LL DSM SSM

To produce figures 4a, 4b, 4c, 4d:

.. code:: shell

    ./plot_integral_quantities.py LL DSM SSM

To produce figure 4e:

.. code:: shell

    ./plot_ll_error_gridsizes.py --tokens LL --gridSize 32 64 128 --refs DSM

To produce figures 6a and 6b:

.. code:: shell

    ./plot_spectra.py

To produce figure 6c:

.. code:: shell

    ./plot_energy_modes_histrograms.py  data/HITDNS_RK_UW_CFL010_BPD32

NOTE: In order to limit the size of the repository, only data produced by one DNS simulation was included. Therefore, this last figure will show noisier histograms than the ones shown in the second row of fig 6. As stated in the paper, that figure was produced with data from 20 DNS simulations.