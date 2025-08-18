Adding Your Seed Algorithms
###########################

We designed our LLM-GE framework to easily wrap around user-supplied datasets and source code.

To utilize LLM-GE there are a few steps you will need to take to integrate your algorithm and objectives:

#. We recommend placing your algorithmic components such as source code for training and evaluation of your algorithm within the `sota` folder.
#. You will need to modify :py:mod:`constants` in the following ways:
