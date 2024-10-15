# Finite-Hypothesis-Class
{
 "cells": [
  {
   "cell_type": "markdown",
   "id": "70352e3c-2713-4f7f-889e-a18c1bf280e6",
   "metadata": {},
   "source": [
    "# Finite Hypothesis Classes\n",
    "\n",
    "In this homework, you will explore how to implement and use finite hypothesis classes.\n",
    "You will use the famous MNIST dataset for handwritten digit recognition.\n",
    "\n",
    "<img src=\"img/mnist.png\" />\n",
    "\n",
    "With the simple finite hypothesis classes we've studied so far, you should be able to get about 90% accuracy on this problem.\n",
    "By the end of the semester, you'll be able to get state-of-the-art (SOTA) accuracy of >99%.\n",
    "\n",
    "**Instructions:**\n",
    "You should read through the code starting with Part 1 below.\n",
    "The comments contain detailed descriptions of what the code is doing.\n",
    "There are three FIXME annotations in the comments (in parts 1, 3, and 4).\n",
    "You should complete the tasks specified by those FIXME annotations by modifying the jupyter notebook directly.\n",
    "Once you've completed all the tasks,\n",
    "create a new github repo and upload this notebook to github.\n",
    "Submit the assignment by copying the link to your repo in sakai."
   ]
  },
  {
   "cell_type": "markdown",
   "id": "229b33e5-37f7-4023-872b-b06e97dfbf23",
   "metadata": {},
   "source": [
    "## Part 0: Background\n",
    "\n",
    "This section contains imports and helper functions used by the code in parts 1+.\n",
    "It is safe to skip reading the code in this section for now,\n",
    "but you will likely have to refer to it later to see how to use the helper functions."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 1,
   "id": "75b19d3c-f3d6-445a-b7c7-00febf80477b",
   "metadata": {},
   "outputs": [],
   "source": [
    "import math\n",
    "import sklearn\n",
    "import sklearn.metrics\n",
    "import time\n",
    "\n",
    "# make numpy random numbers deterministic\n",
    "import numpy as np\n",
    "np.random.seed(0)\n",
    "\n",
    "# enable plotting\n",
    "import matplotlib.pyplot as plt\n",
    "%matplotlib inline"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "id": "edf84e74-8bbc-432f-8b3a-282623776629",
   "metadata": {},
   "outputs": [],
   "source": [
    "def sign(a):\n",
    "    '''\n",
    "    Convert a boolean value into +/- 1.\n",
    "\n",
    "    >>> sign(12.5)\n",
    "    1\n",
    "    >>> sign(-12.5)\n",
    "    -1\n",
    "    '''\n",
    "    if a > 0:\n",
    "        return 1\n",
    "    if a <= 0:\n",
    "        return -1\n",
    "\n",
    "\n",
    "def set_exp(xs, p):\n",
    "    '''\n",
    "    Compute the \"set exponential\" function.\n",
    "\n",
    "    For efficiency, this function is a generator.\n",
    "    This means that large sets will never be explicitly stored,\n",
    "    and this function will always use O(1) memory.\n",
    "\n",
    "    The doctests below first convert the generator into a list for visualization.\n",
    "\n",
    "    >>> list(set_exp([-1, +1], 0))\n",
    "    []\n",
    "    >>> list(set_exp([-1, +1], 1))\n",
    "    [[-1], [1]]\n",
    "    >>> list(set_exp([-1, +1], 2))\n",
    "    [[-1, -1], [1, -1], [-1, 1], [1, 1]]\n",
    "    >>> list(set_exp([-1, +1], 3))\n",
    "    [[-1, -1, -1], [1, -1, -1], [-1, 1, -1], [1, 1, -1], [-1, -1, 1], [1, -1, 1], [-1, 1, 1], [1, 1, 1]]\n",
    "\n",
    "    Observe that the length grows exponentially with the power.\n",
    "\n",
    "    >>> len(list(set_exp([-1, +1], 4)))\n",
    "    16\n",
    "    >>> len(list(set_exp([-1, +1], 5)))\n",
    "    32\n",
    "    >>> len(list(set_exp([-1, +1], 6)))\n",
    "    64\n",
    "    >>> len(list(set_exp([-1, +1], 7)))\n",
    "    128\n",
    "    >>> len(list(set_exp([-1, +1], 8)))\n",
    "    256\n",
    "    '''\n",
    "    assert(len(xs) > 0)\n",
    "    assert(p >= 0)\n",
    "    assert(type(p) == int)\n",
    "    if p == 1:\n",
    "        for x in xs:\n",
    "            yield [x]\n",
    "    elif p > 1:\n",
    "        for x in xs:\n",
    "            for ys in set_exp(xs, p - 1):\n",
    "                yield ys + [x]"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "659e448f-31d7-4870-8904-5268472c6739",
   "metadata": {},
   "source": [
    "## Part 1: Hypothesis Classes\n",
    "\n",
    "This section explores how to translate the mathematical definitions of the finite hypothesis classes into python code."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "id": "b8de7c07-8eb3-44b4-9316-7566fed7ba26",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "ypred=1\n",
      "ypred=-1\n"
     ]
    }
   ],
   "source": [
    "# The H_binary hypothesis class is easy to represent in code as a list of anonymous functions\n",
    "H_binary = [lambda x: 1, lambda x: -1]\n",
    "\n",
    "# The code below shows how to use one of these functions.\n",
    "# First we define an example datapoint.\n",
    "# Then we apply every hypothesis h in the hypothesis class to the sample.\n",
    "x = np.array([12.5, -12.5])\n",
    "for h in H_binary:\n",
    "    ypred = h(x)\n",
    "    print(f'ypred={ypred}')\n",
    "    \n",
    "# You should ensure that you understand the output of this code."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "id": "987ac5a7-5c3f-4a75-be44-12b4310ca51b",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "ypred=-1\n",
      "ypred=-1\n"
     ]
    }
   ],
   "source": [
    "# H_axis is harder to represent in code;\n",
    "# the \"obvious\" thing to do is something like the following two lines of code\n",
    "d = 2\n",
    "H_axis = [lambda x: sign(x[i]) for i in range(d)]\n",
    "\n",
    "# unfortunately, there is a serious bug in this code\n",
    "# to illustrate the bug, let's try applying every hypothesis to the same data point as above\n",
    "x = np.array([12.5, -12.5])\n",
    "for h in H_axis:\n",
    "    ypred = h(x)\n",
    "    print(f'ypred={ypred}')\n",
    "\n",
    "# Notice that the output is the same for both functions,\n",
    "# but the output *should* be different.\n",
    "# That's because on this particular dataset, sign(x[0]) should be +1 and sign(x[1]) should be -1.\n",
    "# the problem is that we're getting -1 twice"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "id": "1cc4b663-98af-4f57-a1db-9a947e962ff1",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "ypred=1\n",
      "ypred=-1\n"
     ]
    }
   ],
   "source": [
    "# the problem observed above is related to something called a closure in python\n",
    "# closures are a fundamental part of modern programming,\n",
    "# and it is extremely important that you understand how they work.\n",
    "# it is common in technical interviews for SWE positions to be asked syntax-related questions about closures\n",
    "\n",
    "# for the specific case of using a lambda inside of a list comprehension,\n",
    "# you can find a detailed explanation of what is happening on stackoverflow at\n",
    "# <https://stackoverflow.com/questions/28268439/python-list-comprehension-with-lambdas>\n",
    "\n",
    "# to fix our problem, we can use a default argument to change the scope of the variable i\n",
    "# in general, when translating set builder notation into python,\n",
    "# it is necessary to do this step for all variables that you are \"looping over\"\n",
    "H_axis = [lambda x, i=i: sign(x[i]) for i in range(d)]\n",
    "\n",
    "# now observe that we get the correct output\n",
    "x = np.array([12.5, -12.5])\n",
    "for h in H_axis:\n",
    "    ypred = h(x)\n",
    "    print(f'ypred={ypred}')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "id": "d7a19dae-4953-4b87-9ef1-bc787d288b9a",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "H_binary\n",
      "ypred=1\n",
      "ypred=-1\n",
      "H_axis\n",
      "ypred=1\n",
      "ypred=-1\n",
      "ypred=1\n",
      "ypred=-1\n"
     ]
    }
   ],
   "source": [
    "# there's one last minor problem with our definitions of the H_binary and H_axis hypothesis classes above:\n",
    "# we've hard-coded the number of dimensions.\n",
    "# we can make our hypothesis classes more generic by wrapping them in another lambda to specify the dimension\n",
    "\n",
    "H_binary = lambda d: [lambda x: 1, lambda x: -1]\n",
    "H_axis = lambda d: [lambda x, i=i: sign(x[i]) for i in range(d)]\n",
    "\n",
    "# now to use the class, we need to specify the number of dimensions;\n",
    "# here's an example in 4 dimensions\n",
    "x = np.array([12.5, -12.5, 1.2, -1.2])\n",
    "print('H_binary')\n",
    "for h in H_binary(4):\n",
    "    ypred = h(x)\n",
    "    print(f'ypred={ypred}')\n",
    "\n",
    "print('H_axis')\n",
    "for h in H_axis(4):\n",
    "    ypred = h(x)\n",
    "    print(f'ypred={ypred}')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "id": "b44d063c-4ac7-4b1e-9900-fc25f11f6407",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "2\n",
      "2\n"
     ]
    }
   ],
   "source": [
    "# notice that the size of H_binary doesn't change based on the number of dimensions,\n",
    "# but that the size of H_axis does\n",
    "print(len(H_binary(2)))\n",
    "print(len(H_axis(2)))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "id": "60e44e40-0665-4b97-b150-7eab0be5e993",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "  d     len(H_binary(d))      len(H_axis(d))     len(H_axis2(d))     len(H_multiaxis2(d))     len(H_multiaxis3(d))\n",
      "  1                    2                   1                   0                        0                        0\n",
      "  2                    2                   2                   0                        0                        0\n",
      "  3                    2                   3                   0                        0                        0\n",
      "  4                    2                   4                   0                        0                        0\n",
      "  5                    2                   5                   0                        0                        0\n",
      "  6                    2                   6                   0                        0                        0\n",
      "  7                    2                   7                   0                        0                        0\n",
      "  8                    2                   8                   0                        0                        0\n",
      "  9                    2                   9                   0                        0                        0\n",
      " 10                    2                  10                   0                        0                        0\n"
     ]
    }
   ],
   "source": [
    "# FIXME:\n",
    "# define the hypothesis classes below\n",
    "# HINT:\n",
    "# the multiaxis* hypothesis classes should loop over a sigma vector created from the set_exp function defined above\n",
    "H_binary = lambda d: [lambda x: +1, lambda x: -1]\n",
    "H_axis = lambda d: [lambda x, i=i: sign(x[i]) for i in range(d)]\n",
    "H_axis2 = lambda d: []\n",
    "H_multiaxis2 = lambda d: []\n",
    "H_multiaxis3 = lambda d: []\n",
    "\n",
    "# the following code prints a nice table showing the size of the finite hypothesis classes in different dimensions\n",
    "# you will know your implementations above are correct if the sizes match the formulas we derived in the notes \n",
    "print(f'  d {\"len(H_binary(d))\":>20}{\"len(H_axis(d))\":>20}{\"len(H_axis2(d))\":>20}{\"len(H_multiaxis2(d))\":>25}{\"len(H_multiaxis3(d))\":>25}')\n",
    "for d in range(1, 11):\n",
    "    print(f' {d:2} {len(H_binary(d)):20d}{len(H_axis(d)):20d}{len(H_axis2(d)):20d}{len(H_multiaxis2(d)):25d}{len(H_multiaxis3(d)):25d}')"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "d7d195ae-06fb-4703-adaf-dbf27e019c19",
   "metadata": {},
   "source": [
    "## Part 2: Loading the MNIST Data\n",
    "\n",
    "This section loads the dataset.\n",
    "There are no FIXME annotations in this section.\n",
    "It would still be useful to skim this section, however, because the subsequent sections rely on the dataset."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 9,
   "id": "982aca51-8b83-4a41-bcbe-02fe09d9fca5",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "X.shape=(14867, 784)\n",
      "y.shape=(14867,)\n"
     ]
    }
   ],
   "source": [
    "# scikit learn has built-in functions for loading lots of standard datasets\n",
    "# the MNIST dataset is small by machine learning standards,\n",
    "# but it still takes 10-20 seconds to load on my machine\n",
    "from sklearn.datasets import fetch_openml\n",
    "X, y = fetch_openml('mnist_784', version=1, return_X_y=True)\n",
    "X = X.to_numpy()\n",
    "y = y.to_numpy()\n",
    "\n",
    "# the standard MNIST problem has 10 classes, one for each numeric digit\n",
    "# we've only studied binary classification in class so far,\n",
    "# so we'll convert MNIST into a binary problem\n",
    "# the idea is that we will label all \"1\"s as +1, all \"2\"s as -1, and delete all othe other digits\n",
    "label_positive = '1'\n",
    "label_negative = '2'\n",
    "dataset_mask = np.logical_or(y == label_positive, y == label_negative)\n",
    "X = X[dataset_mask]\n",
    "y = y[dataset_mask]\n",
    "y[y == label_positive] = +1\n",
    "y[y == label_negative] = -1\n",
    "y = y.astype(np.int64)\n",
    "\n",
    "# for debugging purposes, I always print the shape of my important tensors\n",
    "print(f\"X.shape={X.shape}\") # shape = N x d\n",
    "print(f\"y.shape={y.shape}\") # shape = N"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 10,
   "id": "4612a35d-05d4-4f67-9491-e88c3fc21f0a",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "i=0, y[i]=1, x[i]=\n"
     ]
    },
    {
     "data": {
      "image/png": "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",
      "text/plain": [
       "<Figure size 100x100 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "i=1, y[i]=-1, x[i]=\n"
     ]
    },
    {
     "data": {
      "image/png": "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",
      "text/plain": [
       "<Figure size 100x100 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "i=2, y[i]=1, x[i]=\n"
     ]
    },
    {
     "data": {
      "image/png": "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",
      "text/plain": [
       "<Figure size 100x100 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "i=3, y[i]=1, x[i]=\n"
     ]
    },
    {
     "data": {
      "image/png": "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",
      "text/plain": [
       "<Figure size 100x100 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "i=4, y[i]=1, x[i]=\n"
     ]
    },
    {
     "data": {
      "image/png": "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",
      "text/plain": [
       "<Figure size 100x100 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "# it's always a good idea to visualize your datapoints after loading them\n",
    "# this lets us sanity check that our labels are in fact correct\n",
    "for i in range(5):\n",
    "    print(f\"i={i}, y[i]={y[i]}, x[i]=\")\n",
    "    image = X[i].reshape([28,28])\n",
    "    fig = plt.figure(figsize=(1,1))\n",
    "    plt.axis('off')\n",
    "    plt.imshow(image, cmap='gray')\n",
    "    plt.show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 11,
   "id": "a2be2af7-b4bc-4408-b5bc-4779cd8f4e24",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Xtrain.shape=(1000, 784)\n",
      "ytrain.shape=(1000,)\n",
      "Xtest.shape=(10000, 784)\n",
      "ytest.shape=(10000,)\n"
     ]
    }
   ],
   "source": [
    "# we now split the dataset into a training a testing datasets\n",
    "# we will use a relatively small N value (just to make your future experiments faster)\n",
    "# we also have a large Ntest value, so we can be fairly confident that |Etest - Eout| is small\n",
    "N = 1000\n",
    "Ntest = 10000\n",
    "Xtrain = X[:N]\n",
    "ytrain = y[:N]\n",
    "Xtest = X[N:N+Ntest]\n",
    "ytest = y[N:N+Ntest]\n",
    "print(f\"Xtrain.shape={Xtrain.shape}\")\n",
    "print(f\"ytrain.shape={ytrain.shape}\")\n",
    "print(f\"Xtest.shape={Xtest.shape}\")\n",
    "print(f\"ytest.shape={ytest.shape}\")"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "6cc413fc-ad13-43dc-883b-6378b3e7f7e6",
   "metadata": {},
   "source": [
    "## Part 3: Model Training\n",
    "\n",
    "This section introduces how to implmement and use scikit learn models.\n",
    "In general, you won't have to implement the more \"interesting\" models because they are already implemented.\n",
    "But you will have to implement the TEA model.\n",
    "The main purpose of this task is to just get you familiar with how scikit learn is structured so that you will be able to effectively use it later on."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 12,
   "id": "c9d0faad-c764-47a6-bb6b-895d39920f8a",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "runtime=0.0 seconds\n",
      "Ein=0.5290\n",
      "Etest=0.5335\n",
      "generalization error=0.0045\n"
     ]
    }
   ],
   "source": [
    "# in scikit learn, all learning models are implemented as a class\n",
    "# these classes are called \"estimators\",\n",
    "# and they follow the interface specified in <https://scikit-learn.org/dev/developers/develop.html>\n",
    "# in particular, all estimators will need at least three methods:\n",
    "# (1) __init__ specifies the hyperparameters to the model\n",
    "# (2) fit takes the training dataset as input and computes the hypothesis in the hypothesis class\n",
    "# (3) predict applies the hypothesis to an input datapoint or dataset\n",
    "\n",
    "class TEA:\n",
    "    def __init__(self, H):\n",
    "        self.H = H\n",
    "        self.g = None\n",
    "\n",
    "    def fit(self, X, Y):\n",
    "        # FIXME:\n",
    "        # implement the fit method following the pseudocode for TEA from the notes;\n",
    "        # your implementation should store the final hypothesis in the self.g variable\n",
    "        # HINT:\n",
    "        # everything so far is deterministic, so your code will be correct if it gets the exact same results as mine\n",
    "        # my results are\n",
    "        # for H_binary: Ein=0.4710\n",
    "        # for H_axis: 0.1540\n",
    "        # for H_axis2: 0.1540\n",
    "        # you won't be able to use the H_multiaxis2 or H_multiaxis3 hypothesis classes because of the exponential runtimes\n",
    "        d = X.shape[1]\n",
    "        self.g = self.H(d)[0]\n",
    "    \n",
    "    def predict(self, X):\n",
    "        assert(self.g is not None)\n",
    "        if len(X.shape) == 1:\n",
    "            return self.g(X)\n",
    "        else:\n",
    "            return np.apply_along_axis(self.g, 1, X)\n",
    "\n",
    "# we now train the model by passing the training data to the .fit method\n",
    "model = TEA(H_axis) # replace H_axis with other hypothesis classes\n",
    "time_start = time.time()\n",
    "model.fit(Xtrain, ytrain)\n",
    "time_end = time.time()\n",
    "runtime = time_end - time_start\n",
    "print(f\"runtime={runtime:0.1f} seconds\")\n",
    "\n",
    "# report the relavent metrics\n",
    "# scikit learn does not have an error metric built in, \n",
    "# but we can compute it as 1 - accuracy\n",
    "ytrain_pred = model.predict(Xtrain)\n",
    "Ein = 1 - sklearn.metrics.accuracy_score(ytrain_pred, ytrain)\n",
    "print(f\"Ein={Ein:0.4f}\")\n",
    "ytest_pred = model.predict(Xtest)\n",
    "Etest = 1 - sklearn.metrics.accuracy_score(ytest, ytest_pred)\n",
    "print(f\"Etest={Etest:0.4f}\")\n",
    "print(f'generalization error={abs(Etest - Ein):0.4f}')"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "3a24c2cd-7e25-43a7-824b-1fb5bef8fde5",
   "metadata": {},
   "source": [
    "## Part 4: Data preprocessing with random projections"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 13,
   "id": "ebc3a5bd-0688-41cf-91fe-4376a892e989",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Afull.shape=(784, 784)\n"
     ]
    }
   ],
   "source": [
    "# recall that one of the most important parts of this class is learning how to effectively preprocess data\n",
    "# in the real world, you will never have to implement your own machine learning algorithms\n",
    "# existing implementations in libraries like scikit learn are highly optimized,\n",
    "# and there's no need to reinvent the wheel.\n",
    "# but you will have to preprocess data\n",
    "\n",
    "# one of the most important forms of preprocessing is the random projection\n",
    "# recall that projecting the data onto a smaller dimension (dprime) will cause:\n",
    "# (1) the runtime for TEA to go down,\n",
    "# (2) the generalization error |Ein - Eout| to go down,\n",
    "# (3) the training error Ein to go up\n",
    "# every application will have different demands on these quantities,\n",
    "# and so you will have to choose an appropriate dimension dprime for your application\n",
    "\n",
    "# to project the data, we need to generate a random d x dprime matrix\n",
    "# we use two tricks to generate the random matrix deterministically:\n",
    "# first, we set the seed; this ensures that every run of this cell will create the same matrix\n",
    "# (although the same seed will result in different matrices on different computers)\n",
    "# second, we create a full matrix and then slice it to the appropriate size\n",
    "# this ensures that the value of dprime does not affect the contents of A\n",
    "np.random.seed(0)\n",
    "d = X.shape[1]\n",
    "Afull = np.random.uniform(low=-1, high=1, size=(d, d))\n",
    "print(f'Afull.shape={Afull.shape}') # shape = d x d"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 14,
   "id": "8174cc89-cfd3-4374-90ab-83823dca5c5e",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Xprime.shape=(14867, 5)\n",
      "i=0, y[i]=1, x[i]=\n"
     ]
    },
    {
     "data": {
      "image/png": "iVBORw0KGgoAAAANSUhEUgAAAPwAAABCCAYAAABza7tiAAAAOXRFWHRTb2Z0d2FyZQBNYXRwbG90bGliIHZlcnNpb24zLjMuNCwgaHR0cHM6Ly9tYXRwbG90bGliLm9yZy8QVMy6AAAACXBIWXMAAA9hAAAPYQGoP6dpAAABEElEQVR4nO3asQ2EMBQFwfPp+k+pgzqohRp8LSAkcLAzsYOXrH7iMeecHyDhu3oA8B7BQ4jgIUTwECJ4CBE8hAgeQgQPIb+rD8/zfHLHY/Z9Xz3htm3bVk+47TiO1RNuGWOsnnDblT90LjyECB5CBA8hgocQwUOI4CFE8BAieAgRPIQIHkIEDyGChxDBQ4jgIUTwECJ4CBE8hAgeQgQPIYKHEMFDiOAhRPAQIngIETyECB5CBA8hgocQwUOI4CFE8BAieAgRPIQIHkIEDyGChxDBQ4jgIUTwECJ4CBE8hAgeQgQPIYKHEMFDiOAhZMw55+oRwDtceAgRPIQIHkIEDyGChxDBQ4jgIUTwECJ4CPkDyXkUfXChLX8AAAAASUVORK5CYII=",
      "text/plain": [
       "<Figure size 300x300 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "i=1, y[i]=-1, x[i]=\n"
     ]
    },
    {
     "data": {
      "image/png": "iVBORw0KGgoAAAANSUhEUgAAAPwAAABCCAYAAABza7tiAAAAOXRFWHRTb2Z0d2FyZQBNYXRwbG90bGliIHZlcnNpb24zLjMuNCwgaHR0cHM6Ly9tYXRwbG90bGliLm9yZy8QVMy6AAAACXBIWXMAAA9hAAAPYQGoP6dpAAABD0lEQVR4nO3awQ3CMBQFQYzoPB2lnTSRDkwJoEjBQjtz9uFdVv/iMeecDyDhuXoA8DuChxDBQ4jgIUTwECJ4CBE8hAgeQl7fPhxj3LnjNv/8r2jbttUTLjuOY/WES87zXD3hsn3fP75x4SFE8BAieAgRPIQIHkIEDyGChxDBQ4jgIUTwECJ4CBE8hAgeQgQPIYKHEMFDiOAhRPAQIngIETyECB5CBA8hgocQwUOI4CFE8BAieAgRPIQIHkIEDyGChxDBQ4jgIUTwECJ4CBE8hAgeQgQPIYKHEMFDiOAhRPAQIngIETyECB5CBA8hY845V48AfsOFhxDBQ4jgIUTwECJ4CBE8hAgeQgQPIYKHkDc0YhR9ANPFoAAAAABJRU5ErkJggg==",
      "text/plain": [
       "<Figure size 300x300 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "i=2, y[i]=1, x[i]=\n"
     ]
    },
    {
     "data": {
      "image/png": "iVBORw0KGgoAAAANSUhEUgAAAPwAAABCCAYAAABza7tiAAAAOXRFWHRTb2Z0d2FyZQBNYXRwbG90bGliIHZlcnNpb24zLjMuNCwgaHR0cHM6Ly9tYXRwbG90bGliLm9yZy8QVMy6AAAACXBIWXMAAA9hAAAPYQGoP6dpAAABEElEQVR4nO3asQ2DQBQFQWO5KDKqoClaochzCwgJLtiZ+IKXrH5yyxhjfICE7+wBwHsEDyGChxDBQ4jgIUTwECJ4CBE8hPyuPjzP88kdj9n3ffaE247jmD3htm3bZk+4ZV3X2RNuu/KHzoWHEMFDiOAhRPAQIngIETyECB5CBA8hgocQwUOI4CFE8BAieAgRPIQIHkIEDyGChxDBQ4jgIUTwECJ4CBE8hAgeQgQPIYKHEMFDiOAhRPAQIngIETyECB5CBA8hgocQwUOI4CFE8BAieAgRPIQIHkIEDyGChxDBQ4jgIUTwECJ4CBE8hCxjjDF7BPAOFx5CBA8hgocQwUOI4CFE8BAieAgRPIQIHkL+22wRfRx+BzIAAAAASUVORK5CYII=",
      "text/plain": [
       "<Figure size 300x300 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "i=3, y[i]=1, x[i]=\n"
     ]
    },
    {
     "data": {
      "image/png": "iVBORw0KGgoAAAANSUhEUgAAAPwAAABCCAYAAABza7tiAAAAOXRFWHRTb2Z0d2FyZQBNYXRwbG90bGliIHZlcnNpb24zLjMuNCwgaHR0cHM6Ly9tYXRwbG90bGliLm9yZy8QVMy6AAAACXBIWXMAAA9hAAAPYQGoP6dpAAABDUlEQVR4nO3VwQ3CQBAEQQ6RqmNwCo7ByR4hgCyZE+qq9z7m09ox55wPIOG5egDwO4KHEMFDiOAhRPAQIngIETyECB5CXt8ebtt2547bnOe5esJlY4zVEy771+37vq+ecNlxHB9vfHgIETyECB5CBA8hgocQwUOI4CFE8BAieAgRPIQIHkIEDyGChxDBQ4jgIUTwECJ4CBE8hAgeQgQPIYKHEMFDiOAhRPAQIngIETyECB5CBA8hgocQwUOI4CFE8BAieAgRPIQIHkIEDyGChxDBQ4jgIUTwECJ4CBE8hAgeQgQPIYKHEMFDyJhzztUjgN/w4SFE8BAieAgRPIQIHkIEDyGChxDBQ4jgIeQNfRcOfR3U6DgAAAAASUVORK5CYII=",
      "text/plain": [
       "<Figure size 300x300 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "i=4, y[i]=1, x[i]=\n"
     ]
    },
    {
     "data": {
      "image/png": "iVBORw0KGgoAAAANSUhEUgAAAPwAAABCCAYAAABza7tiAAAAOXRFWHRTb2Z0d2FyZQBNYXRwbG90bGliIHZlcnNpb24zLjMuNCwgaHR0cHM6Ly9tYXRwbG90bGliLm9yZy8QVMy6AAAACXBIWXMAAA9hAAAPYQGoP6dpAAABDklEQVR4nO3awQnDMBQFwSgE9+0S3KW7UEpIMDgi7MxZh3dZ/kVjzjkfQMJz9QDgdwQPIYKHEMFDiOAhRPAQIngIETyEvL59OMa4c8dt/vlf0XEcqydcdp7n6gmX7Pu+esJl27Z9fOPCQ4jgIUTwECJ4CBE8hAgeQgQPIYKHEMFDiOAhRPAQIngIETyECB5CBA8hgocQwUOI4CFE8BAieAgRPIQIHkIEDyGChxDBQ4jgIUTwECJ4CBE8hAgeQgQPIYKHEMFDiOAhRPAQIngIETyECB5CBA8hgocQwUOI4CFE8BAieAgRPIQIHkLGnHOuHgH8hgsPIYKHEMFDiOAhRPAQIngIETyECB5CBA8hbxyJEX0kJty3AAAAAElFTkSuQmCC",
      "text/plain": [
       "<Figure size 300x300 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "# the code below plots the data points after they have been projected down into a very small dprime\n",
    "dprime = 5\n",
    "A = Afull[:, :dprime]\n",
    "Xprime = X @ A\n",
    "print(f\"Xprime.shape={Xprime.shape}\")\n",
    "for i in range(5):\n",
    "    print(f\"i={i}, y[i]={y[i]}, x[i]=\")\n",
    "    image = Xprime[i].reshape([1, dprime])\n",
    "    fig = plt.figure(figsize=(3,3))\n",
    "    plt.axis('off')\n",
    "    plt.imshow(image, cmap='gray')\n",
    "    plt.show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 17,
   "id": "c32f1c98-b2d8-41d1-a1da-48717613855d",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "runtime=0.0 seconds\n",
      "Ein=0.4970\n",
      "Etest=0.5186\n",
      "generalization error=0.0216\n"
     ]
    }
   ],
   "source": [
    "# the images above are no longer human interpretable;\n",
    "# remarkably, we can still learn an effective hypothesis on these newly transformed data points\n",
    "# with dprime=5, my results on H_axis2 are nearly as good as they were for the full dataset\n",
    "# on this smaller dataset size, we can actually now try the H_multiaxis* hypothesis classes as well\n",
    "\n",
    "model = TEA(H_axis)\n",
    "time_start = time.time()\n",
    "model.fit(Xtrain @ A, ytrain)\n",
    "time_end = time.time()\n",
    "runtime = time_end - time_start\n",
    "print(f\"runtime={runtime:0.1f} seconds\")\n",
    "\n",
    "ytrain_pred = model.predict(Xtrain @ A)\n",
    "Ein = 1 - sklearn.metrics.accuracy_score(ytrain_pred, ytrain)\n",
    "print(f\"Ein={Ein:0.4f}\")\n",
    "\n",
    "ytest_pred = model.predict(Xtest @ A)\n",
    "Etest = 1 - sklearn.metrics.accuracy_score(ytest, ytest_pred)\n",
    "print(f\"Etest={Etest:0.4f}\")\n",
    "\n",
    "print(f'generalization error={abs(Etest - Ein):0.4f}')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 18,
   "id": "b311faaf-3e3e-405a-bc2b-37da77888d8c",
   "metadata": {},
   "outputs": [],
   "source": [
    "# FIXME:\n",
    "# your final task for this homework is to find the best possible combination of hyperparameters for this problem;\n",
    "# that is, find the best combination of hypothesis class and dprime that minimizes Etest\n",
    "# you don't have to consider hypothesis classes that take a very long amount of time to complete\n",
    "\n",
    "# modify the code block above to contain your hyperparameters\n",
    "# write a short description here about why you chose those hyperparameters and how they connect to the theory"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "8a74a914-467e-4277-9db9-8677fce5b0c5",
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.9.2"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
