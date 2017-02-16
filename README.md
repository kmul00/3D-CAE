#This is a simple 3D-CAE for Frame Predicition

**Dataset :** *Bouncing MNIST*

**Input** and **Output** of the format *(channels X frames X height X width)*

**Dimension :** (1 X 5 X 64 X 64)

In the experiments shown below
<br>
<br>

**Input :** ![input][input]

**Optimizer :** *ADAM*

**Loss function :** *Absolute Criterion*

**Model :** In [models](model.lua) file

<br>
##Experiments :

<br>
### 1. Predict same frames
---
**Ground-Truth :** ![output][output_1]

<br>
**Ouput of three epochs :**

![output][same_1] ![output][same_2] ![output][same_3]

<br>
### 2. Predict same frames in reverse order
---
**Ground-Truth :** ![output][output_2]

<br>
**Ouput of three epochs :**

![output][same_reverse_1] ![output][same_reverse_2]  ![output][same_reverse_3]

<br>
### 3. Predict next frames
---
**Ground-Truth :** ![output][output_3]

<br>
**Ouput of fifteen epochs :**

![output][next_1] ![output][next_2]  ![output][next_3] ![output][next_4] ![output][next_5]  ![output][next_6] ![output][next_7]  ![output][next_8] ![output][next_9] ![output][next_10]  ![output][next_11] ![output][next_12] ![output][next_13]  ![output][next_14] ![output][next_15]

<br>
### 4. Predict next frames in reverse order
---
**Ground-Truth :** ![output][output_4]

<br>
**Ouput of fifteen epochs :**

![output][next_reverse_1] ![output][next_reverse_2]  ![output][next_reverse_3] ![output][next_reverse_4] ![output][next_reverse_5]  ![output][next_reverse_6] ![output][next_reverse_7]  ![output][next_reverse_8] ![output][next_reverse_9] ![output][next_reverse_10]  ![output][next_reverse_11] ![output][next_reverse_12] ![output][next_reverse_13]  ![output][next_reverse_14] ![output][next_reverse_15]

<br>
##Analysis :

<br>

* When trying to predict the *same frames* as input, the model learns very easily within first one-two epochs only (both in *actual* and *reverse* orders).

* But while trying to predict the *next frames*, the model somehow learns that the *sixth* frames is most inter-related to the input frames and tries to learn it first, followed by *seventh*, *eighth*, e.t.c. No matter whether we present the frames in *actual* or *reverse* order.

<br>
##To-do :

<br>

Your suggestions :

* Try out *frame dropping* while back-propagating the gradient, when one frame gets better than the others.

* Have *separate loss functions* for each frame. Though this will be harder to generalize in case we want to predict any `n` number of output frames, not just 5 (in this case).

[input]:images/input.png
[output_1]:images/input.png
[output_2]:images/same_reverse.png
[output_3]:images/next.png
[output_4]:images/next_reverse.png
[same_1]:images/same_1.png
[same_2]:images/same_2.png
[same_3]:images/same_3.png
[same_reverse_1]:images/same_reverse_1.png
[same_reverse_2]:images/same_reverse_2.png
[same_reverse_3]:images/same_reverse_3.png
[next_1]:images/next_1.png
[next_2]:images/next_2.png
[next_3]:images/next_3.png
[next_4]:images/next_4.png
[next_5]:images/next_5.png
[next_6]:images/next_6.png
[next_7]:images/next_7.png
[next_8]:images/next_8.png
[next_9]:images/next_9.png
[next_10]:images/next_10.png
[next_11]:images/next_11.png
[next_12]:images/next_12.png
[next_13]:images/next_13.png
[next_14]:images/next_14.png
[next_15]:images/next_15.png
[next_reverse_1]:images/next_reverse_1.png
[next_reverse_2]:images/next_reverse_2.png
[next_reverse_3]:images/next_reverse_3.png
[next_reverse_4]:images/next_reverse_4.png
[next_reverse_5]:images/next_reverse_5.png
[next_reverse_6]:images/next_reverse_6.png
[next_reverse_7]:images/next_reverse_7.png
[next_reverse_8]:images/next_reverse_8.png
[next_reverse_9]:images/next_reverse_9.png
[next_reverse_10]:images/next_reverse_10.png
[next_reverse_11]:images/next_reverse_11.png
[next_reverse_12]:images/next_reverse_12.png
[next_reverse_13]:images/next_reverse_13.png
[next_reverse_14]:images/next_reverse_14.png
[next_reverse_15]:images/next_reverse_15.png
