
-   [](https://students-support.hbtn.io/hc)
    
      
    

----------

----------

Curriculum

SpecializationsAverage:  83.43%

# 0x00. Dimensionality Reduction

-   By Alexa Orrico, Software Engineer at Holberton School
-   Weight: 2
-   Ongoing project - started
    
    Jul 26, 2022
    
    , must end by
    
    Jul 29, 2022
    
    - you're done with  0% of tasks.
-   Checker was released at
    
    Jul 27, 2022 12:00 PM
    
-   **Manual QA review must be done**  (request it when you are done with the project)
-   An auto review will be launched at the deadline

![](https://holbertonintranet.s3.amazonaws.com/uploads/medias/2018/11/77f77fafb61bd7249233.jpg?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIARDDGGGOU5BHMTQX4%2F20220728%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20220728T220957Z&X-Amz-Expires=86400&X-Amz-SignedHeaders=host&X-Amz-Signature=00306c10a3b1c446772ea2ff2eac16a5fab790b64d8911b2b4ef849361b78c61)

## Resources

**Read or watch**:

-   [Dimensionality Reduction For Dummies — Part 1: Intuition](https://intranet.hbtn.io/rltoken/rC9v-NVuobViH7oYHpAjRQ "Dimensionality Reduction For Dummies — Part 1: Intuition")
-   [Singular Value Decomposition](https://intranet.hbtn.io/rltoken/Qg_s08ni0zOWkqvvRM8ZwQ "Singular Value Decomposition")
-   [Understanding SVD (Singular Value Decomposition)](https://intranet.hbtn.io/rltoken/S5_1WIYhOFl2lEkdO5RFMg "Understanding SVD (Singular Value Decomposition)")
-   [Intuitively, what is the difference between Eigendecomposition and Singular Value Decomposition?](https://intranet.hbtn.io/rltoken/WyHO8ZBDqbKmzUoD0Ukf7Q "Intuitively, what is the difference between Eigendecomposition and Singular Value Decomposition?")
-   [Dimensionality Reduction: Principal Components Analysis, Part 1](https://intranet.hbtn.io/rltoken/euVIN9M2jJ-PHyOEBnI1lA "Dimensionality Reduction: Principal Components Analysis, Part 1")
-   [Dimensionality Reduction: Principal Components Analysis, Part 2](https://intranet.hbtn.io/rltoken/co3YVWGBIdcto2q3HPu51A "Dimensionality Reduction: Principal Components Analysis, Part 2")
-   [StatQuest: t-SNE, Clearly Explained](https://intranet.hbtn.io/rltoken/XGKIL0TBES-GY6gO6VoSmg "StatQuest: t-SNE, Clearly Explained")
-   [t-SNE tutorial Part1](https://intranet.hbtn.io/rltoken/IaO5r9ba0T_flqHcQv83fA "t-SNE tutorial Part1")
-   [t-SNE tutorial Part2](https://intranet.hbtn.io/rltoken/hariVnyW46RIjyXj6DefGA "t-SNE tutorial Part2")
-   [How to Use t-SNE Effectively](https://intranet.hbtn.io/rltoken/ZGyuMFuDwY6SzE-pM3ZrTw "How to Use t-SNE Effectively")

**Definitions to skim:**

-   [Dimensionality Reduction](https://intranet.hbtn.io/rltoken/3__-0sq0ymVc6rUhSUF46Q "Dimensionality Reduction")
-   [Principal component analysis](https://intranet.hbtn.io/rltoken/-Q1NQBRaQiPLZAlpnXDQoQ "Principal component analysis")
-   [Eigendecomposition of a matrix](https://intranet.hbtn.io/rltoken/ZicQZ9TndU2Khb4QLnU9Rg "Eigendecomposition of a matrix")
-   [Singular value decomposition](https://intranet.hbtn.io/rltoken/pW3EQwurOaQp4f9SIFXs0w "Singular value decomposition")
-   [Manifold](https://intranet.hbtn.io/rltoken/W_DWK5vN6rSRqN6jaVe7Ag "Manifold")  _check this out if you have never heard this term before_
-   [Kullback–Leibler divergence](https://intranet.hbtn.io/rltoken/EAzyLBFVORoaaWgWc8K9yQ "Kullback–Leibler divergence")
-   [T-distributed stochastic neighbor embedding](https://intranet.hbtn.io/rltoken/EnCpSMJZOJ2E7IMdOof0Jg "T-distributed stochastic neighbor embedding")

**As references**:

-   [numpy.cumsum](https://intranet.hbtn.io/rltoken/TUz_LerlFe9fPhMuHxJXLg "numpy.cumsum")
-   [Visualizing Data using t-SNE](https://intranet.hbtn.io/rltoken/2l3jXLWneQVGdNfoXsWMQQ "Visualizing Data using t-SNE")  (paper)
-   [Visualizing Data Using t-SNE](https://intranet.hbtn.io/rltoken/mgNNPvYr_iahfCU8hEZsHQ "Visualizing Data Using t-SNE")  (video)

**Advanced**:

-   [Kernel principal component analysis](https://intranet.hbtn.io/rltoken/61bPYClgo7vCg7FHEzSVdQ "Kernel principal component analysis")
-   [Nonlinear Dimensionality Reduction: KPCA](https://intranet.hbtn.io/rltoken/34dL3ML5vCExK-iUR9_0Rg "Nonlinear Dimensionality Reduction: KPCA")

## Learning Objectives

-   What is eigendecomposition?
-   What is singular value decomposition?
-   What is the difference between eig and svd?
-   What is dimensionality reduction and what are its purposes?
-   What is principal components analysis (PCA)?
-   What is t-distributed stochastic neighbor embedding (t-SNE)?
-   What is a manifold?
-   What is the difference between linear and non-linear dimensionality reduction?
-   Which techniques are linear/non-linear?

## Requirements

### General

-   Allowed editors:  `vi`,  `vim`,  `emacs`
-   All your files will be interpreted/compiled on Ubuntu 16.04 LTS using  `python3`  (version 3.5)
-   Your files will be executed with  `numpy`  (version 1.15)
-   All your files should end with a new line
-   The first line of all your files should be exactly  `#!/usr/bin/env python3`
-   A  `README.md`  file, at the root of the folder of the project, is mandatory
-   Your code should use the  `pycodestyle`  style (version 2.4)
-   All your modules should have documentation (`python3 -c 'print(__import__("my_module").__doc__)'`)
-   All your classes should have documentation (`python3 -c 'print(__import__("my_module").MyClass.__doc__)'`)
-   All your functions (inside and outside a class) should have documentation (`python3 -c 'print(__import__("my_module").my_function.__doc__)'`  and  `python3 -c 'print(__import__("my_module").MyClass.my_function.__doc__)'`)
-   Unless otherwise noted, you are not allowed to import any module except  `import numpy as np`
-   All your files must be executable
-   **Your code should use the minimum number of operations to avoid floating point errors**

## Data

Please test your main files with the following data:

-   [mnist2500_X.txt](https://intranet-projects-files.s3.amazonaws.com/holbertonschool-ml/mnist2500_X.txt "mnist2500_X.txt")
-   [mnist2500_labels.txt](https://holbertonintranet.s3.amazonaws.com/uploads/text/2019/10/72a86270e2a1c2cbc14b.txt?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIARDDGGGOU5BHMTQX4%2F20220728%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20220728T220957Z&X-Amz-Expires=345600&X-Amz-SignedHeaders=host&X-Amz-Signature=37769c4eea61d3d031c38bde18fa835e86786550be083377b09957bd5ff820d0 "mnist2500_labels.txt")

## Watch Out!

Just like lists,  `np.ndarray`s are mutable objects:

```
>>> vector = np.ones((100, 1))
>>> m1 = vector[55]
>>> m2 = vector[55, 0]
>>> vector[55] = 2
>>> m1
array([2.])
>>> m2
1.0

```

## Performance between SVD and EIG

Here a graph of execution time (Y-axis) for the number of iteration (X-axis) - red line is EIG and blue line is SVG

![](https://holbertonintranet.s3.amazonaws.com/uploads/medias/2020/10/df2eac7a51b56139b4a179a83398b18fbda8868c.png?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIARDDGGGOU5BHMTQX4%2F20220728%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20220728T220957Z&X-Amz-Expires=86400&X-Amz-SignedHeaders=host&X-Amz-Signature=b32068bbb7b03b6c5328042d623307380635c0ab0a752b8372fdb9304aa40f8e)

## Tasks

### 0. PCA



Write a function  `def pca(X, var=0.95):`  that performs PCA on a dataset:

-   `X`  is a  `numpy.ndarray`  of shape  `(n, d)`  where:
    -   `n`  is the number of data points
    -   `d`  is the number of dimensions in each point
    -   all dimensions have a mean of 0 across all data points
-   `var`  is the fraction of the variance that the PCA transformation should maintain
-   Returns: the weights matrix,  `W`, that maintains  `var`  fraction of  `X`‘s original variance
-   `W`  is a  `numpy.ndarray`  of shape  `(d, nd)`  where  `nd`  is the new dimensionality of the transformed  `X`

```
alexa@ubuntu-xenial:0x00-dimensionality_reduction$ cat 0-main.py 
#!/usr/bin/env python3

import numpy as np
pca = __import__('0-pca').pca

np.random.seed(0)
a = np.random.normal(size=50)
b = np.random.normal(size=50)
c = np.random.normal(size=50)
d = 2 * a
e = -5 * b
f = 10 * c

X = np.array([a, b, c, d, e, f]).T
m = X.shape[0]
X_m = X - np.mean(X, axis=0)
W = pca(X_m)
T = np.matmul(X_m, W)
print(T)
X_t = np.matmul(T, W.T)
print(np.sum(np.square(X_m - X_t)) / m)
alexa@ubuntu-xenial:0x00-dimensionality_reduction$ ./0-main.py 
[[-16.71379391   3.25277063  -3.21956297]
 [ 16.22654311  -0.7283969   -0.88325252]
 [ 15.05945199   3.81948929  -1.97153621]
 [ -7.69814111   5.49561088  -4.34581561]
 [ 14.25075197   1.37060228  -4.04817187]
 [-16.66888233  -3.77067823   2.6264981 ]
 [  6.71765183   0.18115089  -1.91719288]
 [ 10.20004065  -0.84380128   0.44754302]
 [-16.93427229   1.72241573   0.9006236 ]
 [-12.4100987    0.75431367  -0.36518129]
 [-16.40464248   1.98431953   0.34907508]
 [ -6.69439671   1.30624703  -2.77438892]
 [ 10.84363895   4.99826372  -1.36502623]
 [-17.2656016    7.29822621   0.63226953]
 [  5.32413372  -0.54822516  -0.79075935]
 [ -5.63240657   1.50278876  -0.27590797]
 [ -7.63440366   7.72788006  -2.58344477]
 [  4.3348786   -2.14969035   0.61262033]
 [ -3.95417052   4.22254889  -0.14601319]
 [ -6.59947069  -1.00867621   2.29551761]
 [ -0.78942283  -4.15454151   5.87117533]
 [ 13.62292856   0.40038586  -1.36043631]
 [  0.03536684  -5.85950737  -1.86196569]
 [-11.1841298    5.20313078   2.37753549]
 [  9.62095425  -1.17179699  -4.97535412]
 [  3.85296648   3.55808      3.65166717]
 [  6.57934417   4.87503426   0.30243418]
 [-16.17025935   1.49358788   1.0663259 ]
 [ -4.33639793   1.26186205  -2.99149191]
 [ -1.52947063  -0.39342225  -2.96475006]
 [  9.80619496   6.65483286   0.07714817]
 [ -2.45893463  -4.89091813  -0.6918453 ]
 [  9.56282904  -1.8002211    2.06720323]
 [  1.70293073   7.68378254   5.03581954]
 [  9.58030378  -6.97453776   0.64558546]
 [ -3.41279182 -10.07660784  -0.39277019]
 [ -2.74983634  -6.25461193  -2.65038235]
 [  4.54987003   1.28692201  -2.40001675]
 [ -1.81149682   5.16735962   1.4245976 ]
 [ 13.97823555  -4.39187437   0.57600155]
 [ 17.39107161   3.26808567   2.50429006]
 [ -1.25835112  -6.60720376   3.24220508]
 [  1.06405562  -1.25980089   4.06401644]
 [ -3.44578711  -5.21002054  -4.20836152]
 [-21.1181523   -3.72353504   1.6564066 ]
 [ -6.56723647  -4.31268383   1.22783639]
 [ 11.77670231   0.67338386   2.94885044]
 [ -7.89417224  -9.82300322  -1.69743681]
 [ 15.87543091   0.3804009    3.67627751]
 [  7.38044431  -1.58972122   0.60154138]]
1.7550484837045842e-29
alexa@ubuntu-xenial:0x00-dimensionality_reduction$

```


### 1. PCA v2



Write a function  `def pca(X, ndim):`  that performs PCA on a dataset:

-   `X`  is a  `numpy.ndarray`  of shape  `(n, d)`  where:
    -   `n`  is the number of data points
    -   `d`  is the number of dimensions in each point
-   `ndim`  is the new dimensionality of the transformed  `X`
-   Returns:  `T`, a  `numpy.ndarray`  of shape  `(n, ndim)`  containing the transformed version of  `X`

```
alexa@ubuntu-xenial:0x00-dimensionality_reduction$ cat 1-main.py 
#!/usr/bin/env python3

import numpy as np
pca = __import__('1-pca').pca

X = np.loadtxt("mnist2500_X.txt")
print('X:', X.shape)
print(X)
T = pca(X, 50)
print('T:', T.shape)
print(T)
alexa@ubuntu-xenial:0x00-dimensionality_reduction$ ./1-main.py 
X: (2500, 784)
[[1. 1. 1. ... 1. 1. 1.]
 [1. 1. 1. ... 1. 1. 1.]
 [1. 1. 1. ... 1. 1. 1.]
 ...
 [1. 1. 1. ... 1. 1. 1.]
 [1. 1. 1. ... 1. 1. 1.]
 [1. 1. 1. ... 1. 1. 1.]]
T: (2500, 50)
[[-0.61344587  1.37452188 -1.41781926 ... -0.42685217  0.02276617
   0.1076424 ]
 [-5.00379081  1.94540396  1.49147124 ...  0.26249077 -0.4134049
  -1.15489853]
 [-0.31463237 -2.11658407  0.36608266 ... -0.71665401 -0.18946283
   0.32878802]
 ...
 [ 3.52302175  4.1962009  -0.52129062 ... -0.24412645  0.02189273
   0.19223197]
 [-0.81387035 -2.43970416  0.33244717 ... -0.55367626 -0.64632309
   0.42547833]
 [-2.25717018  3.67177791  2.83905021 ... -0.35014766 -0.01807652
   0.31548087]]
alexa@ubuntu-xenial:0x00-dimensionality_reduction$

```


