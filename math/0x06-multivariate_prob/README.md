# 0x06. Multivariate Probability


![](https://holbertonintranet.s3.amazonaws.com/uploads/medias/2019/5/108edd4c06fdede87f5e.png?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIARDDGGGOU5BHMTQX4%2F20220726%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20220726T005347Z&X-Amz-Expires=86400&X-Amz-SignedHeaders=host&X-Amz-Signature=39b13c30e323a8bfbf100d489a671bba2546b76bdb2798174f64589299076ce8)

## Resources

**Read or watch**:

-   [Joint Probability Distributions](https://intranet.hbtn.io/rltoken/4xMIpOV5WyS7452AZmoVVQ "Joint Probability Distributions")
-   [Multivariate Gaussian distributions](https://intranet.hbtn.io/rltoken/QZPnk9D-zitqKRuGQ2FJnA "Multivariate Gaussian distributions")
-   [The Multivariate Gaussian Distribution](https://intranet.hbtn.io/rltoken/SniKXPpORw-Ec2Dx3rHW4Q "The Multivariate Gaussian Distribution")
-   [An Introduction to Variance, Covariance & Correlation](https://intranet.hbtn.io/rltoken/BKfHY5628XVUcvEdbh_tQw "An Introduction to Variance, Covariance & Correlation")
-   [Variance-covariance matrix using matrix notation of factor analysis](https://intranet.hbtn.io/rltoken/acNlYV2Xp5jhRwqvMzcyCA "Variance-covariance matrix using matrix notation of factor analysis")

**Definitions to skim:**

-   [Carl Friedrich Gauss](https://intranet.hbtn.io/rltoken/L7tmwtP3paHm21K-1dW1mA "Carl Friedrich Gauss")
-   [Joint probability distribution](https://intranet.hbtn.io/rltoken/XG4dOn0SR9WMyQfz532KZw "Joint probability distribution")
-   [Covariance](https://intranet.hbtn.io/rltoken/kd41eNEZZoHCgATin5q1Ig "Covariance")
-   [Covariance matrix](https://intranet.hbtn.io/rltoken/VyObnl7THMlKDO3DCMRDlw "Covariance matrix")

**As references**:

-   [numpy.cov](https://intranet.hbtn.io/rltoken/W0lqUXo-qdC-9zfQJ4Ry3Q "numpy.cov")
-   [numpy.corrcoef](https://intranet.hbtn.io/rltoken/KwA7tjXB8o1ZKmyYkjkg_A "numpy.corrcoef")
-   [numpy.linalg.det](https://intranet.hbtn.io/rltoken/Bv5-Jv36lS8QlCHYLU2E6g "numpy.linalg.det")
-   [numpy.linalg.inv](https://intranet.hbtn.io/rltoken/Gc7D378kmocN2kFzsZGRHA "numpy.linalg.inv")
-   [numpy.random.multivariate_normal](https://intranet.hbtn.io/rltoken/gTQQsuOo2XWCH6Nv-q-jzA "numpy.random.multivariate_normal")

## Learning Objectives

At the end of this project, you are expected to be able to  [explain to anyone](https://intranet.hbtn.io/rltoken/5TEljUzzPkF61W05LQywTw "explain to anyone"),  **without the help of Google**:

### General

-   Who is Carl Friedrich Gauss?
-   What is a joint/multivariate distribution?
-   What is a covariance?
-   What is a correlation coefficient?
-   What is a covariance matrix?
-   What is a multivariate Gaussian distribution?

## Requirements

### General

-   Allowed editors:  `vi`,  `vim`,  `emacs`
-   All your files will be interpreted/compiled on Ubuntu 16.04 LTS using  `python3`  (version 3.5)
-   Your files will be executed with  `numpy`  (version 1.15)
-   All your files should end with a new line
-   The first line of all your files should be exactly  `#!/usr/bin/env python3`
-   A  `README.md`  file, at the root of the folder of the project, is mandatory
-   Your code should use the  `pycodestyle`  style (version 2.5)
-   All your modules should have documentation (`python3 -c 'print(__import__("my_module").__doc__)'`)
-   All your classes should have documentation (`python3 -c 'print(__import__("my_module").MyClass.__doc__)'`)
-   All your functions (inside and outside a class) should have documentation (`python3 -c 'print(__import__("my_module").my_function.__doc__)'`  and  `python3 -c 'print(__import__("my_module").MyClass.my_function.__doc__)'`)
-   Unless otherwise noted, you are not allowed to import any module except  `import numpy as np`
-   All your files must be executable
-   The length of your files will be tested using  `wc`

### Quiz questions

**Great!**  You've completed the quiz successfully! Keep going!  (Show quiz)

## Tasks

### 0. Mean and Covariance



Write a function  `def mean_cov(X):`  that calculates the mean and covariance of a data set:

-   `X`  is a  `numpy.ndarray`  of shape  `(n, d)`  containing the data set:
    -   `n`  is the number of data points
    -   `d`  is the number of dimensions in each data point
    -   If  `X`  is not a 2D  `numpy.ndarray`, raise a  `TypeError`  with the message  `X must be a 2D numpy.ndarray`
    -   If  `n`  is less than 2, raise a  `ValueError`  with the message  `X must contain multiple data points`
-   Returns:  `mean`,  `cov`:
    -   `mean`  is a  `numpy.ndarray`  of shape  `(1, d)`  containing the mean of the data set
    -   `cov`  is a  `numpy.ndarray`  of shape  `(d, d)`  containing the covariance matrix of the data set
-   You are not allowed to use the function  `numpy.cov`

```
alexa@ubuntu-xenial:0x06-multivariate_prob$ cat 0-main.py 
#!/usr/bin/env python3

if __name__ == '__main__':
    import numpy as np
    mean_cov = __import__('0-mean_cov').mean_cov

    np.random.seed(0)
    X = np.random.multivariate_normal([12, 30, 10], [[36, -30, 15], [-30, 100, -20], [15, -20, 25]], 10000)
    mean, cov = mean_cov(X)
    print(mean)
    print(cov)
alexa@ubuntu-xenial:0x06-multivariate_prob$ ./0-main.py 
[[12.04341828 29.92870885 10.00515808]]
[[ 36.2007391  -29.79405239  15.37992641]
 [-29.79405239  97.77730626 -20.67970134]
 [ 15.37992641 -20.67970134  24.93956823]]
alexa@ubuntu-xenial:0x06-multivariate_prob$

```


### 1. Correlation



Write a function  `def correlation(C):`  that calculates a correlation matrix:

-   `C`  is a  `numpy.ndarray`  of shape  `(d, d)`  containing a covariance matrix
    -   `d`  is the number of dimensions
    -   If  `C`  is not a  `numpy.ndarray`, raise a  `TypeError`  with the message  `C must be a numpy.ndarray`
    -   If  `C`  does not have shape  `(d, d)`, raise a  `ValueError`  with the message  `C must be a 2D square matrix`
-   Returns a  `numpy.ndarray`  of shape  `(d, d)`  containing the correlation matrix

```
alexa@ubuntu-xenial:0x06-multivariate_prob$ cat 1-main.py 
#!/usr/bin/env python3

if __name__ == '__main__':
    import numpy as np
    correlation = __import__('1-correlation').correlation

    C = np.array([[36, -30, 15], [-30, 100, -20], [15, -20, 25]])
    Co = correlation(C)
    print(C)
    print(Co)
alexa@ubuntu-xenial:0x06-multivariate_prob$ ./1-main.py 
[[ 36 -30  15]
 [-30 100 -20]
 [ 15 -20  25]]
[[ 1.  -0.5  0.5]
 [-0.5  1.  -0.4]
 [ 0.5 -0.4  1. ]]
alexa@ubuntu-xenial:0x06-multivariate_prob$

```



### 2. Initialize



Create the class  `MultiNormal`  that represents a Multivariate Normal distribution:

-   class constructor  `def __init__(self, data):`
    -   `data`  is a  `numpy.ndarray`  of shape  `(d, n)`  containing the data set:
    -   `n`  is the number of data points
    -   `d`  is the number of dimensions in each data point
    -   If  `data`  is not a 2D  `numpy.ndarray`, raise a  `TypeError`  with the message  `data must be a 2D numpy.ndarray`
    -   If  `n`  is less than 2, raise a  `ValueError`  with the message  `data must contain multiple data points`
-   Set the public instance variables:
    -   `mean`  - a  `numpy.ndarray`  of shape  `(d, 1)`  containing the mean of  `data`
    -   `cov`  - a  `numpy.ndarray`  of shape  `(d, d)`  containing the covariance matrix  `data`
-   You are not allowed to use the function  `numpy.cov`

```
alexa@ubuntu-xenial:0x06-multivariate_prob$ cat 2-main.py 
#!/usr/bin/env python3

if __name__ == '__main__':
    import numpy as np
    from multinormal import MultiNormal

    np.random.seed(0)
    data = np.random.multivariate_normal([12, 30, 10], [[36, -30, 15], [-30, 100, -20], [15, -20, 25]], 10000).T
    mn = MultiNormal(data)
    print(mn.mean)
    print(mn.cov)
alexa@ubuntu-xenial:0x06-multivariate_prob$ ./2-main.py 
[[12.04341828]
 [29.92870885]
 [10.00515808]]
[[ 36.2007391  -29.79405239  15.37992641]
 [-29.79405239  97.77730626 -20.67970134]
 [ 15.37992641 -20.67970134  24.93956823]]
alexa@ubuntu-xenial:0x06-multivariate_prob$

```



### 3. PDF



Update the class  `MultiNormal`:

-   public instance method  `def pdf(self, x):`  that calculates the PDF at a data point:
    -   `x`  is a  `numpy.ndarray`  of shape  `(d, 1)`  containing the data point whose PDF should be calculated
        -   `d`  is the number of dimensions of the  `MultiNormal`  instance
    -   If  `x`  is not a  `numpy.ndarray`, raise a  `TypeError`  with the message  `x must be a numpy.ndarray`
    -   If  `x`  is not of shape  `(d, 1)`, raise a  `ValueError`  with the message  `x must have the shape ({d}, 1)`
    -   Returns the value of the PDF
    -   You are not allowed to use the function  `numpy.cov`

```
alexa@ubuntu-xenial:0x06-multivariate_prob$ cat 3-main.py 
#!/usr/bin/env python3

if __name__ == '__main__':
    import numpy as np
    from multinormal import MultiNormal

    np.random.seed(0)
    data = np.random.multivariate_normal([12, 30, 10], [[36, -30, 15], [-30, 100, -20], [15, -20, 25]], 10000).T
    mn = MultiNormal(data)
    x = np.random.multivariate_normal([12, 30, 10], [[36, -30, 15], [-30, 100, -20], [15, -20, 25]], 1).T
    print(x)
    print(mn.pdf(x))
alexa@ubuntu-xenial:0x06-multivariate_prob$ ./3-main.py 
[[ 8.20311936]
 [32.84231319]
 [ 9.67254478]]
0.00022930236202143824
alexa@ubuntu-xenial:0x06-multivariate_prob$ 

```

