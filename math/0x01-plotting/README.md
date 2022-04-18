
# 0x01. Plotting


![](https://holbertonintranet.s3.amazonaws.com/uploads/medias/2018/9/b4601426ad02130836f9.jpg?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIARDDGGGOU5BHMTQX4%2F20220418%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20220418T144435Z&X-Amz-Expires=86400&X-Amz-SignedHeaders=host&X-Amz-Signature=eadeca9f018e79ca8e63772fa2c9a1202b0de84d57060240d4c213c20603cf79)

## Resources

**Read or watch**:

-   [Plot (graphics)](https://intranet.hbtn.io/rltoken/U-55m7o6No-_W4OJP-oTCg "Plot (graphics)")
-   [Scatter plot](https://intranet.hbtn.io/rltoken/ewQvwktgrnrccqp9PInBpQ "Scatter plot")
-   [Line chart](https://intranet.hbtn.io/rltoken/nUnDxiEeIAMxoV0Vk9dsOg "Line chart")
-   [Bar chart](https://intranet.hbtn.io/rltoken/YZcEmsWNuQcQXYqzfyBfPg "Bar chart")
-   [Histogram](https://intranet.hbtn.io/rltoken/7icFpl6tgO6OvwSvee0S2Q "Histogram")

**References**:

-   [Pyplot tutorial](https://intranet.hbtn.io/rltoken/9GES4KAFhBUOKYj9BI9vgQ "Pyplot tutorial")
-   [matplotlib.pyplot](https://intranet.hbtn.io/rltoken/GaHr4hgXE3LE3skZDGH2pQ "matplotlib.pyplot")
-   [matplotlib.pyplot.plot](https://intranet.hbtn.io/rltoken/IUhQVdCg4MaCdUFEOuaXig "matplotlib.pyplot.plot")
-   [matplotlib.pyplot.scatter](https://intranet.hbtn.io/rltoken/oZ9O1frltXpknQLJGalGPg "matplotlib.pyplot.scatter")
-   [matplotlib.pyplot.bar](https://intranet.hbtn.io/rltoken/gqW7RjVdB5G3WtuzJTcdew "matplotlib.pyplot.bar")
-   [matplotlib.pyplot.hist](https://intranet.hbtn.io/rltoken/K-yG7lADPJCb_FSWyOGerA "matplotlib.pyplot.hist")
-   [matplotlib.pyplot.xlabel](https://intranet.hbtn.io/rltoken/jhcagbtOr5Xq98SmXs8WTQ "matplotlib.pyplot.xlabel")
-   [matplotlib.pyplot.ylabel](https://intranet.hbtn.io/rltoken/jxrkMnJZTqhaRuvfIal5hQ "matplotlib.pyplot.ylabel")
-   [matplotlib.pyplot.title](https://intranet.hbtn.io/rltoken/5yPCtvA_2CSecHenfen8cQ "matplotlib.pyplot.title")
-   [matplotlib.pyplot.subplot](https://intranet.hbtn.io/rltoken/ex_hmQCXTo2gHAbUFfPTyw "matplotlib.pyplot.subplot")
-   [matplotlib.pyplot.subplots](https://intranet.hbtn.io/rltoken/3465mnzNsJp36kpDEd7tCA "matplotlib.pyplot.subplots")
-   [matplotlib.pyplot.subplot2grid](https://intranet.hbtn.io/rltoken/6AIYCbwzqy67xdvhSzj1Aw "matplotlib.pyplot.subplot2grid")
-   [matplotlib.pyplot.suptitle](https://intranet.hbtn.io/rltoken/S5YwnEoLjpTYGDz5VryX6w "matplotlib.pyplot.suptitle")
-   [matplotlib.pyplot.xscale](https://intranet.hbtn.io/rltoken/Gy6aJCznMv4uSNn2LWS6rg "matplotlib.pyplot.xscale")
-   [matplotlib.pyplot.yscale](https://intranet.hbtn.io/rltoken/XmLFrfjIS2WnwnjumbHLrg "matplotlib.pyplot.yscale")
-   [matplotlib.pyplot.xlim](https://intranet.hbtn.io/rltoken/1zKdiptFjaMmbv8iqBVY1Q "matplotlib.pyplot.xlim")
-   [matplotlib.pyplot.ylim](https://intranet.hbtn.io/rltoken/NDvu8opoi1B_uhJjB8SA0g "matplotlib.pyplot.ylim")
-   [mplot3d tutorial](https://intranet.hbtn.io/rltoken/ENFsqb4q1lbSwCEUgTAt0Q "mplot3d tutorial")
-   [additional tutorials](https://intranet.hbtn.io/rltoken/-4sdqeB5ey_3u3htSZZQpw "additional tutorials")

## Learning Objectives

At the end of this project, you are expected to be able to  [explain to anyone](https://intranet.hbtn.io/rltoken/6I2Jz9J1x6X6NndNB-tT8Q "explain to anyone"),  **without the help of Google**:

### General

-   What is a plot?
-   What is a scatter plot? line graph? bar graph? histogram?
-   What is  `matplotlib`?
-   How to plot data with  `matplotlib`
-   How to label a plot
-   How to scale an axis
-   How to plot multiple sets of data at the same time

## Requirements

### General

-   Allowed editors:  `vi`,  `vim`,  `emacs`
-   All your files will be interpreted/compiled on Ubuntu 20.04 LTS using  `python3`  (version 3.8)
-   Your files will be executed with  `numpy`  (version 1.19.2) and  `matplotlib`  (version 3.3.4)
-   All your files should end with a new line
-   The first line of all your files should be exactly  `#!/usr/bin/env python3`
-   A  `README.md`  file, at the root of the folder of the project, is 
-   Your code should use the  `pycodestyle`  style (version 2.6)
-   All your modules should have documentation (`python3 -c 'print(__import__("my_module").__doc__)'`)
-   All your classes should have documentation (`python3 -c 'print(__import__("my_module").MyClass.__doc__)'`)
-   All your functions (inside and outside a class) should have documentation (`python3 -c 'print(__import__("my_module").my_function.__doc__)'`  and  `python3 -c 'print(__import__("my_module").MyClass.my_function.__doc__)'`)
-   Unless otherwise noted, you are not allowed to import any module
-   All your files must be executable
-   The length of your files will be tested using  `wc`

## More Info

### Installing Matplotlib 3.3.4

```
pip install --user matplotlib==3.3.4
pip install --user Pillow
sudo apt-get install python3-tk

```

To check that it has been successfully downloaded, use  `pip list`.

### Configure X11 Forwarding

Update your  `Vagrantfile`  to include the following:

```
Vagrant.configure(2) do |config|
  ...
  config.ssh.forward_x11 = true
end

```

If you are running  `vagrant`  on a Mac, you will have to install  [XQuartz](https://intranet.hbtn.io/rltoken/OVdbL0nPcj2IXiTQoIBwAw "XQuartz")  and restart your computer.

If you are running  `vagrant`  on a Windows computer, you may have to follow  [these instructions](https://intranet.hbtn.io/rltoken/ZGU33rI2v1sPC_WvoXukkg "these instructions").

Once complete, you should simply be able to  `vagrant ssh`  to log into your VM and then any GUI application should forward to your local machine.

_Hint for  `emacs`  users: you will have to use  `emacs -nw`  to prevent it from launching its GUI._

## Tasks

### 0. Line Graph



Complete the following source code to plot  `y`  as a line graph:

-   `y`  should be plotted as a solid red line
-   The x-axis should range from 0 to 10

```
#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt

y = np.arange(0, 11) ** 3

# your code here

```

![](https://holbertonintranet.s3.amazonaws.com/uploads/medias/2018/9/664b2543b48ef4918687.png?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIARDDGGGOU5BHMTQX4%2F20220418%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20220418T144435Z&X-Amz-Expires=86400&X-Amz-SignedHeaders=host&X-Amz-Signature=ad0b988e044d6d2da3ab28e1d594119de5c2fd6e02996c09dfb80147d144d224)



### 1. Scatter



Complete the following source code to plot  `x ↦ y`  as a scatter plot:

-   The x-axis should be labeled  `Height (in)`
-   The y-axis should be labeled  `Weight (lbs)`
-   The title should be  `Men's Height vs Weight`
-   The data should be plotted as magenta points

```
#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt

mean = [69, 0]
cov = [[15, 8], [8, 15]]
np.random.seed(5)
x, y = np.random.multivariate_normal(mean, cov, 2000).T
y += 180

# your code here

```

![](https://holbertonintranet.s3.amazonaws.com/uploads/medias/2018/9/1b143961d254e65afa2c.png?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIARDDGGGOU5BHMTQX4%2F20220418%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20220418T144435Z&X-Amz-Expires=86400&X-Amz-SignedHeaders=host&X-Amz-Signature=edef3674c20f631eb03be8b9e7be342c18f96092ab063281fbbd33822199891b)


### 2. Change of scale



Complete the following source code to plot  `x ↦ y`  as a line graph:

-   The x-axis should be labeled  `Time (years)`
-   The y-axis should be labeled  `Fraction Remaining`
-   The title should be  `Exponential Decay of C-14`
-   The y-axis should be logarithmically scaled
-   The x-axis should range from 0 to 28650

```
#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt

x = np.arange(0, 28651, 5730)
r = np.log(0.5)
t = 5730
y = np.exp((r / t) * x)

# your code here

```

![](https://holbertonintranet.s3.amazonaws.com/uploads/medias/2018/9/2b6334feb069ae1b6014.png?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIARDDGGGOU5BHMTQX4%2F20220418%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20220418T144435Z&X-Amz-Expires=86400&X-Amz-SignedHeaders=host&X-Amz-Signature=9997f909be67e938d84e3b71e9917664fdaa8c86bf90d0f473d9d2065e518a48)


### 3. Two is better than one



Complete the following source code to plot  `x ↦ y1`  and  `x ↦ y2`  as line graphs:

-   The x-axis should be labeled  `Time (years)`
-   The y-axis should be labeled  `Fraction Remaining`
-   The title should be  `Exponential Decay of Radioactive Elements`
-   The x-axis should range from 0 to 20,000
-   The y-axis should range from 0 to 1
-   `x ↦ y1`  should be plotted with a dashed red line
-   `x ↦ y2`  should be plotted with a solid green line
-   A legend labeling  `x ↦ y1`  as  `C-14`  and  `x ↦ y2`  as  `Ra-226`  should be placed in the upper right hand corner of the plot

```
#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt

x = np.arange(0, 21000, 1000)
r = np.log(0.5)
t1 = 5730
t2 = 1600
y1 = np.exp((r / t1) * x)
y2 = np.exp((r / t2) * x)

# your code here

```

![](https://holbertonintranet.s3.amazonaws.com/uploads/medias/2018/9/39eac4e8c8eb71469784.png?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIARDDGGGOU5BHMTQX4%2F20220418%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20220418T144435Z&X-Amz-Expires=86400&X-Amz-SignedHeaders=host&X-Amz-Signature=bf31cfc36a9f804c81605a39f8f7b88443aaf966a1cf061869f9f96efefaa9d3)



### 4. Frequency



Complete the following source code to plot a histogram of student scores for a project:

-   The x-axis should be labeled  `Grades`
-   The y-axis should be labeled  `Number of Students`
-   The x-axis should have bins every 10 units
-   The title should be  `Project A`
-   The bars should be outlined in black

```
#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(5)
student_grades = np.random.normal(68, 15, 50)

# your code here

```

![](https://holbertonintranet.s3.amazonaws.com/uploads/medias/2018/9/10a48ad296d16ee8fb63.png?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIARDDGGGOU5BHMTQX4%2F20220418%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20220418T144435Z&X-Amz-Expires=86400&X-Amz-SignedHeaders=host&X-Amz-Signature=158cb810ffe51a6dbf55ce319f770598b235b1e2af07becec4fdcdd8bd4fd7d5)



### 5. All in One



Complete the following source code to plot all 5 previous graphs in one figure:

-   All axis labels and plot titles should have a font size of  `x-small`  (to fit nicely in one figure)
-   The plots should make a 3 x 2 grid
-   The last plot should take up two column widths (see below)
-   The title of the figure should be  `All in One`

```
#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt

y0 = np.arange(0, 11) ** 3

mean = [69, 0]
cov = [[15, 8], [8, 15]]
np.random.seed(5)
x1, y1 = np.random.multivariate_normal(mean, cov, 2000).T
y1 += 180

x2 = np.arange(0, 28651, 5730)
r2 = np.log(0.5)
t2 = 5730
y2 = np.exp((r2 / t2) * x2)

x3 = np.arange(0, 21000, 1000)
r3 = np.log(0.5)
t31 = 5730
t32 = 1600
y31 = np.exp((r3 / t31) * x3)
y32 = np.exp((r3 / t32) * x3)

np.random.seed(5)
student_grades = np.random.normal(68, 15, 50)

# your code here

```

![](https://holbertonintranet.s3.amazonaws.com/uploads/medias/2018/9/e58d423ffd060a779753.png?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIARDDGGGOU5BHMTQX4%2F20220418%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20220418T144435Z&X-Amz-Expires=86400&X-Amz-SignedHeaders=host&X-Amz-Signature=08eb4f491dbc57b0cc0abb31b95e6c9a8d6df974caf47996360c2620cbf0db91)



### 6. Stacking Bars



Complete the following source code to plot a stacked bar graph:

-   `fruit`  is a matrix representing the number of fruit various people possess
    -   The columns of  `fruit`  represent the number of fruit  `Farrah`,  `Fred`, and  `Felicia`  have, respectively
    -   The rows of  `fruit`  represent the number of  `apples`,  `bananas`,  `oranges`, and  `peaches`, respectively
-   The bars should represent the number of fruit each person possesses:
    -   The bars should be grouped by person, i.e, the horizontal axis should have one labeled tick per person
    -   Each fruit should be represented by a specific color:
        -   `apples`  = red
        -   `bananas`  = yellow
        -   `oranges`  = orange (`#ff8000`)
        -   `peaches`  = peach (`#ffe5b4`)
        -   A legend should be used to indicate which fruit is represented by each color
    -   The bars should be stacked in the same order as the rows of  `fruit`, from bottom to top
    -   The bars should have a width of  `0.5`
-   The y-axis should be labeled  `Quantity of Fruit`
-   The y-axis should range from 0 to 80 with ticks every 10 units
-   The title should be  `Number of Fruit per Person`

```
#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(5)
fruit = np.random.randint(0, 20, (4,3))

# your code here

```

![](https://holbertonintranet.s3.amazonaws.com/uploads/medias/2018/10/8058e8f96e697612d50d.png?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIARDDGGGOU5BHMTQX4%2F20220418%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20220418T144436Z&X-Amz-Expires=86400&X-Amz-SignedHeaders=host&X-Amz-Signature=494dd0e267e55cf88a4b4ddd2175be06fd461f4877027412864aa2b167ddc70b)


### 7. Gradient



Complete the following source code to create a scatter plot of sampled elevations on a mountain:

-   The x-axis should be labeled  `x coordinate (m)`
-   The y-axis should be labeled  `y coordinate (m)`
-   The title should be  `Mountain Elevation`
-   A colorbar should be used to display elevation
-   The colorbar should be labeled  `elevation (m)`

```
#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(5)

x = np.random.randn(2000) * 10
y = np.random.randn(2000) * 10
z = np.random.rand(2000) + 40 - np.sqrt(np.square(x) + np.square(y))

# your code here

```

![](https://holbertonintranet.s3.amazonaws.com/uploads/medias/2018/9/209d635d81bc43ca9ba5.png?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIARDDGGGOU5BHMTQX4%2F20220418%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20220418T144436Z&X-Amz-Expires=86400&X-Amz-SignedHeaders=host&X-Amz-Signature=4ff1c09df474930215c379b6cda4aac8eb6d1608a152c5b766f25b20b95427c9)

### 8. PCA



Principle Component Analysis (PCA) is a vital procedure used in data science for reducing the dimensionality of data (in turn, decreasing computation cost). It is also largely used for visualizing high dimensional data in 2 or 3 dimensions. For this task, you will be visualizing the  [Iris flower data set](https://intranet.hbtn.io/rltoken/XdwrHc6FQIzsyOg8N4nq9A "Iris flower data set ") . You will need to download the file  [pca.npz](https://holbertonintranet.s3.amazonaws.com/uploads/misc/2020/1/cdec57e313874348ba9a.npz?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIARDDGGGOU5BHMTQX4%2F20220418%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20220418T144436Z&X-Amz-Expires=345600&X-Amz-SignedHeaders=host&X-Amz-Signature=1eb0c526e58f830911938748bd7ec158c96c8d68c4d598601572e7d7dad4aedf "pca.npz")  to test your code. You do not need to push this dataset to github. Complete the following source code to visualize the data in 3D:

```
#!/usr/bin/env python3
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np

lib = np.load("pca.npz")
data = lib["data"]
labels = lib["labels"]

data_means = np.mean(data, axis=0)
norm_data = data - data_means
_, _, Vh = np.linalg.svd(norm_data)
pca_data = np.matmul(norm_data, Vh[:3].T)

# your code here

```

-   The title of the plot should be  `PCA of Iris Dataset`
-   `data`  is a  `np.ndarray`  of shape  `(150, 4)`
    -   `150`  => the number of flowers
    -   `4`  => petal length, petal width, sepal length, sepal width
-   `labels`  is a  `np.ndarray`  of shape  `(150,)`  containing information about what species of iris each data point represents:
    -   `0`  => Iris Setosa
    -   `1`  => Iris Versicolor
    -   `2`  => Iris Virginica
-   `pca_data`  is a  `np.ndarray`  of shape  `(150, 3)`
    -   The columns of  `pca_data`  represent the 3 dimensions of the reduced data, i.e., x, y, and z, respectively
-   The x, y, and z axes should be labeled  `U1`,  `U2`, and  `U3`, respectively
-   The data points should be colored based on their labels using the  `plasma`  color map

![](https://holbertonintranet.s3.amazonaws.com/uploads/medias/2019/9/a5834eeaf3eaa42c6530.png?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIARDDGGGOU5BHMTQX4%2F20220418%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20220418T144436Z&X-Amz-Expires=86400&X-Amz-SignedHeaders=host&X-Amz-Signature=10e74354d5a05a37be8e9ce53e8ea160bae9a7026e4e425746af92ef6f280573)
