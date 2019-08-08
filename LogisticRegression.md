**逻辑回归与线性回归的联系与区别**：

​		分类与回归:回归模型就是预测一个连续变量(如降水量，价格等)。在分类问题中，预测属于某类的概率，可以看成回归问题。这可以说是使用回归算法的分类方法。

​		输出:直接使用线性回归的输出作为概率是有问题的，因为其值有可能小于0或者大于1,这是不符合实际情况的，逻辑回归的输出正是[0,1]区间。

​		参数估计方法：https://blog.csdn.net/lx_ros/article/details/81263209

**逻辑回归的原理**：逻辑回归将线性回归的输出通过sigmoid函数转换为类别的概率，即：线性回归+sigmoid函数=逻辑回归。

**正则化与模型评估指标**：

正则化防止过拟合（scikit-learn默认L2正则化）
模型评估指标accuracy

**逻辑回归的优缺点**：
-优点：简单快速
-缺点：无法解决非线性可分的问题

样本不均衡问题的解决办法：重采样/增大较少出现的类别的权重。

sigmod函数：
$$
g(z) = \frac{1}{1+e^-z}
$$
Cost Function:
$$
J(\theta) = -\frac{1}{m}[-y^ilog(h_\theta(x^i)) - (1 - y^i)log(1 -h_\theta(x^i))]+\frac{\lambda}{2m}\sum_{j=1}^n \theta^2_j
$$
λ越小，越容易过拟合，λ越大，越容易欠拟合。

梯度下降：
$$
\frac{∂J(\theta)}{∂\theta_0} = \frac{1}{m}\sum_{i=1}^m(h_\theta(x^i) - y^i)x^i_j   \\
for j = 0
$$

$$
\frac{∂J(\theta)}{∂\theta_j} = (\frac{1}{m}\sum_{i=1}^m(h_\theta(x^i) - y^i)x^i_j) +\frac{\lambda}{m}\theta_j     \\
for   j>=1
$$




**sklearn参数详解**：

class sklearn.linear_model.LogisticRegression(penalty='l2', dual=False, tol=0.0001, C=1.0, fit_intercept=True, intercept_scaling=1, class_weight=None, random_state=None, solver='lbfgs', max_iter=100, multi_class='auto', verbose=0, warm_start=False, n_jobs=None, l1_ratio=None)

penalty:正则化方式（l1/l2/l1+l2）
dual:解原始问题还是对偶问题
tol:控制算法何时结束
C:正则化系数
fit_intercept:是否拟合b
intercept_scaling:LIBLINEAR专用变量
class_weight:类别权重
random_state:随机种子
solver:逻辑回归的解法
max_iter:迭代次数
multi_class:处理多分类问题的方法
verbose:控制输出
warm_start:是否使用之前的输出
n_jobs:控制并行

l1_ratio:正则化系数

