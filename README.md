# 基于迭代最近点的点云配准方法性能优化

## 简介

本项目为上海交通大学计算机图形学（CS337）课程大作业，由[王梓涵][https://github.com/wzh99]和[刘权](https://github.com/liuQuan98/)完成。

本小组在现有标准 ICP 和 Go-ICP 代码基础上尝试若干性能优化。

## 配置

### 环境

* Windows 10
* MSVC 14.0 及以上
* CMake 3.16

### 依赖

* PCL 1.9
* VTK 8.2
* Qt 5.12.5
* Eigen 3.3.7
* Boost 1.71
* FLANN 1.7.1

## 代码

所有源文件位于 [src](src) 目录下，后续提及的测试函数均在 [src/main.cpp](src/main.cpp) 中。

### 标准 ICP

标准 ICP 共有两个实现：一个是在 [PCL](https://github.com/PointCloudLibrary/pcl) 基础上简化的单线程版本，见 [`SingleThreadedICP`](src/SingleThreadedICP.hpp)；另一个是使用并行算法的优化版本 [`ICP`](src/ICP.hpp)。两者均使用优化版本的 K-d 树进行最近邻点查找。

测试 `testICP` 中考察了并行算法的性能提升。

### Go-ICP

Go-ICP 实现在[论文作者提供代码](https://github.com/yangjiaolong/Go-ICP)基础上改造完成，见 [`GoICP`](src/GoICP.hpp)。

测试 `testGoICP` 中对 Go-ICP 的配准结果和运行时间进行考察，`testFpcsGoicpImprovement` 对预先使用 4-PCS 进行初始对齐再使用 Go-ICP 配准的设想进行了探索。

### K-d 树

K-d 树共有两个实现：一个是平凡的链接实现，见 [`NaiveKdTree`](src/NaiveKdTree.hpp)；另一个为进行内存优化的版本 [`KdTree`](src/KdTree.hpp)。

测试 `testKdTreeCorrectness` 以蛮力搜索结果为标准，验证了各 K-d 树实现的正确性；`testKdTreeEfficiency` 考察了使用内存优化的性能提升。

### DT

DT 共有三个实现，一个是直接使用 K-d 树进行搜索的平凡版本 [`NaiveDT`](src/NaiveDT.hpp)；一个是 [Go-ICP](https://github.com/yangjiaolong/Go-ICP) 中附带的 [`DT3D`](src/jly_3ddt.h)；第三个是使用线性时间算法的 [`LinearDT`](src/LinearDT.hpp)。

测试 `testdt` 比较了三者构建时间的差异以及精确度差异。

