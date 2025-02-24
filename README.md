# 0ITFuzz
## 说明
在AFL++ 4.30c基础上魔改，做到了对元数据的定向模糊测试。替换原文件编译即可。

## 使用方法：
针对性模糊测试 bin文件下存放提取到的元数据信息（文件名必须是bin）
~~~
afl-fuzz -i in/ -o out/ -k bin/ -- xxx 
~~~
变异前，获取每个文件对应的bitmap图(不是bin即可)
~~~
afl-fuzz -i in/ -o out/ -k t -- xxx
~~~
