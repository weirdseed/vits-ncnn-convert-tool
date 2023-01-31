# 介绍

这是一个将vits的pytorch权重转换为ncnn支持的格式的脚本，项目基于[pnnx](https://github.com/pnnx/pnnx)和[vits](https://github.com/jaywalnut310/vits)开发，配合此项目使用 https://github.com/weirdseed/Vits-Android-ncnn

# 使用说明

## Windows
命令行输入：
```
convert_model.exe -c \path\to\config.json -m \path\to\model.pth -fp16
```
## Linux
命令行输入
```
./convert_model -c /path/to/config.json -m /path/to/model.pth -fp16
```
## python用户
1.创建虚拟环境
```
conda create -n vits-ncnn python=3.10
```
2.安装依赖
```
pip install -r requirements.txt
```
3.启动脚本
```
python convert_model.py -c /path/to/config.json -m /path/to/model.pth -fp16
```
# 参考
https://github.com/jaywalnut310/vits

https://github.com/pnnx/pnnx

https://github.com/Tencent/ncnn

https://github.com/CjangCjengh/MoeGoe
