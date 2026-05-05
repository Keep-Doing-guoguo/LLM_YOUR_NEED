## 1.装饰器的作用。

它可以在 不修改原函数代码 的情况下，增加额外功能。123 

```python   
def my_decorator(func):
    def wrapper(*args, **kwargs):
        print("👉 调用前")
        result = func(*args, **kwargs)
        print("👈 调用后")
        return result
    return wrapper

@my_decorator
def say_hello(name):
    print(f"Hello, {name}!")

say_hello("zgw")

#等价于下面
say_hello = my_decorator(say_hello)
```

## 2.seesion和request和cookie的区别

Request 是一次请求，Session 是一系列连续请求的会话，Cookie 是浏览器用来保存会话信息的数据。

## 3.什么是闭包
闭包是一个函数及其相关的引用环境组合而成的实体。它允许函数访问其外部作用域中的变量，即使外部函数已经返回。

```python
def outer_function(x):
    def inner_function(y):
        return x + y
    return inner_function
closure = outer_function(10)
print(closure(5))  # 输出 15
```
## 4.什么是迭代器和生成器
迭代器是一个对象，它实现了迭代协议，包含 `__iter__()` 和 `__next__()` 方法。生成器是使用 `yield` 关键字定义的函数，返回一个迭代器。

```python
# 迭代器示例
class MyIterator:
    def __init__(self, limit):
        self.limit = limit
        self.current = 0

    def __iter__(self):
        return self

    def __next__(self):
        if self.current < self.limit:
            self.current += 1
            return self.current - 1
        else:
            raise StopIteration
it = MyIterator(3) 
for num in it:
    print(num)  # 输出 0, 1, 2
```

```python
# 生成器示例
def my_generator(limit):
    for i in range(limit):
        yield i
gen = my_generator(3)
for num in gen:
    print(num)  # 输出 0, 1, 2
```

## 5.什么是多线程和多进程
多线程是指在同一进程内同时运行多个线程，共享进程的资源。多进程是指同时运行多个独立的进程，每个进程有自己的内存空间。

```python
# 多线程示例
import threading
def thread_function(name):
    print(f"Thread {name} starting")
    print(f"Thread {name} finishing")
threads = []
for index in range(3):
    x = threading.Thread(target=thread_function, args=(index,))
    threads.append(x)
    x.start()
for t in threads:
    t.join()
```

```python
# 多进程示例
import multiprocessing
def process_function(name):
    print(f"Process {name} starting")
    print(f"Process {name} finishing")
processes = []
for index in range(3):
    p = multiprocessing.Process(target=process_function, args=(index,))
    processes.append(p)
    p.start()
for p in processes:
    p.join()
```
## 6.什么是GIL

GIL（Global Interpreter Lock，全局解释器锁）是Python解释器中的一个机制，它确保同一时间只有一个线程在执行Python字节码。GIL的存在使得多线程在CPU密集型任务中无法充分利用多核处理器的优势，但它简化了内存管理，避免了多线程编程中的许多复杂问题。

## 7.什么是垃圾回收机制

Python的垃圾回收机制主要通过引用计数和循环引用检测来管理内存。每个对象都有一个引用计数，当引用计数为零时，对象会被立即销毁。对于循环引用，Python使用标记-清除算法来检测并回收这些对象。

## 8.什么是面向对象编程

面向对象编程（OOP）是一种编程范式，它将数据和操作数据的函数封装在对象中。OOP的核心概念包括类、对象、继承、多态和封装。


## 9.fastapi和flask的区别
FastAPI 是一个现代的、快速（高性能）的 web 框架，基于标准的 Python 类型提示，适合构建 APIs。Flask 是一个轻量级的 web 框架，灵活且易于扩展，适合构建各种 web 应用。
## 10.什么是ORM
ORM（Object-Relational Mapping，对象关系映射）是一种技术，它允许开发者使用面向对象的编程语言来操作数据库，而无需直接编写 SQL 语句。ORM 将数据库表映射为类，将表中的记录映射为类的实例，从而简化了数据库操作。

## 11.python中的with语句

`with` 语句用于简化资源管理（如文件操作、网络连接等），确保资源在使用后正确释放。它依赖于上下文管理器协议，包含 `__enter__()` 和 `__exit__()` 方法。

## 12.asycio
`asyncio` 是 Python 的一个库，用于编写异步 I/O 操作的代码。它提供了事件循环、协程和任务等机制，使得在单线程中处理大量 I/O 操作变得高效。

```python
import asyncio
async def main():
    print("Hello")
    await asyncio.sleep(1)
    print("World")
asyncio.run(main())

# Hello
# （等待 1 秒）
# World

#
```
async 定义异步函数，await 暂停执行等待协程完成，asyncio.run() 负责启动整个异步事件循环。

## 13.python中的深拷贝和浅拷贝

浅拷贝创建一个新的对象，但不复制嵌套对象的引用。深拷贝创建一个新的对象，并递归地复制所有嵌套对象。

## 14.fastapi的依赖注入
FastAPI 的依赖注入系统允许你定义可重用的组件（如数据库连接、认证等），并将它们作为参数传递给路径操作函数。FastAPI 会自动处理这些依赖项的创建和销毁。

```python
from fastapi import FastAPI, Depends
app = FastAPI()
def get_db():
    db = "数据库连接"
    try:
        yield db
    finally:
        print("关闭数据库连接")
@app.get("/items/")
def read_items(db: str = Depends(get_db)):
    return {"db": db}
# 当访问 /items/ 时，FastAPI 会调用 get_db()，并将返回的 db 传递给 read_items()。
# 访问 /items/ 会输出 {"db": "数据库连接"}，并在请求结束后打印 "关闭数据库连接"。
```
## 15.python的魔法函数
魔法函数（也称为特殊方法）是以双下划线开头和结尾的方法，用于实现类的特定行为，如初始化、字符串表示、算术运算等。

```python
class MyClass:
    def __init__(self, value):
        self.value = value
    def __str__(self):
        return f"MyClass with value: {self.value}"
    def __add__(self, other):
        return MyClass(self.value + other.value)
obj1 = MyClass(10)
obj2 = MyClass(20)
print(obj1)  # 输出 MyClass with value: 10
obj3 = obj1 + obj2
print(obj3)  # 输出 MyClass with value: 30



class MyList:
    def __init__(self, data):
        self.data = data

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)

a = MyList([1, 2, 3])
print(len(a))   # 3
print(a[1])     # 2


class Vector:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __add__(self, other):
        return Vector(self.x + other.x, self.y + other.y)

    def __str__(self):
        return f"Vector({self.x}, {self.y})"

v1 = Vector(1, 2)
v2 = Vector(3, 4)
print(v1 + v2)   # Vector(4, 6)


class Dog:
    def __init__(self, name):
        self.name = name

    def __str__(self):
        return f"🐶 {self.name}"

dog = Dog("旺财")
print(dog)   # 自动调用 __str__ → 🐶 旺财
```


## 16.__str__ __repr__:
`__str__` 用于定义对象的字符串表示，适合用户阅读；`__repr__` 用于定义对象的正式字符串表示，适合开发者调试。

p = Obj('5')
print(p)#没有str，使用repr。
print(repr(p))


## 17.python的pop
`pop()` 方法用于从列表或字典中移除并返回指定位置的元素。对于列表，默认移除最后一个元素；对于字典，移除指定键的键值对。

dict.pop(key[, default])。如果该值不存在，那么返回指定的default值，如果没有指定default值，那么抛出KeyError异常。

## 17.
asyncio.run()
uvicorn.run()
作用
启动异步任务
启动 Web 服务器


## 18.jinjia2和f-string的区别

jinja2 支持模板逻辑（如 if/for 语句），适合复杂动态内容；f-string 只做简单变量替换，适合固定结构的字符串格式化。



## 19.线程池的实现：
```python

from concurrent.futures import ThreadPoolExecutor, as_completed
import time

def task(n):
    print(f"开始任务 {n}")
    time.sleep(1)
    return f"任务 {n} 完成"

# 创建线程池，最多3个线程
with ThreadPoolExecutor(max_workers=3) as executor:
    futures = [executor.submit(task, i) for i in range(5)]

    for future in as_completed(futures):
        print(future.result())
```
线程的自动管理复用，所有任务自动关闭。


## 20.什么是单例模式

单例模式（Singleton Pattern）是一种设计模式，它保证一个类在整个程序运行期间只创建一个实例，并提供一个全局访问点。

简单理解：

```text
无论你创建多少次对象，拿到的都是同一个对象。
```

常见使用场景：

| 场景 | 为什么适合单例 |
|------|----------------|
| 配置管理器 | 全局配置只需要一份 |
| 日志对象 | 日志写入器通常全局复用 |
| 数据库连接池 | 避免重复创建连接池 |
| 缓存对象 | 全局缓存统一管理 |
| 模型加载器 | 大模型或机器学习模型加载成本高 |


### 20.1 普通类每次都会创建新对象

```python
class Config:
    pass

a = Config()
b = Config()

print(a is b)  # False
print(id(a))
print(id(b))
```

`a is b` 为 `False`，说明它们是两个不同对象。


### 20.2 使用 `__new__` 实现单例

`__new__` 是真正创建对象的方法，`__init__` 是对象创建后做初始化的方法。

所以想控制“只创建一个对象”，应该重写 `__new__`。

```python
class Singleton:
    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance


a = Singleton()
b = Singleton()

print(a is b)  # True
print(id(a))
print(id(b))
```

输出结果中，`a` 和 `b` 的 `id` 相同，说明它们是同一个对象。


### 20.3 `__init__` 会被重复执行的问题

注意：上面的单例虽然只创建一个对象，但每次调用类时，`__init__` 仍然会执行。

```python
class Config:
    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self, name):
        print("执行 __init__")
        self.name = name


a = Config("第一次")
b = Config("第二次")

print(a is b)      # True
print(a.name)      # 第二次
print(b.name)      # 第二次
```

虽然 `a` 和 `b` 是同一个对象，但第二次初始化会覆盖第一次的属性。


### 20.4 防止 `__init__` 重复初始化

可以增加一个 `_initialized` 标记。

```python
class Config:
    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self, name):
        if getattr(self, "_initialized", False):
            return

        self.name = name
        self._initialized = True
        print("只初始化一次")


a = Config("第一次")
b = Config("第二次")

print(a is b)  # True
print(a.name)  # 第一次
print(b.name)  # 第一次
```

这样对象只会初始化一次。


### 20.5 装饰器实现单例

也可以用装饰器保存类和实例的映射关系。

```python
def singleton(cls):
    instances = {}

    def wrapper(*args, **kwargs):
        if cls not in instances:
            instances[cls] = cls(*args, **kwargs)
        return instances[cls]

    return wrapper


@singleton
class Logger:
    def __init__(self):
        print("创建 Logger")


a = Logger()
b = Logger()

print(a is b)  # True
```

这种写法简单，但被装饰后的 `Logger` 本质上变成了 `wrapper` 函数，不再是原始类对象。简单业务可以用，复杂场景更推荐 `__new__` 或元类方式。


### 20.6 模块天然就是单例

Python 的模块只会被导入并初始化一次，所以模块本身也常被当作单例使用。

例如：

```python
# config.py
class Config:
    def __init__(self):
        self.debug = True

config = Config()
```

其他地方直接导入：

```python
from config import config

print(config.debug)
```

只要导入的是同一个模块，拿到的就是同一个 `config` 对象。

这是 Python 中最简单、最自然的单例方式。


### 20.7 线程安全单例

如果多个线程同时创建对象，普通单例可能出现并发问题。可以使用锁。

```python
import threading


class SafeSingleton:
    _instance = None
    _lock = threading.Lock()

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance
```

这里使用了双重检查：

```text
第一次检查：避免每次都加锁
第二次检查：防止多个线程同时进入创建逻辑
```


### 20.8 单例模式的优点和缺点

| 优点 | 说明 |
|------|------|
| 节省资源 | 避免重复创建昂贵对象 |
| 统一管理 | 全局配置、日志、缓存更集中 |
| 使用方便 | 任何地方都能拿到同一个实例 |

| 缺点 | 说明 |
|------|------|
| 全局状态 | 容易让代码产生隐藏依赖 |
| 测试困难 | 单例状态可能污染不同测试用例 |
| 并发风险 | 多线程下要考虑线程安全 |
| 扩展受限 | 子类化和替换实现不如普通对象灵活 |


### 20.9 面试常问点

| 问题 | 回答 |
|------|------|
| 单例模式是什么？ | 保证一个类只有一个实例，并提供全局访问点 |
| Python 中如何实现单例？ | `__new__`、装饰器、模块、元类 |
| 为什么重写 `__new__`？ | 因为 `__new__` 控制对象创建，`__init__` 只负责初始化 |
| 单例有什么风险？ | 全局状态、测试污染、线程安全问题 |
| Python 最简单的单例是什么？ | 模块级对象 |


### 20.10 一句话总结

单例模式用于保证某个类只创建一个对象。Python 中最常见的实现方式是重写 `__new__`，最自然的方式是使用模块级对象；如果在多线程环境中使用，要额外考虑线程安全。
