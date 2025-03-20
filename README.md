### 项目结构

---

```
├── train                      
│   ├──mask.py				
│   ├──model.py					# 上周的错误代码
│   ├──new_model.py     		# 模型代码
│   ├──PointerDecoder.py.py  	#指针解码器
│   ├──Position.py				#位置编码
│   └──config.py           		# 配置
```

### 整体思路

---

- 输入：形状[batch_size, 33 , d_model]
  - 33包括1个query和32个shot
- 模型结构：包括两部分——编码器和指针解码器：
  - Transformer encoder：使用标准的Transformer encoder编码器
  - 指针解码器：decoder部分和transformer decoder layer相似，但是在最后输出softmax时，需要添加mask，来mask掉之前已经生成的内容，保证不生成重复的id（mask未实现）
- 输出：一个不重复的k序列

