```python
from pylab import mpl
#修改ubuntu字体用来兼容中文显示
zhfont = mpl.font_manager.FontProperties(fname='/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc')

#下面调用plt的时候，加上参数 fontproperties=zhfont
```



