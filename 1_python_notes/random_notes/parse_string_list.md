### Convert a string of a list to a list

**reference:**
- [stackoverflow: how to convert string representation of list to a list](https://stackoverflow.com/questions/1894269/how-to-convert-string-representation-of-list-to-a-list)

**solution:**
- use  `json.loads(your_data) ` function in the `json` module to parse a stringified list of dictionaries
- **NOTE**: it works for `'["a","b"]'` but not for `"['a','b']"`


```python
import json
x = '[ "A","B","C" , " D"]'

json.loads(x)
```




    ['A', 'B', 'C', ' D']




```python
x = '["a","b"]' 
json.loads(x)    
```




    ['a', 'b']




```python
try:
    x = "['a','b']"
    json.loads(x) 
except:
    print('error')
```

    error
    
