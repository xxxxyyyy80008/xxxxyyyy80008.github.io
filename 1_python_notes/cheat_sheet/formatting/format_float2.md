---
title: Python float formatting 
---
***

### Python float formatting 


*Jul 8, 2022*

<h4>References:</h4>

- https://stackoverflow.com/questions/45310254/fixed-digits-after-decimal-with-f-strings
- https://docs.python.org/3/library/string.html#format-string-syntax
- https://peps.python.org/pep-0498/#format-specifiers
- https://cheatography.com/brianallan/cheat-sheets/python-f-strings-number-formatting/
- https://appdividend.com/2022/06/23/how-to-format-float-values-in-python/
    
<h4>Key notes:</h4>

- use [Format specifiers](https://peps.python.org/pep-0498/#format-specifiers) to format `float` numbers
>  `f'{value:{width}.{precision}}'`

- examples:
 - use `%` 
```python
x = 1234.5678
print("%.3f" % x)
```
```
    179.496
```
 - use `f`
```python
x = 1e6 + 0.12345
```
   -  add comma to large numbers
```python
print(f'{x:,.6f}')  #display 6 digits after the decimal point
print(f'{x:,.3f}')  #display 3 digits after the decimal point
```
```
    1,000,000.123450
    1,000,000.123
```
   -   no comma for large numbers
```python
print(f'{x:.6f}')  #display 6 digits after the decimal point
```
```
    1000000.123450
```
   -  pad (i.e. fixed width)
```python
print(f'{x:20,.3f}') 
```
```
           1,000,000.123
```


<h4>Python float format: Complete Table</h4>

The basic packages for data processing, mathematical and scitific calculations.



| Number | Format | Output | Description | 
| ------  | ------  | ------  | ------  |
| 3.1415926 | {:.2f} | 3.14 | Format float 2 decimal places | 
| 3.1415926 | {:+.2f} | +3.14 | Format float 2 decimal places with sign | 
| -1 | {:+.2f} | -1.00 | Format float 2 decimal places with sign | 
| 2.71828 | {:.0f} | 3 | Format float with no decimal places | 
| 4 | {:0>2d} | 04 | Pad number with zeros (left padding, width 2) | 
| 4 | {:x<4d} | 4xxx | Pad number with x’s (right padding, width 4) | 
| 10 | {:x<4d} | 10xx | Pad number with x’s (right padding, width 4) | 
| 1000000 | {:,} | 1,000,000 | Number format with comma separator | 
| 0.35 | {:.2%} | 35.00% | Format percentage | 
| 1000000000 | {:.2e} | 1.00e+09 | Exponent notation | 
| 11 | {:11d} | 11 | Right aligned (default, width 10) | 
| 11 | {:<11d} | 11 | Left aligned (width 10) | 
| 11 | {:^11d} | 11 | Center aligned (width 10) | 








